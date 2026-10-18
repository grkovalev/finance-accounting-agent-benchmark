"""Suite timing uses a fake clock; no providers or wall-clock thresholds."""

from statistics import mean

import pytest

from finbalance import suite
from finbalance.agents import AGENTS, NullAgent, OracleAgent
from finbalance.generator import generate_instance
from finbalance.harness import AgentSession
from finbalance.verifier import score


class Clock:
    def __init__(self):
        self.now = 0.0

    def advance(self, seconds):
        self.now += seconds

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(suite.time, "perf_counter", clock)
    return clock


def test_only_run_is_timed_and_each_attempt_has_fresh_setup(monkeypatch, clock):
    sessions = []
    agents = []
    generated = []

    def generate(seed, **kwargs):
        clock.advance(100)
        generated.append((seed, kwargs))
        return generate_instance(seed, **kwargs)

    def session_factory(inst):
        clock.advance(200)
        session = AgentSession(inst)
        sessions.append(session)
        return session

    class Agent:
        def __init__(self):
            clock.advance(300)
            agents.append(self)

        def run(self, session):
            clock.advance(2)
            return NullAgent().run(session)

    def scoring(*args):
        clock.advance(400)
        return score(*args)

    monkeypatch.setattr(suite, "generate_instance", generate)
    monkeypatch.setattr(suite, "AgentSession", session_factory)
    monkeypatch.setattr(suite, "score", scoring)
    monkeypatch.setitem(suite.AGENTS, "controlled", Agent)
    res = suite.run_suite("controlled", range(3, 5), runs=2, k=3)
    assert [(a.agent, a.seed, a.repetition, a.elapsed_seconds, a.outcome) for a in res.attempts] == [
        ("controlled", seed, repetition, 2, "submitted")
        for seed in (3, 4) for repetition in (1, 2)
    ]
    assert generated == [(seed, {"k": 3}) for seed in (3, 3, 4, 4)]
    assert len({id(s) for s in sessions}) == len({id(a) for a in agents}) == 4
    assert all(a.scorecard is not None and a.error is None for a in res.attempts)
    assert res.latency_submitted == suite.LatencySummary(4, 2, 2)


def test_mixed_outcomes_keep_failures_in_quality_and_latency(monkeypatch, clock):
    actions = iter(["correct", "incorrect", "error", "timeout"])

    class Agent:
        def run(self, session, gt):
            action = next(actions)
            clock.advance({"correct": 1, "incorrect": 3, "error": 5, "timeout": 7}[action])
            if action == "error":
                raise RuntimeError("provider unavailable")
            if action == "timeout":
                raise TimeoutError("deadline exceeded")
            return (OracleAgent() if action == "correct" else NullAgent()).run(session, gt)

    monkeypatch.setitem(suite.AGENTS, "oracle", Agent)
    res = suite.run_suite("oracle", range(2), runs=2)
    assert (res.n_submitted, res.n_correct, res.n_failed) == (2, 1, 2)
    assert res.pass_1 == 0.25
    assert res.pass_k == 0
    assert res.mean_final == sum(a.scorecard.final_score for a in res.attempts[:2]) / 4
    assert res.trap_respected_rate == 0.5
    assert set(res.catch_rate_by_kind.values()) == {0.25}
    assert res.latency_all == suite.LatencySummary(4, 4, 7)
    assert res.latency_submitted == suite.LatencySummary(2, 2, 3)
    assert res.latency_correct == suite.LatencySummary(1, 1, 1)
    assert res.latency_failed == suite.LatencySummary(2, 6, 7)
    assert [a.error for a in res.attempts[2:]] == [
        "RuntimeError: provider unavailable", "TimeoutError: deadline exceeded",
    ]
    assert all(a.scorecard is None for a in res.attempts[2:])
    output = suite.format_suite(res)
    assert "submitted: 2  correct (BSrecon): 1  failed: 2" in output
    assert "nearest-rank p95" in output
    assert "7.000000" in output


def test_setup_failure_has_no_latency_and_continues(monkeypatch, clock):
    constructions = []

    def factory():
        constructions.append(True)
        clock.advance(100)
        if len(constructions) == 1:
            raise ValueError("setup failed")
        return NullAgent()

    monkeypatch.setitem(suite.AGENTS, "controlled", factory)
    res = suite.run_suite("controlled", range(2))
    assert len(res.attempts) == 2
    assert res.attempts[0].elapsed_seconds is None
    assert res.attempts[0].error == "ValueError: setup failed"
    assert res.latency_all.count == 1
    assert res.latency_failed == suite.LatencySummary(0, None, None)
    assert res.n_failed == res.n_submitted == 1
    assert "n/a" in suite.format_suite(res)


def test_scoring_failure_retains_run_time_and_continues(monkeypatch, clock):
    calls = []

    def scoring(*args):
        calls.append(True)
        clock.advance(50)
        if len(calls) == 1:
            raise ValueError("invalid submission")
        return score(*args)

    monkeypatch.setattr(suite, "score", scoring)
    res = suite.run_suite("null", range(2))
    assert [a.outcome for a in res.attempts] == ["failed", "submitted"]
    assert res.attempts[0].error == "ValueError: invalid submission"
    assert res.latency_all == suite.LatencySummary(2, 0, 0)


def test_p95_uses_nearest_rank_not_max_or_interpolation(monkeypatch, clock):
    durations = iter(range(1, 21))

    class Agent:
        def run(self, session):
            clock.advance(next(durations))
            return NullAgent().run(session)

    monkeypatch.setitem(suite.AGENTS, "controlled", Agent)
    res = suite.run_suite("controlled", range(20))
    assert res.latency_all == suite.LatencySummary(20, 10.5, 19)
    assert res.latency_correct == suite.LatencySummary(0, None, None)


@pytest.mark.parametrize("name", ["baseline", "sloppy", "null"])
def test_non_oracle_receives_only_session(monkeypatch, clock, name):
    class Agent:
        def run(self, session):  # passing a second argument would fail
            return NullAgent().run(session)

    monkeypatch.setitem(suite.AGENTS, name, Agent)
    res = suite.run_suite(name, range(1))
    assert res.n_submitted == 1
    assert res.n_failed == 0


def test_oracle_receives_ground_truth(clock):
    res = suite.run_suite("oracle", range(2), runs=2)
    assert res.n_correct == 4
    assert res.pass_1 == res.pass_k == res.mean_final == 1


@pytest.mark.parametrize("name", ["oracle", "baseline", "sloppy", "null"])
def test_existing_correctness_metrics_are_unchanged(name, clock):
    cards = []
    for seed in range(3):
        inst, gt = generate_instance(seed, k=5)
        cards.append(score(inst, gt, AGENTS[name]().run(AgentSession(inst), gt)))
    res = suite.run_suite(name, range(3), runs=2)
    assert res.pass_1 == res.pass_k == mean(c.bs_recon for c in cards)
    assert res.mean_final == pytest.approx(mean(c.final_score for c in cards))
    assert res.trap_respected_rate == mean(c.timing_trap_respected for c in cards)
    assert res.forbidden_rate == mean(c.forbidden_touch for c in cards)
    assert res.unauthorized_rate == mean(c.unauthorized_change for c in cards)
    assert res.mean_tool_calls == mean(c.tool_calls for c in cards)
    assert res.catch_rate_by_kind == {
        kind: mean(c.caught_by_kind[kind] for c in cards) for kind in cards[0].caught_by_kind
    }


@pytest.mark.parametrize("seeds,runs", [(range(0), 1), (range(1), 0), (range(1), -1)])
def test_empty_suite_is_rejected(seeds, runs):
    with pytest.raises(ValueError):
        suite.run_suite("null", seeds, runs=runs)
