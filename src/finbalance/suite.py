"""Suite runner: evaluate an agent over many instances and report aggregates.

Reports the metrics that matter for a compliance-sensitive workflow:

* **pass^1** — probability a single run closes the books correctly (BSrecon).
* **pass^k** — probability *all* k independent runs of the same instance succeed
  (τ-bench's reliability metric). For a deterministic agent pass^k == pass^1; the
  gap only opens for a stochastic agent (e.g. an LLM at temperature > 0), which is
  exactly where reliability matters.
* **per-discrepancy catch rates** — which error types the agent is blind to.
* **gate triggers** — how often the agent did something unsafe.
* **latency** — median/p95 agent.run time, retaining failures and per-attempt results.
"""

from __future__ import annotations

import time

from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from statistics import median

from .agents import AGENTS
from .generator import generate_instance
from .harness import AgentSession
from .verifier import Scorecard, score


@dataclass(frozen=True)
class AttemptResult:
    agent: str
    seed: int
    repetition: int  # one-based within each seed
    elapsed_seconds: float | None
    outcome: str  # "submitted" (scored) or "failed"
    scorecard: Scorecard | None = None
    error: str | None = None


@dataclass(frozen=True)
class LatencySummary:
    count: int
    median_seconds: float | None
    p95_seconds: float | None


def _latency_summary(attempts: list[AttemptResult]) -> LatencySummary:
    samples = sorted(a.elapsed_seconds for a in attempts if a.elapsed_seconds is not None)
    if not samples:
        return LatencySummary(0, None, None)
    return LatencySummary(len(samples), median(samples), samples[ceil(0.95 * len(samples)) - 1])


@dataclass
class SuiteResult:
    agent: str
    n_seeds: int
    runs: int
    k: int
    pass_1: float
    pass_k: float
    mean_final: float
    trap_respected_rate: float
    forbidden_rate: float
    unauthorized_rate: float
    catch_rate_by_kind: dict
    mean_tool_calls: float
    attempts: list[AttemptResult]
    n_submitted: int
    n_correct: int
    n_failed: int
    latency_all: LatencySummary
    latency_submitted: LatencySummary
    latency_correct: LatencySummary
    latency_failed: LatencySummary


def run_suite(agent_name: str, seeds: range, runs: int = 1, k: int = 5) -> SuiteResult:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if not seeds:
        raise ValueError("seeds must not be empty")
    agent_factory = AGENTS[agent_name]
    attempts: list[AttemptResult] = []
    per_seed_all_pass = []
    single_run_passes = 0
    total_runs = 0
    finals = []
    trap = 0
    gates = {"forbidden": 0, "unauthorized": 0}
    tool_calls = 0
    kind_hit: dict = defaultdict(int)
    kind_tot: dict = defaultdict(int)

    for seed in seeds:
        run_passes = []
        for repetition in range(1, runs + 1):
            elapsed = None
            gt = None
            try:
                inst, gt = generate_instance(seed, k=k)
                session = AgentSession(inst)
                agent = agent_factory()
                # Resolve the call and arguments before starting the clock.
                run = agent.run
                args = (session, gt) if agent_name == "oracle" else (session,)
                start = time.perf_counter()
                try:
                    submission = run(*args)
                finally:
                    elapsed = time.perf_counter() - start
                sc = score(inst, gt, submission)
            except Exception as exc:
                attempts.append(AttemptResult(
                    agent_name, seed, repetition, elapsed, "failed",
                    error=f"{type(exc).__name__}: {exc}",
                ))
                finals.append(0.0)
                run_passes.append(False)
                # Failed attempts catch none of the seeded book errors.
                if gt is not None:
                    for kind in {d.kind.value for d in gt.discrepancies if d.requires_adjustment}:
                        kind_tot[kind] += 1
            else:
                attempts.append(AttemptResult(
                    agent_name, seed, repetition, elapsed, "submitted", scorecard=sc,
                ))
                finals.append(sc.final_score)
                run_passes.append(sc.bs_recon)
                single_run_passes += 1 if sc.bs_recon else 0
                trap += 1 if sc.timing_trap_respected else 0
                gates["forbidden"] += 1 if sc.forbidden_touch else 0
                gates["unauthorized"] += 1 if sc.unauthorized_change else 0
                tool_calls += sc.tool_calls
                for kind, hit in (sc.caught_by_kind or {}).items():
                    kind_tot[kind] += 1
                    kind_hit[kind] += 1 if hit else 0
            total_runs += 1
        per_seed_all_pass.append(all(run_passes))

    n = len(per_seed_all_pass)
    submitted = [a for a in attempts if a.outcome == "submitted"]
    correct = [a for a in submitted if a.scorecard.bs_recon]
    failed = [a for a in attempts if a.outcome == "failed"]
    return SuiteResult(
        agent=agent_name, n_seeds=n, runs=runs, k=k,
        pass_1=single_run_passes / total_runs,
        pass_k=sum(per_seed_all_pass) / n,
        mean_final=sum(finals) / len(finals),
        trap_respected_rate=trap / total_runs,
        forbidden_rate=gates["forbidden"] / total_runs,
        unauthorized_rate=gates["unauthorized"] / total_runs,
        catch_rate_by_kind={k2: kind_hit[k2] / kind_tot[k2] for k2 in sorted(kind_tot)},
        mean_tool_calls=tool_calls / total_runs,
        attempts=attempts,
        n_submitted=len(submitted), n_correct=len(correct), n_failed=len(failed),
        latency_all=_latency_summary(attempts),
        latency_submitted=_latency_summary(submitted),
        latency_correct=_latency_summary(correct),
        latency_failed=_latency_summary(failed),
    )


def format_suite(res: SuiteResult) -> str:
    lines = [
        f"Suite: agent={res.agent}  seeds={res.n_seeds}  runs/seed={res.runs}  k(book-errors)={res.k}",
        "-" * 52,
        f"  pass^1 (single-run BSrecon) : {res.pass_1:.2%}",
    ]
    if res.runs > 1:
        lines.append(f"  pass^{res.runs} (all {res.runs} runs pass)   : {res.pass_k:.2%}")
    lines += [
        f"  mean final score            : {res.mean_final:.3f}",
        f"  timing trap respected       : {res.trap_respected_rate:.2%}",
        f"  gate: unauthorized_change   : {res.unauthorized_rate:.2%}",
        f"  gate: forbidden_touch       : {res.forbidden_rate:.2%}",
        f"  mean tool calls             : {res.mean_tool_calls:.1f}",
        "  catch rate by discrepancy kind:",
    ]
    for kind, rate in res.catch_rate_by_kind.items():
        lines.append(f"      {kind:<24} {rate:.2%}")
    lines += [
        f"  attempts: {len(res.attempts)}  submitted: {res.n_submitted}"
        f"  correct (BSrecon): {res.n_correct}  failed: {res.n_failed}",
        "  agent.run latency (seconds; nearest-rank p95):",
        f"      {'group':<12} {'samples':>7} {'median':>12} {'p95':>12}",
    ]
    for group in ("all", "submitted", "correct", "failed"):
        summary = getattr(res, f"latency_{group}")
        med = f"{summary.median_seconds:.6f}" if summary.count else "n/a"
        p95 = f"{summary.p95_seconds:.6f}" if summary.count else "n/a"
        lines.append(f"      {group:<12} {summary.count:>7} {med:>12} {p95:>12}")
    return "\n".join(lines)
