"""Exercise the shared tool loop, native JEV handoffs and actual-call budgets."""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("typesafe_sdk")
pytest.importorskip("pydantic_ai.models.typesafe")

from pydantic_ai import models
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.decision import UnfillableRoute
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider
from typesafe_sdk import ChoiceAnswer, SystemOneResponse, Usage

from finbalance import jev_agent
from finbalance.generator import generate_instance
from finbalance.harness import AgentSession
from finbalance.jev_agent import JevAgent, JevSettings
from finbalance.suite import run_suite


@pytest.fixture(autouse=True)
def offline_models(monkeypatch):
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)


def call(name, **args):
    return ModelResponse(parts=[ToolCallPart(name, args)])


def returns(messages):
    return [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]


def session():
    return AgentSession(generate_instance(1000)[0])


def handoff(messages, info):
    raise UnfillableRoute("jev-test", "post_journal_entry", .99)


def scripted(jev_steps, gpt_steps=()):
    jev, gpt = iter(jev_steps), iter(gpt_steps)
    fallback = FunctionModel(lambda m, i: next(gpt)(m, i)) if gpt_steps else None
    return JevAgent(model=FunctionModel(lambda m, i: next(jev)(m, i)), llm_model=fallback)


def post(messages, info):
    return call("post_journal_entry", entry_id="fee", date="2025-01-31", memo="Model-selected fee adjustment",
                lines=[{"account": "5100", "debit": 100}, {"account": "1000", "credit": 100}])


def submit(messages, info):
    balances = next(p.content for p in reversed(returns(messages)) if p.tool_name == "get_trial_balance")
    return call("submit_reconciliation", reported_balance_sheet=balances)


def test_handoff_returns_to_jev_with_tool_results_and_shared_history():
    s = session()
    def resumed(m, i):
        assert returns(m)[-1].content["ok"] is True
        assert returns(m)[-1].tool_name == "post_journal_entry"
        return call("get_trial_balance")
    agent = scripted([lambda m, i: call("list_documents"), lambda m, i: call("read_document", doc_id="policy"),
                      handoff, resumed, handoff], [post, submit])
    result = agent.run(s)
    assert [e.entry_id for e in result.entries] == ["fee"]
    assert result.reported_bs == s.get_trial_balance()
    assert agent.requests.jev == 5
    assert agent.requests.gpt == 2  # multiple GPT calls are allowed in the shared budget
    assert not hasattr(agent, "graph")


def test_closed_read_arguments_come_only_from_public_observed_data():
    s = session()
    def initial(m, i):
        assert s.trace == []
        assert not any(t.name == "read_document" for t in i.function_tools)
        query = next(t for t in i.function_tools if t.name == "query_ledger")
        assert "1000" in query.parameters_json_schema["properties"]["account"]["anyOf"][0]["enum"]
        return call("list_documents")
    def listed(m, i):
        read = next(t for t in i.function_tools if t.name == "read_document")
        assert read.parameters_json_schema["properties"]["doc_id"]["enum"] == returns(m)[-1].content
        return call("get_trial_balance")
    agent = scripted([initial, listed, submit])
    agent.run(s)
    assert agent.requests.gpt == 0


def test_backend_errors_do_not_silently_fallback():
    def outage(m, i):
        raise ModelHTTPError(503, "jev-test", "backend unavailable")
    def no_gpt(m, i):
        pytest.fail("An outage must not call GPT")
    agent = scripted([outage], [no_gpt])
    with pytest.raises(ModelHTTPError):
        agent.run(session())
    assert agent.requests.jev == 1
    assert agent.requests.gpt == 0


def test_shared_actual_call_budget_counts_handoff_and_gpt(monkeypatch):
    monkeypatch.setattr(jev_agent, "REQUEST_LIMIT", 2)
    agent = scripted([handoff, handoff], [lambda m, i: call("get_trial_balance")])
    with pytest.raises(UsageLimitExceeded, match="JEV=1, GPT=1"):
        agent.run(session())


def test_no_fallback_when_disabled():
    def no_gpt(m, i):
        pytest.fail("Fallback disabled")
    agent = JevAgent(model=FunctionModel(handoff), llm_model=FunctionModel(no_gpt), settings=JevSettings(allow_llm=False))
    with pytest.raises(UnfillableRoute):
        agent.run(session())
    assert agent.requests.gpt == 0


def test_submission_is_terminal_even_for_fallback():
    def terminal(m, i):
        return ModelResponse(parts=[ToolCallPart("submit_reconciliation", {"reported_balance_sheet": {}}),
                                    post(m, i).parts[0]])
    agent = scripted([handoff], [terminal])
    assert not agent.run(session()).entries


def test_failed_gpt_entry_can_be_corrected_through_normal_tool_results():
    def bad(m, i):
        return call("post_journal_entry", entry_id="bad", date="2025-01-31", memo="bad",
                    lines=[{"account": "5100", "debit": 100}])
    def correct(m, i):
        assert returns(m)[-1].content["ok"] is False
        return post(m, i)
    agent = scripted([handoff, handoff, lambda m, i: call("get_trial_balance"), handoff], [bad, correct, submit])
    assert [e.entry_id for e in agent.run(session()).entries] == ["fee"]


def test_completion_without_submission_is_failure(monkeypatch):
    from finbalance.agents import AGENTS
    monkeypatch.setitem(AGENTS, "jev", lambda: scripted([handoff], [lambda m, i: call("finish_without_submission", response="Done")]))
    result = run_suite("jev", range(1000, 1001))
    assert result.n_failed == 1
    assert "without calling submit_reconciliation" in result.attempts[0].error


def test_fresh_counters_for_each_run():
    agent = scripted([lambda m, i: call("get_trial_balance"), submit,
                      lambda m, i: call("get_trial_balance"), submit])
    agent.run(session())
    first = agent.requests
    agent.run(session())
    assert agent.requests is not first
    assert agent.requests.jev == 2


def test_run_deadline_stops_without_submission(monkeypatch):
    async def slow(m, i):
        await asyncio.sleep(1)
    monkeypatch.setattr(jev_agent, "RUN_TIMEOUT_SECONDS", .01)
    with pytest.raises(TimeoutError):
        JevAgent(model=FunctionModel(slow)).run(session())


def test_env_configuration_without_override_or_obsolete_limits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("TYPESAFE_API_KEY", "TYPESAFE_BASE_URL", "TYPESAFE_DEFAULT_MODEL", "JEV_ALLOW_LLM_ESCALATION"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text("TYPESAFE_API_KEY=test-secret\nTYPESAFE_DEFAULT_MODEL=file-model\nJEV_ALLOW_LLM_ESCALATION=false\n")
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "env-model")
    monkeypatch.setenv("JEV_MAX_LLM_REQUESTS", "1")  # former interpretation cap no longer applies
    settings = JevSettings.from_env()
    assert settings.model == "env-model"
    assert settings.api_key == "test-secret"
    assert "test-secret" not in repr(settings)
    assert not settings.allow_llm
    monkeypatch.setenv("JEV_ALLOW_LLM_ESCALATION", "invalid")
    with pytest.raises(ValueError, match="JEV_ALLOW_LLM_ESCALATION"):
        JevSettings.from_env()
    monkeypatch.delenv("TYPESAFE_API_KEY")
    (tmp_path / ".env").write_text("")
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        JevSettings.from_env()


def test_disabled_fallback_needs_no_llm_configuration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    agent = JevAgent(model=FunctionModel(lambda m, i: call("submit_reconciliation", reported_balance_sheet={})),
                     settings=JevSettings(allow_llm=False))
    agent.run(session())


def native_client(routes):
    routes = iter(routes)
    class Client:
        _config = SimpleNamespace(base_url="https://example.test")
        calls = 0
        async def system_one(self, state, questions, **kwargs):
            self.calls += 1
            chosen = next(routes)
            answers = {}
            for name, question in questions.items():
                if name == "route":
                    value = chosen
                elif name.endswith("doc_id"):
                    value = "policy"
                elif name.endswith("account") and "none" in question.criteria:
                    value = "none"
                else:
                    value = next(iter(question.criteria))
                assert value in question.criteria
                answers[name] = ChoiceAnswer(choice=value, confidence=.99,
                    probabilities={k: float(k == value) for k in question.criteria})
            return SystemOneResponse(model="jev-test-pinned", usage=Usage(input_tokens=10, output_tokens=2), answers=answers)
    return Client()


def test_real_typesafe_model_routes_reads_and_native_gpt_handoffs(monkeypatch):
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", True)
    client = native_client(["list_documents", "read_document", "query_ledger", "get_trial_balance",
                            "post_journal_entry", "submit_reconciliation", "submit_reconciliation"])
    gpt = iter([post, lambda m, i: call("get_trial_balance"), submit])
    agent = JevAgent(model=TypeSafeModel("jev-test", provider=TypeSafeProvider(typesafe_client=client)),
                     llm_model=FunctionModel(lambda m, i: next(gpt)(m, i)))
    s = session()
    result = agent.run(s)
    assert [e.entry_id for e in result.entries] == ["fee"]
    assert result.reported_bs == s.get_trial_balance()
    assert agent.requests.jev == client.calls == 7
    assert agent.requests.gpt == 3


def test_native_handoff_cannot_bypass_the_actual_call_budget(monkeypatch):
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", True)
    monkeypatch.setattr(jev_agent, "REQUEST_LIMIT", 1)
    client = native_client(["submit_reconciliation"])
    agent = JevAgent(model=TypeSafeModel("jev-test", provider=TypeSafeProvider(typesafe_client=client)),
                     llm_model=FunctionModel(submit))
    s = session()
    with pytest.raises(UsageLimitExceeded, match="JEV=1, GPT=0"):
        agent.run(s)
    assert s.trace == []
    assert client.calls == 1
