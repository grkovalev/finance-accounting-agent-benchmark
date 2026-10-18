"""Exercise the real tool loop with scripted models, never a live provider."""

import os
import subprocess
import sys

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai import models
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from finbalance import llm_agent
from finbalance.coa import build_chart_of_accounts
from finbalance.generator import generate_instance
from finbalance.harness import AgentSession
from finbalance.llm_agent import LlmAgent
from finbalance.suite import run_suite


@pytest.fixture(autouse=True)
def no_live_models(monkeypatch):
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)


def call(name, **args):
    return ModelResponse(parts=[ToolCallPart(name, args)])


def returns(messages):
    return [p for message in messages for p in message.parts if isinstance(p, ToolReturnPart)]


def scripted(responses):
    steps = iter(responses)
    return LlmAgent(model=FunctionModel(lambda messages, info: next(steps)(messages, info)))


def test_reads_posts_recovers_and_submits():
    session = AgentSession(generate_instance(1)[0])
    before = session.get_trial_balance()
    session.trace.clear()
    seen_tools = set()

    def read(messages, info):
        seen_tools.update(t.name for t in info.function_tools + info.output_tools)
        prompt = "\n".join(m.instructions or "" for m in messages if hasattr(m, "instructions"))
        for account in build_chart_of_accounts():
            assert account.code in prompt
            assert account.name in prompt
            assert account.normal_side.value in prompt
        return call("query_ledger")

    def docs(messages, info):
        assert returns(messages)[-1].content
        return call("list_documents")

    def policy(messages, info):
        assert "policy" in returns(messages)[-1].content
        return call("read_document", doc_id="policy")

    def rejected(messages, info):
        assert "period_end" in returns(messages)[-1].content
        return call("post_journal_entry", entry_id="bad", date="2025-01-31", memo="bad",
                    lines=[{"account": "5100", "debit": 123}])

    def accepted(messages, info):
        assert returns(messages)[-1].content["ok"] is False
        assert session.get_trial_balance() == before
        return call("post_journal_entry", entry_id="good", date="2025-01-31", memo="fee",
                    lines=[{"account": "5100", "debit": 123}, {"account": "1000", "credit": 123}])

    def balance(messages, info):
        assert returns(messages)[-1].content["ok"] is True
        return call("get_trial_balance")

    def submit(messages, info):
        return call("submit_reconciliation", reported_balance_sheet=returns(messages)[-1].content)

    result = scripted([read, docs, policy, rejected, accepted, balance, submit]).run(session)
    assert seen_tools == {"query_ledger", "list_documents", "read_document", "get_trial_balance",
                          "post_journal_entry", "submit_reconciliation"}
    assert [entry.entry_id for entry in result.entries] == ["good"]
    assert result.reported_bs["1000"] == before["1000"] - 123
    assert result.reported_bs["5100"] == before["5100"] + 123
    assert result.entries[0].source == "AGENT"


def test_malformed_lines_can_be_corrected():
    def corrected(messages, info):
        assert any(isinstance(p, RetryPromptPart) for m in messages for p in m.parts)
        return call("submit_reconciliation", reported_balance_sheet={})

    agent = scripted([
        lambda m, i: call("post_journal_entry", entry_id="bad", date="2025-01-31", memo="bad",
                          lines=[{"account": "1000", "debit": -1}]),
        corrected,
    ])
    assert not agent.run(AgentSession(generate_instance(1)[0])).entries


def test_unknown_account_does_not_partially_post():
    session = AgentSession(generate_instance(1)[0])
    before = session.get_trial_balance()

    def submit(messages, info):
        assert returns(messages)[-1].content["ok"] is False
        assert session.get_trial_balance() == before
        return call("submit_reconciliation", reported_balance_sheet=before)

    result = scripted([
        lambda m, i: call("post_journal_entry", entry_id="bad", date="2025-01-31", memo="bad",
                          lines=[{"account": "1000", "debit": 5}, {"account": "unknown", "credit": 5}]),
        submit,
    ]).run(session)
    assert not result.entries


def test_submission_is_terminal():
    agent = scripted([lambda m, i: ModelResponse(parts=[
        ToolCallPart("submit_reconciliation", {"reported_balance_sheet": {}}),
        ToolCallPart("post_journal_entry", {"entry_id": "late", "date": "2025-01-31", "memo": "late",
                     "lines": [{"account": "1000", "debit": 1}, {"account": "5100", "credit": 1}]}),
    ])])
    session = AgentSession(generate_instance(1)[0])
    before = session.get_trial_balance()
    assert not agent.run(session).entries
    assert session.get_trial_balance() == before


def test_completion_without_submission_is_retained_as_failure(monkeypatch):
    from finbalance.agents import AGENTS
    monkeypatch.setitem(AGENTS, "llm", lambda: scripted([
        lambda m, i: ModelResponse(parts=[TextPart("Done")]),
    ]))
    result = run_suite("llm", range(1), runs=1)
    assert result.n_failed == 1
    assert "without calling submit_reconciliation" in result.attempts[0].error
    assert result.latency_failed.count == 1


def test_request_limit(monkeypatch):
    monkeypatch.setattr(llm_agent, "REQUEST_LIMIT", 2)
    agent = LlmAgent(model=FunctionModel(lambda m, i: call("get_trial_balance")))
    with pytest.raises(UsageLimitExceeded):
        agent.run(AgentSession(generate_instance(1)[0]))


def test_provider_error_propagates():
    def fail(messages, info):
        raise RuntimeError("provider unavailable")
    with pytest.raises(RuntimeError, match="provider unavailable"):
        LlmAgent(model=FunctionModel(fail)).run(AgentSession(generate_instance(1)[0]))


def test_env_file_and_environment_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(
        "LLM_BASE_URL=https://example.test/v1\nLLM_API_KEY=test-secret\nLLM_MODEL=file-model\n"
    )
    monkeypatch.setenv("LLM_MODEL", "override-model")
    captured = {}

    class Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_agent, "AsyncOpenAI", Client)
    monkeypatch.setattr(llm_agent, "OpenAIProvider", lambda **kwargs: None)
    monkeypatch.setattr(llm_agent, "OpenAIChatModel", lambda name, **kwargs: (
        captured.update(model=name) or FunctionModel(lambda m, i: call("get_trial_balance"))
    ))
    LlmAgent()
    assert captured == {"base_url": "https://example.test/v1", "api_key": "test-secret",
                        "model": "override-model", "timeout": 60, "max_retries": 0}


def test_missing_configuration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="Copy .env_example") as error:
        LlmAgent()
    assert all(name in str(error.value) for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"))


def test_reference_agents_without_optional_dependencies():
    code = '''
import sys
class BlockOptional:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'pydantic_ai', 'dotenv', 'openai'}:
            raise ImportError('blocked optional dependency')
sys.meta_path.insert(0, BlockOptional())
from finbalance.cli import main
assert main(['demo', '--seed', '1']) == 0
from finbalance.agents import AGENTS
try:
    AGENTS['llm']()
except ImportError as exc:
    assert 'uv sync --extra llm' in str(exc)
else:
    raise AssertionError('expected missing dependency error')
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": "src"})
    assert result.returncode == 0, result.stderr


def test_cli_reports_setup_error(monkeypatch, capsys):
    from finbalance.agents import AGENTS
    from finbalance.cli import main

    def fail():
        raise ValueError("Missing LLM configuration: LLM_MODEL")

    monkeypatch.setitem(AGENTS, "llm", fail)
    assert main(["suite", "--agent", "llm", "--seeds", "0:1", "--runs", "1"]) == 0
    output = capsys.readouterr().out
    assert "failed: 1" in output
    assert "first failure (seed=0, repetition=1)" in output
    assert "Missing LLM configuration: LLM_MODEL" in output


def test_cli_scores_mocked_llm_submission(monkeypatch, capsys):
    from finbalance.agents import AGENTS
    from finbalance.cli import main
    monkeypatch.setitem(AGENTS, "llm", lambda: scripted([
        lambda m, i: call("get_trial_balance"),
        lambda m, i: call("submit_reconciliation", reported_balance_sheet=returns(m)[-1].content),
    ]))
    main(["suite", "--agent", "llm", "--seeds", "0:1", "--runs", "1"])
    output = capsys.readouterr().out
    assert "submitted: 1" in output
    assert "failed: 0" in output
