"""Optional Pydantic AI adapter; all accounting decisions belong to the model."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import asdict

from openai import AsyncOpenAI
from pydantic import StrictInt
from pydantic_ai import Agent, RunContext, Tool, ToolOutput
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

from .coa import build_chart_of_accounts
from .harness import AgentSession
from .model import JournalEntry, JournalLine
from .model_support import chart_context, llm_configuration, make_llm_model
from .verifier import Submission

REQUEST_LIMIT = 40
REQUEST_TIMEOUT_SECONDS = 60


def _prompt() -> str:
    return """Reconcile this synthetic company-month using only the supplied tools.
Read the ledger, external statements, and policy before deciding on adjustments.
Amounts are integer cents. Journal lines use nonnegative debit or credit amounts,
never both. Balances are signed natural balances: debits increase debit-normal
accounts and credits increase credit-normal accounts, including contra accounts.
The working trial balance already includes period GL activity; do not post it again.
Find supported book errors and post balanced adjusting entries dated at period end.
Respect protected accounts and policy. Do not plug unexplained differences or adjust
for timing differences such as deposits in transit and outstanding checks.
A rejected entry has not been accepted; use its reason to reconsider your entry.
Use unique entry IDs. Read the final working trial balance and submit the complete
account-code-to-natural-balance mapping with submit_reconciliation. Submit in a
separate response after all other tool calls; submission ends the run.
Public fixed chart of accounts:
""" + chart_context()


def query_ledger(ctx: RunContext[AgentSession], account: str | None = None) -> list[dict]:
    """Read posted general-ledger entries, optionally filtered by account code."""
    return ctx.deps.query_ledger(account)


def list_documents(ctx: RunContext[AgentSession]) -> list[str]:
    """List available statement and policy document IDs."""
    return ctx.deps.list_documents()


def read_document(ctx: RunContext[AgentSession], doc_id: str) -> dict:
    """Read an external statement or the policy sheet by document ID."""
    try:
        return ctx.deps.read_document(doc_id)
    except KeyError:
        return {"ok": False, "reason": f"Unknown document: {doc_id}"}


def get_trial_balance(ctx: RunContext[AgentSession]) -> dict[str, int]:
    """Read current signed natural balances, including accepted adjustments."""
    return ctx.deps.get_trial_balance()


def post_journal_entry(
    ctx: RunContext[AgentSession], entry_id: str, date: str, memo: str,
    lines: list[JournalLine],
) -> dict:
    """Post a balanced adjusting entry; return acceptance or the rejection reason."""
    entry = JournalEntry(entry_id, date, memo, tuple(lines), source="AGENT")
    # The fixed public chart also lets us reject unknown codes before the ledger
    # starts applying lines (its validation and posting happen in one pass).
    codes = {a.code for a in build_chart_of_accounts()}
    if any(line.account not in codes for line in lines):
        return {"ok": False, "entry_id": entry_id, "reason": "Unknown account code"}
    return asdict(ctx.deps.post_journal_entry(entry))


def submit_reconciliation(
    ctx: RunContext[AgentSession], reported_balance_sheet: dict[str, StrictInt],
) -> Submission:
    """Finish reconciliation with the complete final signed natural balances."""
    return ctx.deps.submit_reconciliation(reported_balance_sheet)


def build_agent(model: Model, *, capabilities=(), text_output=str) -> Agent:
    """Shared prompt, tool contracts and termination behavior for both model adapters."""
    return Agent(
        model, deps_type=AgentSession, instructions=_prompt(), capabilities=capabilities,
        tools=[Tool(fn, sequential=True) for fn in (
            query_ledger, list_documents, read_document, get_trial_balance,
            post_journal_entry,
        )],
        output_type=[ToolOutput(submit_reconciliation, name="submit_reconciliation"), text_output],
        end_strategy="early", retries=2,
        model_settings={"timeout": REQUEST_TIMEOUT_SECONDS, "parallel_tool_calls": False},
    )


class LlmAgent:
    def __init__(self, *, model: Model | None = None):
        self._client = None
        if model is None:
            model, client = make_llm_model(
                llm_configuration(), REQUEST_TIMEOUT_SECONDS, client_factory=AsyncOpenAI,
                provider_factory=OpenAIProvider, model_factory=OpenAIChatModel,
            )
            self._client = client
        self._agent = build_agent(model)

    def run(self, session: AgentSession) -> Submission:
        return asyncio.run(self._run(session))

    async def _run(self, session: AgentSession) -> Submission:
        async with AsyncExitStack() as stack:
            if self._client is not None:
                await stack.enter_async_context(self._client)
            result = await self._agent.run(
                "Reconcile the company-month and submit your final balances.",
                deps=session, usage_limits=UsageLimits(request_limit=REQUEST_LIMIT),
            )
        if not isinstance(result.output, Submission):
            raise RuntimeError("LLM finished without calling submit_reconciliation")
        return result.output
