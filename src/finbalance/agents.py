"""Reference stub agents that bound and calibrate the metric.

* :class:`OracleAgent` — reads ground truth, posts the gold adjustments. Ceiling: 1.0.
* :class:`NullAgent` — does nothing. Floor: low.
* :class:`BaselineAgent` — an honest rule-based reconciler using ONLY the session
  tools (never ground truth). It ties the ledger to the bank statement and applies
  mechanical fixes, but has a realistic blind spot (policy-driven accruals), so it
  lands mid-scale. It doubles as the executable proof that instances are solvable
  from the documents alone.
* :class:`SloppyAgent` — forces the books to tie by dumping the unreconciled
  difference into an untouched clearing account, tripping the ``unauthorized_change``
  gate. Demonstrates why the gates exist: decent raw score, final score 0.

A real LLM agent implements the same ``run(session, gt=None)`` protocol, using only
the tools on ``session``.
"""

from __future__ import annotations

from typing import Protocol

from . import coa as C
from .harness import AgentSession
from .model import GroundTruth, JournalEntry, JournalLine
from .verifier import Submission


class Agent(Protocol):
    def run(self, session: AgentSession, gt: "GroundTruth | None" = None) -> Submission: ...


def _je(eid: str, memo: str, *lines: JournalLine) -> JournalEntry:
    return JournalEntry(entry_id=eid, date="2025-01-31", memo=memo, lines=tuple(lines), source="AGENT")


class OracleAgent:
    def run(self, session: AgentSession, gt: "GroundTruth | None" = None) -> Submission:
        assert gt is not None, "OracleAgent requires ground truth"
        for entry in gt.gold_adjustments:
            session.post_journal_entry(entry)
        return session.submit_reconciliation(dict(gt.gold_final_balances))


class NullAgent:
    def run(self, session: AgentSession, gt: "GroundTruth | None" = None) -> Submission:
        return session.submit_reconciliation(session.get_trial_balance())


class BaselineAgent:
    """Mechanical ledger-to-bank reconciler. Catches cash/statement-driven errors
    and visible miscodings; deliberately does NOT perform period-end accruals."""

    def run(self, session: AgentSession, gt: "GroundTruth | None" = None) -> Submission:
        gl = session.query_ledger()
        by_id = {e["entry_id"]: e for e in gl}
        gl_ids = set(by_id)
        bank = session.read_document(f"stmt::bank::{C.CASH}")
        n = 0

        def nid() -> str:
            nonlocal n
            n += 1
            return f"adj-{n:02d}"

        def cash_of(entry) -> int:
            return sum(d - c for (a, d, c) in entry["lines"] if a == C.CASH)

        def noncash_account(entry) -> "str | None":
            for (a, d, c) in entry["lines"]:
                if a != C.CASH:
                    return a
            return None

        # (1) Unrecorded bank charges: a bank line with no matching ledger entry.
        for (_lid, _dt, _desc, amount, ext) in bank["lines"]:
            if ext not in gl_ids and amount < 0:
                fee = -amount
                session.post_journal_entry(_je(nid(), "Record bank service charge",
                                               JournalLine(C.BANK_FEES, debit=fee),
                                               JournalLine(C.CASH, credit=fee)))

        # (2) Transposition: a ledger cash entry whose amount disagrees with the bank.
        for (_lid, _dt, _desc, amount, ext) in bank["lines"]:
            if ext in by_id:
                delta = amount - cash_of(by_id[ext])
                if delta != 0:
                    counter = noncash_account(by_id[ext]) or C.REVENUE
                    if delta > 0:
                        session.post_journal_entry(_je(nid(), "Correct transposed payment",
                                                       JournalLine(C.CASH, debit=delta),
                                                       JournalLine(counter, credit=delta)))
                    else:
                        session.post_journal_entry(_je(nid(), "Correct transposed payment",
                                                       JournalLine(counter, debit=-delta),
                                                       JournalLine(C.CASH, credit=-delta)))

        # (3) Duplicate: identical (memo, lines) posted more than once → reverse extras.
        seen: dict = {}
        for e in gl:
            key = (e["memo"], tuple(tuple(l) for l in e["lines"]))
            seen[key] = seen.get(key, 0) + 1
        for (memo, lines), count in seen.items():
            for _ in range(count - 1):
                rev = [JournalLine(a, debit=c, credit=d) for (a, d, c) in lines]
                session.post_journal_entry(_je(nid(), f"Reverse duplicate: {memo}", *rev))

        # (4) Misclassification: a 'Bank service charge' debited to Operating Expenses.
        for e in gl:
            if e["memo"] == "Bank service charge":
                for (a, d, c) in e["lines"]:
                    if a == C.OP_EXPENSE and d:
                        session.post_journal_entry(_je(nid(), "Reclassify bank fee",
                                                       JournalLine(C.BANK_FEES, debit=d),
                                                       JournalLine(C.OP_EXPENSE, credit=d)))

        # (5) Period-end accruals: NOT performed — the baseline's blind spot.
        # (6) Deposits in transit (ledger cash entry absent from the bank) are
        #     correctly left alone.
        return session.submit_reconciliation(session.get_trial_balance())


class SloppyAgent:
    """Plugs the whole book-vs-bank cash difference into the credit-card clearing
    account — an account with no legitimate reason to move. Trips the gate."""

    def run(self, session: AgentSession, gt: "GroundTruth | None" = None) -> Submission:
        bank = session.read_document(f"stmt::bank::{C.CASH}")
        tb = session.get_trial_balance()
        diff = bank["closing_balance"] - tb.get(C.CASH, 0)
        if diff > 0:
            session.post_journal_entry(_je("plug", "Force cash to bank (plug)",
                                           JournalLine(C.CASH, debit=diff),
                                           JournalLine(C.CREDIT_CARD, credit=diff)))
        elif diff < 0:
            session.post_journal_entry(_je("plug", "Force cash to bank (plug)",
                                           JournalLine(C.CREDIT_CARD, debit=-diff),
                                           JournalLine(C.CASH, credit=-diff)))
        return session.submit_reconciliation(session.get_trial_balance())


# Registry so the CLI/suite can select an agent by name.
def _llm_agent():
    try:
        from .llm_agent import LlmAgent
    except ImportError as exc:
        raise ImportError("LLM agent dependencies are missing. Run: uv sync --extra llm") from exc
    return LlmAgent()


AGENTS = {
    "oracle": OracleAgent,
    "baseline": BaselineAgent,
    "sloppy": SloppyAgent,
    "null": NullAgent,
    "llm": _llm_agent,
}
