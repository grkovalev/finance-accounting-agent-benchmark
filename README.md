# finbalance — a month-end reconciliation / close agent benchmark

**What it is.** A benchmark that tells you whether an LLM agent is any good at a
real, tool-using, long-horizon finance workflow: doing a **month-end
reconciliation / close**. The agent is handed a company's books and external
statements, must find and fix seeded errors with adjusting journal entries, and
produce a correct, balanced set of financials — acting entirely through a tool
interface.

The design goal is an **objective, programmatic verifier** — no LLM judge in the
core score. This is possible because accounting has built-in consistency laws
(debits = credits, the balance sheet balances, the ledger ties to the statement),
so "the right answer" is *computable*, not a matter of opinion.

The full design writeup is in [`proposal/proposal.md`](proposal/proposal.md).

## Quickstart (one command, zero dependencies)

The reference implementation is pure Python standard library (Python ≥ 3.10).

```bash
PYTHONPATH=src python -m finbalance.cli demo --seed 1
```

You'll see a fresh synthetic company-month generated and scored by four reference
agents that calibrate the metric — a perfect **Oracle**, a rule-based
**Baseline** (tools only, no ground truth), a **Sloppy** agent that trips a gate,
and a do-nothing **Null**:

```
Instance seed-0001 | 14 accounts | 13 GL entries | 6 discrepancies seeded (5 book errors + 1 timing trap)

metric                       ORACLE   BASELINE     SLOPPY       NULL
--------------------------------------------------------------------
BSexact                        PASS       FAIL       FAIL       FAIL
BSrecon                        PASS       FAIL       FAIL       FAIL
accounts_reconciled             1.0        1.0       0.25        0.5
discrepancies_caught            1.0        0.8        0.0        0.0
timing_trap_respected           yes        yes        yes        yes
forbidden_touch                  no         no         no         no
unauthorized_change              no         no        yes         no
tool_calls                        5          7          4          1
--------------------------------------------------------------------
FINAL SCORE                    1.00       0.35       0.00       0.07
```

The spread (1.00 / 0.35 / 0.00 / 0.07) shows the metric has real resolution: the
Baseline does honest work but is blind to policy-driven accruals; the Sloppy agent
earns a non-trivial raw score yet is **gated to zero** for moving an account it had
no business touching.

**Evaluate an agent over many instances** — pass^k reliability + per-error-type
catch rates:

```bash
PYTHONPATH=src python -m finbalance.cli suite --agent baseline --seeds 0:50
```

The suite also reports median/p95 latency and sample counts for all, submitted,
correct (`BSrecon`), and failed runs. Only `agent.run()` is timed with
`time.perf_counter()`; generation, setup, and scoring are excluded. P95 uses
nearest rank: sorted sample at one-based position `ceil(0.95 * n)`.
Per-attempt results retain scores and exceptions; failures count as unsuccessful,
and setup failures have no latency sample.

Compare agents in separate suite runs with identical `--seeds`, `--k`, and
`--runs`, for example:

```bash
PYTHONPATH=src python -m finbalance.cli suite --agent oracle --seeds 0:20 --k 5 --runs 3
PYTHONPATH=src python -m finbalance.cli suite --agent baseline --seeds 0:20 --k 5 --runs 3
```

**Run a multi-month campaign** — the long-horizon dimension, where the agent's own
carried-forward balances make errors compound:

```bash
PYTHONPATH=src python -m finbalance.cli campaign --agent baseline --seed 1 --months 4
# 1.00 → 1.00 → 0.34 → 0.34 : misses an accrual, drifts, and never recovers
```

Inspect the generated data (committed under [`samples/`](samples/)):

```bash
PYTHONPATH=src python -m finbalance.cli generate --seed 1 --out /tmp/inst --with-solution
# instance.json  = what the agent sees   |   solution.json = hidden answer key
```

Run the tests (needs `pytest`):

```bash
python -m venv .venv && ./.venv/bin/pip install pytest
./.venv/bin/python -m pytest -q
```

## How it works (three pieces)

1. **Procedural generator** (`generator.py`, `discrepancies.py`) — emits one
   synthetic company-month per seed by building two worlds: what a perfect
   accountant *should* have booked, and what the company *actually* booked (with
   errors). Because the error lives in the **documents** — a literally duplicated
   ledger entry, a transposed amount the bank contradicts, a fee on the statement
   that never hit the books — a real agent can discover it, not just infer it from
   balance math. Seeded errors: unrecorded bank fee, transposition, misclassification,
   duplicate entry, missing accrual, **+ a timing-difference trap that must NOT be
   adjusted**. Every instance passes a generation-time self-check that replaying the
   gold fixes reproduces the clean balances — so **ground truth is correct by
   construction** (proven across 100+ seeds in the tests).
2. **Deterministic ledger-replay verifier** (`ledger.py`, `verifier.py`) — a
   double-entry engine replays the agent's adjusting entries from the opening TB and
   scores the resulting balances against gold: `BSrecon` (replay matches gold) and
   `BSexact` (self-report matches gold), plus per-account and per-discrepancy partial
   credit, an independent timing-trap check, and **hard gates** (touching a protected
   equity account, or moving an account with no legitimate reason, zeroes the score).
3. **Tool-using harness + calibration agents** (`harness.py`, `agents.py`) — the
   agent acts through `query_ledger`, `read_document`, `get_trial_balance`,
   `post_journal_entry` (rejected if unbalanced), `submit_reconciliation`; every call
   is traced. Four reference agents bound and calibrate the metric: `OracleAgent`
   (1.0 ceiling), `BaselineAgent` (an honest rule-based reconciler using only the
   tools — also the executable proof that instances are solvable), `SloppyAgent`
   (trips a gate), `NullAgent` (floor).

## What's simplified (honest scope)

This is a **reference implementation** to prove the design is real and runnable,
not a production benchmark. Deliberate simplifications:

- **One synthetic company**, ~14-account chart of accounts, 6 discrepancy types.
  A full version scales breadth (more accounts, multi-entity) and error variety.
- **One gold balance vector per instance.** The chart of accounts is kept
  unambiguous (each error maps to exactly one correct account) so there is a single
  correct answer. Scoring is on *balances*, not entry text, so alternative
  journal-entry orderings/codings already pass; the verifier exposes a documented
  seam to accept a *set* of gold vectors when genuine account-choice ambiguity is
  wanted.
- **Statements are structured**, not scanned PDFs — document/OCR robustness is a
  separable perception axis, deliberately out of scope.
- **Synthetic only.** Real matched reconciliation data (a ledger paired with the
  bank statement it ties to, plus reconciling items) is not publicly available — it
  is PII- and audit-confidential — so synthetic generation is the only
  contamination-safe way to get matched inputs with trustworthy labels. See the
  proposal's *Ground Truth & Data* section.
