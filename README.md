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

The reference implementation is pure Python standard library (Python ≥ 3.11).

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

**Run the LLM agent** — install the optional dependencies and configure one
OpenAI-compatible chat-completions endpoint that supports tool calling:

```bash
uv sync --extra llm
cp .env_example .env
# Edit .env: set LLM_BASE_URL (including /v1 where required), LLM_API_KEY, LLM_MODEL.
uv run --extra llm finbalance suite --agent llm --seeds 0:1 --runs 1
# Repeatable suite, using the same settings/seeds as the comparison agent:
uv run --extra llm finbalance suite --agent llm --seeds 0:20 --k 5 --runs 3
```

The agent loads `.env` from the current working directory when initialized;
existing environment variables take precedence. `.env` is ignored by Git.
Keep the endpoint and model identifier with experiment notes, never the API key.
Missing configuration or optional dependencies produces a failed attempt with
an actionable error. The suite API retains errors in `SuiteResult.attempts`;
the CLI prints aggregate failure counts, latency summaries, and the first failure
reason. Use `uv run --extra llm` to select the project environment; a system
`python3` may not have the optional dependencies installed.

The [Pydantic AI](https://ai.pydantic.dev/output/) agent uses the existing six
session operations, with submission as its terminal output tool. Its prompt
includes the public fixed chart from `coa.py`, including normal balance sides
and protected-account flags. It has no access to ground truth. Tool calls execute
sequentially, with at most 40 model requests and a 60-second timeout per request;
provider transport retries are disabled. A text answer without submission, an
exhausted request limit, or a provider error is a failed attempt. Model setup is
outside the existing `agent.run()` timer. Reference commands need no LLM
dependencies or credentials. LLM campaign support is outside this initial task.

**Run the JEV agent with GPT fallback** — install the optional TypeSafe dependencies:

```bash
uv sync --extra jev --extra dev
# Add the TYPESAFE_* and JEV_* settings from .env_example to your .env.
# Keep existing LLM_* settings for GPT fallback.
uv run --extra jev finbalance suite --agent jev --seeds 1000:1001 --runs 1
```

JEV uses the same prompt, session tools and Pydantic AI loop as the LLM baseline.
Its model is `FallbackModel(TypeSafeModel(...), GPT, fallback_on=DecisionHandOff)`.
JEV attempts each step first; unsupported native routes hand that step to GPT.
Subsequent steps start with JEV again, using the same message and tool-result history.
There is no custom graph, separate grader or accounting solver.

Read arguments use public account codes and document IDs already returned by
`list_documents`. GPT can construct journal entries, interpret evidence and fill
submission mappings. Its usage is measured rather than capped at one call; fallback
may be frequent for free-form arguments. Backend outages fail explicitly.

Configure `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL` (default
`https://api.typesafe.ai`) and `TYPESAFE_DEFAULT_MODEL` (default `jev-latest`),
plus the existing three `LLM_*` settings for fallback. `JEV_ALLOW_LLM_ESCALATION=false`
disables GPT without requiring its configuration; unsupported JEV routes then fail
explicitly. The former `JEV_MAX_LLM_REQUESTS` and `JEV_MIN_CONFIDENCE` settings are
removed and ignored. `.env` loads without overriding environment variables.

Limits are 40 actual provider calls shared by JEV/GPT, 60 seconds per request and
300 seconds per run. Native routing and argument filling count separately, as do
validation retries. `agent.requests` exposes per-run JEV/GPT counts, and budget
failures include both. Submission ends the run; exhaustion never forces submission.

The optional model dependencies are pinned to Pydantic AI `2.57.0` and
require Python 3.11+. Pin the tested JEV model version instead of a moving alias
before evaluating. Use development seeds such as `1000:1020` for prompt changes;
freeze settings before comparing on the agreed evaluation seeds:

```bash
uv run --extra jev finbalance suite --agent llm --seeds 0:20 --k 5 --runs 3
uv run --extra jev finbalance suite --agent jev --seeds 0:20 --k 5 --runs 3
JEV_ALLOW_LLM_ESCALATION=false uv run --extra jev finbalance suite --agent jev --seeds 0:20 --k 5 --runs 3
```

These compare complete systems, including JEV's routing and GPT fallback. A missing TypeSafe key produces an actionable setup failure; the
mocked tests do not establish live accuracy.

For the mocked tests for both agents, install the JEV and development extras:

```bash
uv sync --extra jev --extra dev
uv run pytest -q
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
