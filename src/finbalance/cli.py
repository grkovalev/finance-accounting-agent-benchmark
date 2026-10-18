"""Command-line entry point: ``finbalance demo|generate|score``.

``demo`` is the one command a reviewer runs to see the whole thing work: it
generates a fresh instance, runs the oracle and null agents through the harness,
and prints a side-by-side scorecard.
"""

from __future__ import annotations

import argparse
import json
import os

from .agents import AGENTS
from .generator import generate_instance
from .harness import AgentSession
from .serialize import (
    dumps,
    instance_from_dict,
    instance_to_dict,
    solution_from_dict,
    solution_to_dict,
)
from .verifier import Submission, score


def _run_agent(agent, instance, gt) -> "Scorecard":
    return score(instance, gt, agent.run(AgentSession(instance), gt))


_DEMO_AGENTS = ("oracle", "baseline", "sloppy", "null")


def _scorecard_table(inst, gt, cards: dict) -> str:
    n_disc = len(gt.discrepancies)
    n_book = sum(1 for d in gt.discrepancies if d.requires_adjustment)
    names = list(cards)
    w = 11
    header = f"{'metric':<24}" + "".join(f"{n.upper():>{w}}" for n in names)
    lines = [
        f"Instance {inst.instance_id} | {len(inst.chart_of_accounts)} accounts | "
        f"{len(inst.general_ledger)} GL entries | {n_disc} discrepancies seeded "
        f"({n_book} book errors + 1 timing trap)",
        "",
        header,
        "-" * len(header),
    ]
    def row(label, fn):
        return f"{label:<24}" + "".join(f"{fn(cards[n]):>{w}}" for n in names)
    yn = lambda v: "PASS" if v else "FAIL"
    yesno = lambda v: "yes" if v else "no"
    lines += [
        row("BSexact", lambda s: yn(s.bs_exact)),
        row("BSrecon", lambda s: yn(s.bs_recon)),
        row("accounts_reconciled", lambda s: s.accounts_reconciled),
        row("discrepancies_caught", lambda s: s.discrepancies_caught),
        row("timing_trap_respected", lambda s: yesno(s.timing_trap_respected)),
        row("forbidden_touch", lambda s: yesno(s.forbidden_touch)),
        row("unauthorized_change", lambda s: yesno(s.unauthorized_change)),
        row("tool_calls", lambda s: s.tool_calls),
        "-" * len(header),
        row("FINAL SCORE", lambda s: f"{s.final_score:.2f}"),
    ]
    return "\n".join(lines)


def cmd_demo(args) -> int:
    inst, gt = generate_instance(args.seed, k=args.k)
    cards = {name: _run_agent(AGENTS[name](), inst, gt) for name in _DEMO_AGENTS}
    print(_scorecard_table(inst, gt, cards))
    return 0


def cmd_generate(args) -> int:
    inst, gt = generate_instance(args.seed, k=args.k)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "instance.json"), "w") as f:
        f.write(dumps(instance_to_dict(inst)))
    if args.with_solution:
        with open(os.path.join(args.out, "solution.json"), "w") as f:
            f.write(dumps(solution_to_dict(gt)))
    print(f"Wrote instance to {args.out} (with_solution={args.with_solution})")
    return 0


def cmd_score(args) -> int:
    with open(args.instance) as f:
        inst = instance_from_dict(json.load(f))
    with open(args.solution) as f:
        gt = solution_from_dict(json.load(f))
    with open(args.submission) as f:
        raw = json.load(f)
    from .serialize import _entry_from_dict
    sub = Submission(
        entries=tuple(_entry_from_dict(e) for e in raw.get("entries", [])),
        reported_bs={k: int(v) for k, v in raw.get("reported_bs", {}).items()},
        trace=tuple(raw.get("trace", [])),
    )
    sc = score(inst, gt, sub)
    print(dumps(sc.as_dict()))
    return 0


def cmd_suite(args) -> int:
    from .suite import format_suite, run_suite
    lo, hi = (int(x) for x in args.seeds.split(":"))
    result = run_suite(args.agent, range(lo, hi), runs=args.runs, k=args.k)
    print(format_suite(result))
    first_failure = next((a for a in result.attempts if a.outcome == "failed"), None)
    if first_failure is not None:
        print(f"  first failure (seed={first_failure.seed}, repetition={first_failure.repetition}): "
              f"{first_failure.error}")
    return 0


def cmd_campaign(args) -> int:
    from .agents import AGENTS
    from .campaign import format_campaign, run_campaign
    results = run_campaign(AGENTS[args.agent], args.seed, months=args.months, k=args.k)
    print(format_campaign(args.agent, results))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="finbalance", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="generate an instance and run all reference agents")
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--k", type=int, default=4, help="number of book-error discrepancies")
    d.set_defaults(func=cmd_demo)

    su = sub.add_parser("suite", help="evaluate an agent over a seed range (pass^k, per-type)")
    su.add_argument("--agent", default="baseline", choices=list(AGENTS))
    su.add_argument("--seeds", default="0:50", help="range as LO:HI")
    su.add_argument("--runs", type=int, default=1, help="runs per seed (for pass^k)")
    su.add_argument("--k", type=int, default=5)
    su.set_defaults(func=cmd_suite)

    ca = sub.add_parser("campaign", help="run a multi-month campaign (long-horizon)")
    ca.add_argument("--agent", default="baseline", choices=["oracle", "baseline", "sloppy", "null"])
    ca.add_argument("--seed", type=int, default=1)
    ca.add_argument("--months", type=int, default=3)
    ca.add_argument("--k", type=int, default=4)
    ca.set_defaults(func=cmd_campaign)

    g = sub.add_parser("generate", help="write instance.json (+solution.json)")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--k", type=int, default=4)
    g.add_argument("--out", required=True)
    g.add_argument("--with-solution", action="store_true")
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("score", help="score a submission against a solution")
    s.add_argument("--instance", required=True)
    s.add_argument("--solution", required=True)
    s.add_argument("--submission", required=True)
    s.set_defaults(func=cmd_score)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
