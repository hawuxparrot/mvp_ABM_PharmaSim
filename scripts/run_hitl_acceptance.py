#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON_DIR = REPO / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
sys.meta_path = [
    finder
    for finder in sys.meta_path
    if not finder.__class__.__module__.startswith("_editable_skbc_pharmasim")
]

from learning.acceptance import run_seed_holdout_acceptance  # noqa: E402
from learning.operator import HiddenOperator  # noqa: E402
from learning.runner import (  # noqa: E402
    HitlDemoConfig,
    build_bulgaria_streams,
    repo_commit_hash,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run five-seed holdout acceptance for both HITL query policies."
    )
    parser.add_argument("--scenario", default="bulgaria")
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument(
        "--perturbations",
        default="COUNTERFEIT,SMUGGLING,THEFT,VOLUME_SPIKE",
    )
    parser.add_argument("--operator", default="configs/hitl/operator_rules.v1.json")
    parser.add_argument("--output", default="artifacts/hitl_acceptance")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-strength", type=float, default=0.0001)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--initial-bias", type=float, default=-1.0)
    parser.add_argument("--safe-sample-rate", type=float, default=0.10)
    parser.add_argument("--query-budget-per-tick", type=int, default=8)
    parser.add_argument("--uncertainty-margin", type=float, default=0.15)
    parser.add_argument("--exploration-rate", type=float, default=0.05)
    parser.add_argument("--max-locations", type=int, default=350)
    parser.add_argument("--packs-per-obp", type=int, default=60)
    parser.add_argument("--packs-per-wholesaler", type=int, default=12)
    parser.add_argument("--n-obp-hubs", type=int, default=6)
    parser.add_argument("--no-geocode", action="store_true")
    parser.add_argument("--geocode-cache", default="data/geocode_cache_bg.json")
    parser.add_argument("--geocode-max-new-requests", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    seeds = _parse_ints(args.seeds)
    config = HitlDemoConfig(
        scenario=args.scenario,
        ticks=args.ticks,
        seeds=seeds,
        perturbations=_parse_strings(args.perturbations),
        operator_path=Path(args.operator),
        output_dir=Path(args.output),
        learning_rate=args.learning_rate,
        l2_strength=args.l2_strength,
        decision_threshold=args.decision_threshold,
        initial_bias=args.initial_bias,
        safe_sample_rate=args.safe_sample_rate,
        query_budget_per_tick=args.query_budget_per_tick,
        uncertainty_margin=args.uncertainty_margin,
        exploration_rate=args.exploration_rate,
        max_locations=args.max_locations,
        packs_per_obp=args.packs_per_obp,
        packs_per_wholesaler=args.packs_per_wholesaler,
        n_obp_hubs=args.n_obp_hubs,
        geocode=not args.no_geocode,
        geocode_cache_path=Path(args.geocode_cache) if args.geocode_cache else None,
        geocode_max_new_requests=args.geocode_max_new_requests,
        progress=not args.no_progress,
    )
    if config.scenario != "bulgaria":
        parser.error("the acceptance suite currently supports only --scenario bulgaria")

    streams = build_bulgaria_streams(config)
    operator = HiddenOperator.from_json(config.operator_path)
    result = run_seed_holdout_acceptance(streams, operator, config=config)
    payload = {
        "commit_hash": repo_commit_hash(),
        "config": {
            "ticks": config.ticks,
            "seeds": list(config.seeds),
            "perturbations": list(config.perturbations),
            "operator": str(config.operator_path),
        },
        **result,
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.output_dir / "acceptance.json"
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    state = "PASS" if result["passed"] else "FAIL"
    print(f"{state}: wrote {report_path}")
    if not result["passed"]:
        raise SystemExit(1)


def _parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _parse_strings(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


if __name__ == "__main__":
    main()
