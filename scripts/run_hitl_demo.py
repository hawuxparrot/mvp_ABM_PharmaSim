#!/usr/bin/env python
from __future__ import annotations

import argparse
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

from learning.runner import HitlDemoConfig, run_hitl_demo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PharmaSim HITL alert-learning demo.")
    parser.add_argument("--scenario", default="bulgaria")
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--perturbations", default="COUNTERFEIT,SMUGGLING,THEFT,VOLUME_SPIKE")
    parser.add_argument("--operator", default="configs/hitl/operator_rules.v1.json")
    parser.add_argument("--query-policy", default="all", choices=["all", "uncertainty", "uncertainty-query"])
    parser.add_argument("--output", default="artifacts/hitl_demo")
    parser.add_argument("--mode", default="oracle", choices=["oracle", "human"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--safe-sample-rate", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-strength", type=float, default=0.0001)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--initial-bias", type=float, default=-1.0)
    parser.add_argument("--query-budget-per-tick", type=int, default=8)
    parser.add_argument("--uncertainty-margin", type=float, default=0.15)
    parser.add_argument("--exploration-rate", type=float, default=0.05)
    parser.add_argument("--metric-interval", type=int, default=25)
    parser.add_argument("--evaluation-fraction", type=float, default=0.30)
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        help="Hold out this complete seed (default: the last seed).",
    )
    parser.add_argument("--metric-eval-max-cases", type=int, default=2_000)
    parser.add_argument("--max-locations", type=int, default=350)
    parser.add_argument("--packs-per-obp", type=int, default=60)
    parser.add_argument("--packs-per-wholesaler", type=int, default=12)
    parser.add_argument("--n-obp-hubs", type=int, default=6)
    parser.add_argument("--no-geocode", action="store_true")
    parser.add_argument("--geocode-cache", default="data/geocode_cache_bg.json")
    parser.add_argument("--geocode-max-new-requests", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    config = HitlDemoConfig(
        scenario=args.scenario,
        ticks=args.ticks,
        seeds=_parse_ints(args.seeds),
        perturbations=_parse_strings(args.perturbations),
        operator_path=Path(args.operator),
        query_policy=args.query_policy,
        output_dir=Path(args.output),
        mode=args.mode,
        resume=args.resume,
        learning_rate=args.learning_rate,
        l2_strength=args.l2_strength,
        decision_threshold=args.decision_threshold,
        initial_bias=args.initial_bias,
        safe_sample_rate=args.safe_sample_rate,
        query_budget_per_tick=args.query_budget_per_tick,
        uncertainty_margin=args.uncertainty_margin,
        exploration_rate=args.exploration_rate,
        metric_interval=args.metric_interval,
        evaluation_fraction=args.evaluation_fraction,
        evaluation_seed=args.evaluation_seed,
        metric_eval_max_cases=args.metric_eval_max_cases,
        max_locations=args.max_locations,
        packs_per_obp=args.packs_per_obp,
        packs_per_wholesaler=args.packs_per_wholesaler,
        n_obp_hubs=args.n_obp_hubs,
        geocode=not args.no_geocode,
        geocode_cache_path=Path(args.geocode_cache) if args.geocode_cache else None,
        geocode_max_new_requests=args.geocode_max_new_requests,
        progress=not args.no_progress,
    )
    result = run_hitl_demo(config)
    final = result.metrics["final"]
    print(
        f"Wrote {result.output_dir} | labels={result.labels_requested} "
        f"cases={result.cases_processed} F1={final['f1']:.3f} FPR={final['false_positive_rate']:.3f}"
    )
    print(f"Convergence plot: {result.output_dir / 'oracle_convergence.png'}")


def _parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _parse_strings(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


if __name__ == "__main__":
    main()
