"""
Performance benchmarks with explicit per-phase timing.

Each benchmark test times the full pipeline in order:

  1. build_scenario
  2. compile_scenario
  3. create_simulator
  4. run_ticks

Phase times are recorded with ``time.perf_counter`` and printed in the pytest
terminal summary (see ``tests/conftest.py``). The sum of phases plus a tiny
measurement overhead equals the reported total wall clock.

Environment variables (optional):

  PHARMASIM_BENCH_PROFILE   quick | bulgaria | stress  (default: stress)
  PHARMASIM_BENCH_TICKS     integer tick count for run_ticks (default varies by profile)

  stress-only overrides:
    PHARMASIM_BENCH_MARKETS=DE,FR,IT,UK
    PHARMASIM_BENCH_LOCATIONS=10000
    PHARMASIM_BENCH_PACKS=1000000
    PHARMASIM_BENCH_PRECOMPUTED_EDGES=1  # set to 0 for legacy list-edge path

Examples:

  # Fast sanity check (~seconds)
  PHARMASIM_BENCH_PROFILE=quick uv run pytest tests/bench.py -v

  # Bulgaria registry scale
  PHARMASIM_BENCH_PROFILE=bulgaria uv run pytest tests/bench.py -v

  # Full stress profile (can take hours)
  PHARMASIM_BENCH_PROFILE=stress uv run pytest tests/bench.py -v
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pytest

from compiler.compile import compile_scenario
from compiler.types import EngineInput
from policy.models import Scenario
from policy.scenarios import bulgaria_registry_scenario, two_markets_demo
from policy.scenarios_large import (
    multi_market_sparse_scenario,
    multi_market_sparse_scenario_precomputed,
)
from runtime.native_bridge import create_native_simulator

T = TypeVar("T")

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class BenchProfile:
    name: str
    description: str
    build_payload: Callable[
        [],
        tuple[Scenario, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None],
    ]
    num_ticks: int


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _resolve_profile() -> BenchProfile:
    profile = os.environ.get("PHARMASIM_BENCH_PROFILE", "stress").strip().lower()

    if profile == "quick":
        num_ticks = _env_int("PHARMASIM_BENCH_TICKS", 5_000)
        return BenchProfile(
            name="quick",
            description="two_markets_demo() — small synthetic sanity benchmark",
            build_payload=lambda: (two_markets_demo(), None),
            num_ticks=num_ticks,
        )

    if profile == "bulgaria":
        num_ticks = _env_int("PHARMASIM_BENCH_TICKS", 1_000)
        return BenchProfile(
            name="bulgaria",
            description=(
                "bulgaria_registry_scenario(geocode=False, max_locations=80) — "
                "data-driven network at modest scale"
            ),
            build_payload=lambda: (
                bulgaria_registry_scenario(
                    bg_registry_path=str(REPO_ROOT / "data" / "data.json"),
                    spor_locations_path=str(REPO_ROOT / "data" / "spor_locations.csv"),
                    geocode=False,
                    max_locations=80,
                    packs_per_obp=20,
                ),
                None,
            ),
            num_ticks=num_ticks,
        )

    if profile == "stress":
        markets = tuple(
            m.strip()
            for m in os.environ.get("PHARMASIM_BENCH_MARKETS", "DE,FR,IT,UK").split(",")
            if m.strip()
        )
        locations = _env_int("PHARMASIM_BENCH_LOCATIONS", 10_000)
        packs = _env_int("PHARMASIM_BENCH_PACKS", 1_000_000)
        num_ticks = _env_int("PHARMASIM_BENCH_TICKS", 100_000)

        use_precomputed = os.environ.get("PHARMASIM_BENCH_PRECOMPUTED_EDGES", "1") != "0"

        def build_stress() -> tuple[Scenario, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None]:
            if use_precomputed:
                scenario, edge_src, edge_dst, edge_cost, edge_capacity = (
                    multi_market_sparse_scenario_precomputed(
                        market_codes=markets,
                        locations_per_market=locations,
                        packs_per_market=packs,
                    )
                )
                return scenario, (edge_src, edge_dst, edge_cost, edge_capacity)
            return (
                multi_market_sparse_scenario(
                    market_codes=markets,
                    locations_per_market=locations,
                    packs_per_market=packs,
                ),
                None,
            )

        return BenchProfile(
            name="stress",
            description=(
                f"multi_market_sparse_scenario(markets={markets}, "
                f"locations_per_market={locations:,}, packs_per_market={packs:,}, "
                f"precomputed_edges={use_precomputed})"
            ),
            build_payload=build_stress,
            num_ticks=num_ticks,
        )

    raise ValueError(
        f"Unknown PHARMASIM_BENCH_PROFILE={profile!r}; "
        "expected one of: quick, bulgaria, stress"
    )


def _time_phase(name: str, fn: Callable[[], T], timings: dict[str, float]) -> T:
    start = time.perf_counter()
    result = fn()
    timings[name] = time.perf_counter() - start
    return result


def _model_summary(compiled: EngineInput) -> str:
    return (
        f"{compiled.n_locations:,} locations, "
        f"{compiled.n_packs:,} packs, "
        f"{compiled.n_edges:,} edges, "
        f"{compiled.n_markets} markets"
    )


def _run_pipeline_benchmark(request: pytest.FixtureRequest, profile: BenchProfile) -> None:
    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    scenario, precomputed_edges = _time_phase("1_build_scenario", profile.build_payload, timings)
    compiled = _time_phase(
        "2_compile_scenario",
        lambda: (
            compile_scenario(
                scenario,
                edge_src_location_id=precomputed_edges[0],
                edge_dst_location_id=precomputed_edges[1],
                edge_cost=precomputed_edges[2],
                edge_capacity=precomputed_edges[3],
            )
            if precomputed_edges is not None
            else compile_scenario(scenario)
        ),
        timings,
    )
    sim = _time_phase(
        "3_create_simulator",
        lambda: create_native_simulator(compiled),
        timings,
    )

    def run_ticks() -> None:
        sim.run_ticks(profile.num_ticks)

    _time_phase("4_run_ticks", run_ticks, timings)

    timings["0_total_wall"] = time.perf_counter() - total_start
    phase_keys = [k for k in timings if k.startswith(("1_", "2_", "3_", "4_"))]
    phase_sum = sum(timings[k] for k in phase_keys)

    ticks_per_second = (
        profile.num_ticks / timings["4_run_ticks"]
        if timings["4_run_ticks"] > 0
        else 0.0
    )
    ms_per_tick = (
        1000.0 * timings["4_run_ticks"] / profile.num_ticks
        if profile.num_ticks > 0
        else 0.0
    )

    meta: dict[str, Any] = {
        "profile": profile.name,
        "description": profile.description,
        "model": _model_summary(compiled),
        "num_ticks": profile.num_ticks,
        "event_count": sim.event_count(),
        "throughput": (
            f"{profile.num_ticks:,} ticks in {timings['4_run_ticks']:.3f}s "
            f"→ {ticks_per_second:,.1f} ticks/s ({ms_per_tick:.3f} ms/tick), "
            f"{sim.event_count():,} events"
        ),
    }

    request.node.user_properties.append(("bench_breakdown", timings))
    request.node.user_properties.append(("bench_meta", meta))

    # Phases are measured sequentially; total should match sum within floating noise.
    assert abs(timings["0_total_wall"] - phase_sum) < 0.05


@pytest.fixture(scope="module")
def bench_profile() -> BenchProfile:
    return _resolve_profile()


def test_bench_pipeline_breakdown(request: pytest.FixtureRequest, bench_profile: BenchProfile) -> None:
    """Time every pipeline phase; breakdown is printed in the terminal summary."""
    _run_pipeline_benchmark(request, bench_profile)
