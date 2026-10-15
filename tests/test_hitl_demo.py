"""Native-backed HITL replay integration checks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PYTHON_DIR = REPO / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
sys.meta_path = [
    finder
    for finder in sys.meta_path
    if not finder.__class__.__module__.startswith("_editable_skbc_pharmasim")
]
sys.modules.pop("runtime", None)

if (
    importlib.util.find_spec("runtime._pharmasim_native") is None
    and importlib.util.find_spec("_pharmasim_native") is None
):
    pytest.skip("native extension _pharmasim_native not available", allow_module_level=True)

from compiler.compile import compile_scenario  # noqa: E402
from learning.case_builder import CaseBuilder, CaseBuilderConfig, build_cases_from_records  # noqa: E402
from learning.operator import HiddenOperator  # noqa: E402
from learning.runner import CaseStream, HitlDemoConfig, run_case_streams  # noqa: E402
from policy.perturbations import PerturbationSpec, PerturbationType, apply_perturbation  # noqa: E402
from policy.scenarios import two_markets_demo  # noqa: E402
from runtime.experiments import run_simulation  # noqa: E402
from runtime.native_bridge import create_native_simulator  # noqa: E402
from runtime.simulation_viz import events_as_records, run_ticks_with_hook  # noqa: E402


RULES = Path("configs/hitl/operator_rules.v1.json")


def test_one_counterfeit_perturbation_in_oracle_mode(tmp_path) -> None:
    scenario = two_markets_demo()
    inp = compile_scenario(scenario)
    perturbed, labels = apply_perturbation(
        inp,
        PerturbationSpec(PerturbationType.COUNTERFEIT, n_targets=1, intensity=3.0, seed=3),
    )
    _sim, records = run_simulation(perturbed, 120)
    builder = CaseBuilder(
        perturbed,
        scenario,
        None,
        config=CaseBuilderConfig(safe_sample_rate=1.0),
    )
    cases = build_cases_from_records(records, builder, stream_id="counterfeit")
    assert any(case.alert_code == "UNKNOWN_SERIAL" for case in cases)
    stream = CaseStream(
        "counterfeit",
        "train",
        cases,
        records,
        engine_input=perturbed,
        scenario=scenario,
        perturbation_type="COUNTERFEIT",
        perturbation_labels=labels,
    )
    config = HitlDemoConfig(
        ticks=120,
        seeds=(3,),
        perturbations=("COUNTERFEIT",),
        operator_path=RULES,
        output_dir=tmp_path,
        safe_sample_rate=1.0,
        geocode=False,
    )
    result = run_case_streams([stream], [stream], HiddenOperator.from_json(RULES), config=config)
    assert result.labels_requested == len(cases)
    assert (tmp_path / "metrics.json").exists()


def test_replay_and_live_mode_case_identifiers_match() -> None:
    scenario = two_markets_demo()
    inp = compile_scenario(scenario)
    replay_sim = create_native_simulator(inp)
    replay_sim.run_ticks(80)
    replay_records = events_as_records(replay_sim, inp)
    replay_builder = CaseBuilder(
        inp,
        scenario,
        None,
        config=CaseBuilderConfig(safe_sample_rate=1.0),
    )
    replay_ids = [
        case.case_id
        for case in build_cases_from_records(replay_records, replay_builder, stream_id="cmp")
    ]

    live_sim = create_native_simulator(inp)
    live_builder = CaseBuilder(
        inp,
        scenario,
        None,
        config=CaseBuilderConfig(safe_sample_rate=1.0),
    )
    live_history = live_builder.new_history()
    live_ids: list[str] = []

    def on_tick(info) -> None:
        cases = live_builder.build(info.new_events, live_history, stream_id="cmp")
        live_ids.extend(case.case_id for case in cases)
        live_history.update(info.new_events)

    run_ticks_with_hook(live_sim, inp, 80, on_tick)
    assert replay_ids == live_ids
