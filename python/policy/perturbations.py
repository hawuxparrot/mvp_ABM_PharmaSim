"""Fraud / smuggling perturbations applied to a calibrated ``EngineInput``.

Per the project decision, fraud lives in a Python policy-driver: each perturbation rewrites
engine parameters (supply capacity, behavior probabilities, location market, injected stock)
so the anomaly manifests in the *real* engine event log produced by ``run_ticks``. Each
perturbation also returns ground-truth :class:`PerturbationLabel`s, kept separate from the
engine state so detectors can be scored without leakage.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from compiler.enums import org_type_u8, pack_state_u8
from compiler.types import EngineInput
from policy.models import OrgType, PackState

WHOLESALER_U8 = int(org_type_u8(OrgType.WHOLESALER))
OBP_U8 = int(org_type_u8(OrgType.OBP))
LOCAL_ORG_U8 = int(org_type_u8(OrgType.LOCAL_ORG))
ACTIVE_U8 = pack_state_u8(PackState.ACTIVE)


class PerturbationType(StrEnum):
    VOLUME_SPIKE = "VOLUME_SPIKE"
    SMUGGLING = "SMUGGLING"
    THEFT = "THEFT"
    COUNTERFEIT = "COUNTERFEIT"


@dataclass(frozen=True)
class PerturbationSpec:
    ptype: PerturbationType
    n_targets: int = 3
    intensity: float = 8.0
    seed: int = 7
    foreign_market_code: str = "XX"


@dataclass(frozen=True)
class PerturbationLabel:
    location_ext_id: str
    location_id: int
    ptype: PerturbationType
    reason: str
    metadata: dict[str, str] = field(default_factory=dict)


def copy_engine_input(inp: EngineInput) -> EngineInput:
    """Deep copy so perturbations never mutate the baseline engine input."""
    return copy.deepcopy(inp)


def _location_org_types(inp: EngineInput) -> np.ndarray:
    return np.asarray(inp.org_type)[np.asarray(inp.location_org_id)]


def _has_out_edges(inp: EngineInput, loc: int) -> bool:
    off = inp.location_out_edge_offset
    return int(off[loc + 1]) - int(off[loc]) > 0


def _preferred_edge_destinations(inp: EngineInput) -> set[int]:
    """Locations that statically receive packs, i.e. the destination of some supply
    node's preferred (push) edge. Flow is deterministic along these edges, so these are
    exactly the nodes guaranteed to see inbound shipments."""
    org_t = _location_org_types(inp)
    no_edge = np.uint32(2**32 - 1)
    reached: set[int] = set()
    for i in range(inp.n_locations):
        if int(org_t[i]) not in (WHOLESALER_U8, OBP_U8):
            continue
        pe = inp.location_preferred_supplier_edge_id[i]
        if pe == no_edge:
            continue
        reached.add(int(inp.edge_dst_location_id[int(pe)]))
    return reached


def _candidate_targets(
    inp: EngineInput,
    org_u8: int,
    *,
    require_out_edge: bool,
    require_inbound_flow: bool = False,
) -> list[int]:
    org_t = _location_org_types(inp)
    reached = _preferred_edge_destinations(inp) if require_inbound_flow else None
    out: list[int] = []
    for i in range(inp.n_locations):
        if int(org_t[i]) != org_u8:
            continue
        if require_out_edge and not _has_out_edges(inp, i):
            continue
        if reached is not None and i not in reached:
            continue
        out.append(i)
    return out


def _select_targets(
    inp: EngineInput,
    spec: PerturbationSpec,
    org_u8: int,
    *,
    require_out_edge: bool,
    require_inbound_flow: bool = False,
) -> list[int]:
    cands = _candidate_targets(
        inp,
        org_u8,
        require_out_edge=require_out_edge,
        require_inbound_flow=require_inbound_flow,
    )
    if not cands:
        return []
    rng = np.random.default_rng(spec.seed)
    n = min(spec.n_targets, len(cands))
    chosen = rng.choice(np.asarray(cands), size=n, replace=False)
    return sorted(int(x) for x in chosen)


def _loc_label(inp: EngineInput, loc: int) -> str:
    if inp.location_ext_id and loc < len(inp.location_ext_id):
        return inp.location_ext_id[loc]
    return f"loc#{loc}"


def _ensure_market(inp: EngineInput, code: str) -> int:
    if code in inp.market_code:
        return inp.market_code.index(code)
    inp.market_code.append(code)
    inp.n_markets = len(inp.market_code)
    return inp.n_markets - 1


def _inject_packs(
    inp: EngineInput,
    loc: int,
    count: int,
    *,
    clone_serials: bool,
    serial_prefix: str,
) -> None:
    if count <= 0:
        return
    product_id = int(inp.pack_product_id[0]) if inp.n_packs > 0 else 0
    batch_id = int(inp.pack_batch_id[0]) if inp.n_packs > 0 else 0
    market_id = int(inp.location_market_id[loc])

    new_product = np.full(count, product_id, dtype=np.uint32)
    new_batch = np.full(count, batch_id, dtype=np.uint32)
    new_loc = np.full(count, loc, dtype=np.uint32)
    new_market = np.full(count, market_id, dtype=np.uint32)
    new_state = np.full(count, ACTIVE_U8, dtype=np.uint8)

    base = inp.n_packs
    serials: list[str] = []
    ext_ids: list[str] = []
    for k in range(count):
        if clone_serials and inp.pack_serial:
            serials.append(inp.pack_serial[k % len(inp.pack_serial)])
        else:
            serials.append(f"{serial_prefix}{base + k:09d}")
        ext_ids.append(f"{serial_prefix.lower()}pack_{base + k}")

    inp.pack_product_id = np.concatenate([inp.pack_product_id, new_product])
    inp.pack_batch_id = np.concatenate([inp.pack_batch_id, new_batch])
    inp.pack_initial_location_id = np.concatenate([inp.pack_initial_location_id, new_loc])
    inp.pack_initial_market_id = np.concatenate([inp.pack_initial_market_id, new_market])
    inp.pack_initial_state = np.concatenate([inp.pack_initial_state, new_state])
    inp.pack_serial = list(inp.pack_serial) + serials
    if inp.pack_ext_id:
        inp.pack_ext_id = list(inp.pack_ext_id) + ext_ids
    inp.n_packs = base + count


def apply_perturbation(
    inp: EngineInput, spec: PerturbationSpec
) -> tuple[EngineInput, list[PerturbationLabel]]:
    """Return a perturbed copy of ``inp`` plus ground-truth labels for injected fraud."""
    out = copy_engine_input(inp)
    labels: list[PerturbationLabel] = []

    if spec.ptype == PerturbationType.VOLUME_SPIKE:
        targets = _select_targets(out, spec, WHOLESALER_U8, require_out_edge=True)
        for loc in targets:
            base_cap = int(out.location_supply_capacity_per_tick[loc])
            out.location_supply_capacity_per_tick[loc] = np.uint32(
                max(base_cap + 1, int(round(max(1, base_cap) * spec.intensity)))
            )
            # Extra stock so the higher capacity has packs to push (a genuine surge).
            _inject_packs(
                out,
                loc,
                int(round(8 * spec.intensity)),
                clone_serials=False,
                serial_prefix="VS",
            )
            labels.append(
                PerturbationLabel(
                    location_ext_id=_loc_label(out, loc),
                    location_id=loc,
                    ptype=spec.ptype,
                    reason="Inflated supply capacity + injected surge stock",
                )
            )

    elif spec.ptype == PerturbationType.COUNTERFEIT:
        targets = _select_targets(out, spec, WHOLESALER_U8, require_out_edge=True)
        for loc in targets:
            _inject_packs(
                out,
                loc,
                int(round(10 * spec.intensity)),
                clone_serials=True,
                serial_prefix="CF",
            )
            labels.append(
                PerturbationLabel(
                    location_ext_id=_loc_label(out, loc),
                    location_id=loc,
                    ptype=spec.ptype,
                    reason="Injected cloned-serial packs entering distribution",
                )
            )

    elif spec.ptype == PerturbationType.THEFT:
        targets = _select_targets(out, spec, WHOLESALER_U8, require_out_edge=True)
        prob = float(min(0.95, max(0.05, spec.intensity / 10.0)))
        for loc in targets:
            out.location_has_behavior[loc] = np.uint8(1)
            out.location_decommission_prob[loc] = np.float32(prob)
            labels.append(
                PerturbationLabel(
                    location_ext_id=_loc_label(out, loc),
                    location_id=loc,
                    ptype=spec.ptype,
                    reason="Abnormal decommission/shrinkage at a distributor",
                    metadata={"decommission_prob": f"{prob:.3f}"},
                )
            )

    elif spec.ptype == PerturbationType.SMUGGLING:
        foreign_id = _ensure_market(out, spec.foreign_market_code)
        # Divert at terminals that actually receive shipments so packs cross into the
        # foreign market on arrival (an observable cross-market MOVE).
        targets = _select_targets(
            out,
            spec,
            LOCAL_ORG_U8,
            require_out_edge=False,
            require_inbound_flow=True,
        )
        if not targets:
            targets = _select_targets(
                out, spec, WHOLESALER_U8, require_out_edge=True, require_inbound_flow=True
            )
        for loc in targets:
            out.location_market_id[loc] = np.uint32(foreign_id)
            labels.append(
                PerturbationLabel(
                    location_ext_id=_loc_label(out, loc),
                    location_id=loc,
                    ptype=spec.ptype,
                    reason="Distributor diverted into a foreign market",
                    metadata={"foreign_market": spec.foreign_market_code},
                )
            )
    else:  # pragma: no cover - exhaustive
        raise ValueError(f"unknown perturbation type {spec.ptype!r}")

    out.validate_shapes()
    return out, labels
