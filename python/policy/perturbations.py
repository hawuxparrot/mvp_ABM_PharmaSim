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

from compiler.enums import decommission_reason_u8, org_type_u8, pack_state_u8
from compiler.types import EngineInput
from policy.models import DecommissionReason, OrgType, PackState

WHOLESALER_U8 = int(org_type_u8(OrgType.WHOLESALER))
OBP_U8 = int(org_type_u8(OrgType.OBP))
LOCAL_ORG_U8 = int(org_type_u8(OrgType.LOCAL_ORG))
ACTIVE_U8 = pack_state_u8(PackState.ACTIVE)
STOLEN_U8 = decommission_reason_u8(DecommissionReason.STOLEN)


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


def _append_counterfeit_batch(inp: EngineInput) -> int:
    """Append a batch provisioned to *no* market, returning its id.

    Counterfeit packs reference this batch so their (fresh) serials are unknown to every NMVS,
    which is exactly what makes the engine raise ``UNKNOWN_SERIAL`` when they are scanned.
    """
    product_id = int(inp.pack_product_id[0]) if inp.n_packs > 0 else 0
    manufacturer = int(inp.batch_manufacturer_org_id[0]) if inp.n_batches > 0 else 0
    batch_id = inp.n_batches
    inp.batch_product_id = np.concatenate(
        [inp.batch_product_id, np.array([product_id], dtype=np.uint32)]
    )
    inp.batch_manufacturer_org_id = np.concatenate(
        [inp.batch_manufacturer_org_id, np.array([manufacturer], dtype=np.uint32)]
    )
    # Empty intended-market range: new offset repeats the previous last offset.
    last_off = int(inp.batch_intended_market_offset[-1])
    inp.batch_intended_market_offset = np.concatenate(
        [inp.batch_intended_market_offset, np.array([last_off], dtype=np.uint32)]
    )
    if inp.batch_ext_id:
        inp.batch_ext_id = list(inp.batch_ext_id) + [f"counterfeit_batch_{batch_id}"]
    inp.n_batches = batch_id + 1
    return batch_id


def _inject_packs(
    inp: EngineInput,
    loc: int,
    count: int,
    *,
    serial_prefix: str,
    batch_id: int | None = None,
) -> None:
    """Append ``count`` Active packs at ``loc``, each with a fresh interned serial id.

    ``batch_id`` selects provisioning: the genuine batch 0 makes them look legitimate (volume
    surge), a counterfeit batch (no markets) makes their serials unknown to the NMVS.
    """
    if count <= 0:
        return
    product_id = int(inp.pack_product_id[0]) if inp.n_packs > 0 else 0
    if batch_id is None:
        batch_id = int(inp.pack_batch_id[0]) if inp.n_packs > 0 else 0
    market_id = int(inp.location_market_id[loc])

    new_product = np.full(count, product_id, dtype=np.uint32)
    new_batch = np.full(count, batch_id, dtype=np.uint32)
    new_loc = np.full(count, loc, dtype=np.uint32)
    new_market = np.full(count, market_id, dtype=np.uint32)
    new_state = np.full(count, ACTIVE_U8, dtype=np.uint8)

    base = inp.n_packs
    serial_base = inp.n_serials
    new_serial_id = np.arange(serial_base, serial_base + count, dtype=np.uint32)
    serials = [f"{serial_prefix}{base + k:09d}" for k in range(count)]
    ext_ids = [f"{serial_prefix.lower()}pack_{base + k}" for k in range(count)]

    inp.pack_product_id = np.concatenate([inp.pack_product_id, new_product])
    inp.pack_batch_id = np.concatenate([inp.pack_batch_id, new_batch])
    inp.pack_initial_location_id = np.concatenate([inp.pack_initial_location_id, new_loc])
    inp.pack_initial_market_id = np.concatenate([inp.pack_initial_market_id, new_market])
    inp.pack_initial_state = np.concatenate([inp.pack_initial_state, new_state])
    inp.pack_serial_id = np.concatenate([inp.pack_serial_id, new_serial_id])
    inp.pack_serial = list(inp.pack_serial) + serials
    if inp.pack_ext_id:
        inp.pack_ext_id = list(inp.pack_ext_id) + ext_ids
    inp.n_packs = base + count
    inp.n_serials = serial_base + count


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
            # Extra (legitimate) stock so the higher capacity has packs to push: a genuine
            # volume surge, not an EMVS rule violation -- detected by transaction volume.
            _inject_packs(
                out,
                loc,
                int(round(8 * spec.intensity)),
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
        # Counterfeits surface where they are dispensed. Target terminals (pharmacies/hospitals)
        # -- they verify (have behavior) and are sinks, so the fake packs do not propagate and
        # the UNKNOWN_SERIAL alerts localize to the injection site.
        targets = _select_targets(out, spec, LOCAL_ORG_U8, require_out_edge=False)
        if not targets:
            targets = _select_targets(out, spec, WHOLESALER_U8, require_out_edge=True)
        cf_batch = _append_counterfeit_batch(out) if targets else 0
        for loc in targets:
            # Guarantee the fakes get scanned at the target.
            out.location_has_behavior[loc] = np.uint8(1)
            out.location_verify_prob[loc] = np.float32(
                max(float(out.location_verify_prob[loc]), 0.5)
            )
            _inject_packs(
                out,
                loc,
                int(round(10 * spec.intensity)),
                serial_prefix="CF",
                batch_id=cf_batch,
            )
            labels.append(
                PerturbationLabel(
                    location_ext_id=_loc_label(out, loc),
                    location_id=loc,
                    ptype=spec.ptype,
                    reason="Injected fabricated-serial packs (unknown to the NMVS)",
                )
            )

    elif spec.ptype == PerturbationType.THEFT:
        targets = _select_targets(out, spec, WHOLESALER_U8, require_out_edge=True)
        prob = float(min(0.95, max(0.05, spec.intensity / 10.0)))
        for loc in targets:
            # Distributor decommissions stock it should be shipping, stamping the EMVS STOLEN
            # reason -- a reason-coded decommission a legitimate distributor never emits.
            out.location_has_behavior[loc] = np.uint8(1)
            out.location_decommission_prob[loc] = np.float32(prob)
            out.location_decommission_reason_id[loc] = STOLEN_U8
            labels.append(
                PerturbationLabel(
                    location_ext_id=_loc_label(out, loc),
                    location_id=loc,
                    ptype=spec.ptype,
                    reason="Abnormal STOLEN decommission/shrinkage at a distributor",
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
            # Make sure the diverted node scans, so packs provisioned elsewhere raise
            # OUT_OF_MARKET when verified in the foreign NMVS.
            out.location_has_behavior[loc] = np.uint8(1)
            out.location_verify_prob[loc] = np.float32(
                max(float(out.location_verify_prob[loc]), 0.5)
            )
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
