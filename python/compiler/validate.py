"""Validate a policy Scenario before compilation (IDs, foreign keys, invariants)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, TypeVar

from policy.models import LocationBehavior, Scenario

T = TypeVar("T")


class ScenarioValidationError(ValueError):
    """Raised when a :class:`~policy.models.Scenario` fails compiler validation."""


def _append_nonempty_ext_id_errors(
    label: str, items: Iterable[T], get_id: Callable[[T], str], errors: list[str]
) -> None:
    for item in items:
        eid = get_id(item)
        if not isinstance(eid, str) or not eid.strip():
            errors.append(f"{label}: ext_id must be a non-empty string, got {eid!r}")


def _unique_by_ext_id(
    label: str,
    items: Sequence[T],
    get_id: Callable[[T], str],
    errors: list[str],
) -> dict[str, T]:
    """Build ext_id -> item; record duplicate ext_ids."""
    out: dict[str, T] = {}
    for item in items:
        eid = get_id(item)
        if eid in out:
            errors.append(f"{label}: duplicate ext_id {eid!r}")
        else:
            out[eid] = item
    return out


def _prob_in_unit_interval(name: str, value: float, errors: list[str]) -> None:
    if not isinstance(value, (int, float)) or value < 0.0 or value > 1.0:
        errors.append(f"{name} must be in [0, 1], got {value!r}")


def validate_scenario(scenario: Scenario) -> Scenario:
    """
    Validate *scenario* and return it unchanged if valid.

    Raises ScenarioValidationError if any check fails. The message lists all collected issues.
    """
    errors: list[str] = []

    _append_nonempty_ext_id_errors(
        "Organization", scenario.organizations, lambda o: o.ext_id, errors
    )
    _append_nonempty_ext_id_errors(
        "Location", scenario.locations, lambda o: o.ext_id, errors
    )
    _append_nonempty_ext_id_errors(
        "Product", scenario.products, lambda o: o.ext_id, errors
    )
    _append_nonempty_ext_id_errors("Batch", scenario.batches, lambda o: o.ext_id, errors)
    _append_nonempty_ext_id_errors("Pack", scenario.packs, lambda o: o.ext_id, errors)

    orgs = _unique_by_ext_id(
        "Organization", scenario.organizations, lambda o: o.ext_id, errors
    )
    locs = _unique_by_ext_id(
        "Location", scenario.locations, lambda o: o.ext_id, errors
    )
    products = _unique_by_ext_id(
        "Product", scenario.products, lambda o: o.ext_id, errors
    )
    batches = _unique_by_ext_id(
        "Batch", scenario.batches, lambda o: o.ext_id, errors
    )
    _ = _unique_by_ext_id("Pack", scenario.packs, lambda o: o.ext_id, errors)

    # check products
    for p in scenario.products:
        codes = p.codes
        if not codes:
            errors.append(f"Product {p.ext_id!r}: at least one ProductCode is required")
            continue
        primary_count = sum(1 for c in codes if c.is_primary)
        if primary_count > 1:
            errors.append(
                f"Product {p.ext_id!r}: at most one code may have is_primary=True "
                f"({primary_count} found)"
            )
        seen_codes: set[tuple[str, str]] = set()
        for c in codes:
            if not c.value or not str(c.value).strip():
                errors.append(
                    f"Product {p.ext_id!r}: empty code value for scheme {c.scheme!r}"
                )
            key = (c.scheme.value, c.value)
            if key in seen_codes:
                errors.append(
                    f"Product {p.ext_id!r}: duplicate code {c.scheme.value}={c.value!r}"
                )
            seen_codes.add(key)

    # check locations foreign keys (FKs)
    for loc in scenario.locations:
        if loc.org_ext_id not in orgs:
            errors.append(
                f"Location {loc.ext_id!r}: unknown org_ext_id {loc.org_ext_id!r}"
            )

    # location graph edges (directed)
    for i, edge in enumerate(scenario.location_edges):
        if edge.src_location_ext_id not in locs:
            errors.append(
                f"LocationEdge[{i}]: unknown src_location_ext_id "
                f"{edge.src_location_ext_id!r}"
            )
        if edge.dst_location_ext_id not in locs:
            errors.append(
                f"LocationEdge[{i}]: unknown dst_location_ext_id "
                f"{edge.dst_location_ext_id!r}"
            )
        if edge.src_location_ext_id == edge.dst_location_ext_id:
            errors.append(
                f"LocationEdge[{i}]: self-loop is not allowed "
                f"({edge.src_location_ext_id!r})"
            )
        if not isinstance(edge.cost, (int, float)) or edge.cost < 0.0:
            errors.append(
                f"LocationEdge[{i}]: cost must be a non-negative number, "
                f"got {edge.cost!r}"
            )
        if not isinstance(edge.capacity, int) or edge.capacity < 0:
            errors.append(
                f"LocationEdge[{i}]: capacity must be a non-negative integer, "
                f"got {edge.capacity!r}"
            )

    # check batches: FKs and intended markets
    for b in scenario.batches:
        if b.product_ext_id not in products:
            errors.append(
                f"Batch {b.ext_id!r}: unknown product_ext_id {b.product_ext_id!r}"
            )
        if b.manufacturer_org_ext_id not in orgs:
            errors.append(
                f"Batch {b.ext_id!r}: unknown manufacturer_org_ext_id "
                f"{b.manufacturer_org_ext_id!r}"
            )
        if not b.intended_markets:
            errors.append(f"Batch {b.ext_id!r}: intended_markets must be non-empty")
        for m in b.intended_markets:
            if not m or not str(m).strip():
                errors.append(
                    f"Batch {b.ext_id!r}: empty string in intended_markets is not allowed"
                )

    # Check packs: FKs, batch/product alignment, location/market, serial uniqueness
    serial_keys: dict[tuple[str, str], str] = {}
    for pk in scenario.packs:
        if pk.product_ext_id not in products:
            errors.append(
                f"Pack {pk.ext_id!r}: unknown product_ext_id {pk.product_ext_id!r}"
            )
        if pk.batch_ext_id not in batches:
            errors.append(
                f"Pack {pk.ext_id!r}: unknown batch_ext_id {pk.batch_ext_id!r}"
            )
        if pk.initial_location_ext_id not in locs:
            errors.append(
                f"Pack {pk.ext_id!r}: unknown initial_location_ext_id "
                f"{pk.initial_location_ext_id!r}"
            )

        batch = batches.get(pk.batch_ext_id)
        if batch is not None and pk.product_ext_id != batch.product_ext_id:
            errors.append(
                f"Pack {pk.ext_id!r}: product_ext_id {pk.product_ext_id!r} does not match "
                f"batch {pk.batch_ext_id!r} product_ext_id {batch.product_ext_id!r}"
            )

        loc = locs.get(pk.initial_location_ext_id)
        if loc is not None and pk.initial_market_code != loc.market_code:
            errors.append(
                f"Pack {pk.ext_id!r}: initial_market_code {pk.initial_market_code!r} "
                f"!= location {loc.ext_id!r} market_code {loc.market_code!r}"
            )

        if batch is not None and batch.intended_markets:
            if pk.initial_market_code not in batch.intended_markets:
                errors.append(
                    f"Pack {pk.ext_id!r}: initial_market_code "
                    f"{pk.initial_market_code!r} not in batch {pk.batch_ext_id!r} "
                    f"intended_markets {batch.intended_markets!r}"
                )

        sk = (pk.product_ext_id, pk.serial)
        if pk.serial is None or not str(pk.serial).strip():
            errors.append(f"Pack {pk.ext_id!r}: serial must be non-empty")
        elif sk in serial_keys:
            errors.append(
                f"Pack {pk.ext_id!r}: duplicate (product_ext_id, serial) "
                f"{sk!r} (also pack {serial_keys[sk]!r})"
            )
        else:
            serial_keys[sk] = pk.ext_id

    # --- behavior_by_location ---
    for loc_id, beh in scenario.behavior_by_location.items():
        if loc_id not in locs:
            errors.append(
                f"behavior_by_location: unknown location ext_id {loc_id!r}"
            )
        if isinstance(beh, LocationBehavior):
            _prob_in_unit_interval(
                f"LocationBehavior({loc_id}).verify_prob", beh.verify_prob, errors
            )
            _prob_in_unit_interval(
                f"LocationBehavior({loc_id}).decomission_prob",
                beh.decomission_prob,
                errors,
            )
            _prob_in_unit_interval(
                f"LocationBehavior({loc_id}).reactivate_prob",
                beh.reactivate_prob,
                errors,
            )

    if errors:
        raise ScenarioValidationError(
            "Scenario validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    return scenario
