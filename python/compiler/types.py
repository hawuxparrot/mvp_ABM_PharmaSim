"""
Compiler output types: stable contract between ``compile_scenario`` and the C++ engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

# ABI version: bump when adding/removing/reordering fields or changing dtypes.
ENGINE_INPUT_SCHEMA_VERSION: str = "engine_input.v3"


@dataclass
class EngineInput:
    """
    Columnar engine bootstrap payload (SoA + CSR for ragged fields).

    Convention

    * Metadata: Python str/int (schema_version, seed, counts).
    * Numeric columns: 1-D :class:numpy.ndarray with fixed dtype (uint32, uint8, float32) — these map directly to C++ std::vector / pointers.
      uint8, float32) — these map directly to C++ std::vector s.
    * Debug I/O parallel strings: list[str] aligned by row index to the matching table (same length as n_*).

    Note: Do not use dtype=object arrays for strings; use list[str] instead.
    Note: SoA refers to "Struct of Arrays" layout, used here for superior performance in contrast to AoS
    """

    schema_version: str
    seed: int

    # counts: to be used for C++ asserts
    n_organizations: int
    n_locations: int
    n_products: int
    n_batches: int
    n_packs: int
    n_markets: int
    n_edges: int

    # markets: market_id is row index into market_code
    market_code: list[str]
    
    # OrgType encoded as uint8; pair with a shared Python/C++ enum table.
    org_type: NDArray[np.uint8]
    

    # locations
    location_org_id: NDArray[np.uint32]
    location_market_id: NDArray[np.uint32]
    location_out_edge_offset: NDArray[np.uint32]  # CSR offset for outgoing edge ids.
    location_out_edge_id: NDArray[np.uint32]      # references rows in edge_* columns.

    # location graph edges (directed)
    edge_src_location_id: NDArray[np.uint32]
    edge_dst_location_id: NDArray[np.uint32]
    edge_cost: NDArray[np.float32]
    edge_capacity: NDArray[np.uint32]

    # batches + CSR for variable-length intended markets
    batch_product_id: NDArray[np.uint32]
    batch_manufacturer_org_id: NDArray[np.uint32]
    batch_intended_market_offset: NDArray[np.uint32]
    batch_intended_market_id: NDArray[np.uint32] # batch_intended_market_id[offset[i]:offset[i+1]] are market ids.

    # packs main simulation SoA
    pack_product_id: NDArray[np.uint32]
    pack_batch_id: NDArray[np.uint32]
    pack_initial_location_id: NDArray[np.uint32]
    pack_initial_market_id: NDArray[np.uint32]
    pack_initial_state: NDArray[np.uint8] # PackState encoded as uint8.
    pack_serial: list[str] # optional parallel ext_id strings (debug / round-trip); empty = omit

    # Location behavior (Option B): always length n_locations; zeros when no policy for that row/scenario.
    location_has_behavior: NDArray[np.uint8]
    location_verify_prob: NDArray[np.float32]
    location_decommission_prob: NDArray[np.float32]
    location_reactivate_prob: NDArray[np.float32]

    org_ext_id: list[str] = field(default_factory=list)
    location_ext_id: list[str] = field(default_factory=list)
    batch_ext_id: list[str] = field(default_factory=list)
    pack_ext_id: list[str] = field(default_factory=list)

    #: Expected schema string for this layout (class attribute, not an instance field).
    EXPECTED_SCHEMA: ClassVar[str] = ENGINE_INPUT_SCHEMA_VERSION

    def validate_shapes(self) -> None:
        """Lightweight consistency check (call from ``compile_scenario`` / tests)."""
        if self.schema_version != ENGINE_INPUT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {self.schema_version!r} != {ENGINE_INPUT_SCHEMA_VERSION!r}"
            )

        def _len(name: str, arr: NDArray[np.generic], n: int) -> None:
            if arr.shape != (n,):
                raise ValueError(f"{name} expected shape ({n},), got {arr.shape}")

        _len("org_type", self.org_type, self.n_organizations)
        _len("location_org_id", self.location_org_id, self.n_locations)
        _len("location_market_id", self.location_market_id, self.n_locations)
        _len("location_out_edge_offset", self.location_out_edge_offset, self.n_locations + 1)
        _len("location_out_edge_id", self.location_out_edge_id, self.n_edges)
        _len("edge_src_location_id", self.edge_src_location_id, self.n_edges)
        _len("edge_dst_location_id", self.edge_dst_location_id, self.n_edges)
        _len("edge_cost", self.edge_cost, self.n_edges)
        _len("edge_capacity", self.edge_capacity, self.n_edges)
        _len("batch_product_id", self.batch_product_id, self.n_batches)
        _len("batch_manufacturer_org_id", self.batch_manufacturer_org_id, self.n_batches)
        _len("batch_intended_market_offset", self.batch_intended_market_offset, self.n_batches + 1)
        _len("pack_product_id", self.pack_product_id, self.n_packs)
        _len("pack_batch_id", self.pack_batch_id, self.n_packs)
        _len("pack_initial_location_id", self.pack_initial_location_id, self.n_packs)
        _len("pack_initial_market_id", self.pack_initial_market_id, self.n_packs)
        _len("pack_initial_state", self.pack_initial_state, self.n_packs)

        if len(self.market_code) != self.n_markets:
            raise ValueError("market_code length must match n_markets")
        if self.org_ext_id and len(self.org_ext_id) != self.n_organizations:
            raise ValueError("org_ext_id length must match n_organizations or be empty")
        if self.location_ext_id and len(self.location_ext_id) != self.n_locations:
            raise ValueError("location_ext_id length must match n_locations or be empty")
        if self.batch_ext_id and len(self.batch_ext_id) != self.n_batches:
            raise ValueError("batch_ext_id length must match n_batches or be empty")
        if len(self.pack_serial) != self.n_packs:
            raise ValueError("pack_serial length must match n_packs")
        if self.pack_ext_id and len(self.pack_ext_id) != self.n_packs:
            raise ValueError("pack_ext_id length must match n_packs or be empty")

        off = self.batch_intended_market_offset
        if off[0] != 0 or off[-1] != len(self.batch_intended_market_id):
            raise ValueError("batch_intended_market_offset must start at 0 and end at len(batch_intended_market_id)")
        if not np.all(off[:-1] <= off[1:]):
            raise ValueError("batch_intended_market_offset must be non-decreasing")

        edge_off = self.location_out_edge_offset
        if edge_off[0] != 0 or edge_off[-1] != len(self.location_out_edge_id):
            raise ValueError("location_out_edge_offset must start at 0 and end at len(location_out_edge_id)")
        if not np.all(edge_off[:-1] <= edge_off[1:]):
            raise ValueError("location_out_edge_offset must be non-decreasing")
        if len(self.location_out_edge_id) > 0 and np.max(self.location_out_edge_id) >= self.n_edges:
            raise ValueError("location_out_edge_id entries must be < n_edges")
        if np.any(self.edge_src_location_id >= self.n_locations):
            raise ValueError("edge_src_location_id entries must be < n_locations")
        if np.any(self.edge_dst_location_id >= self.n_locations):
            raise ValueError("edge_dst_location_id entries must be < n_locations")
        for loc_id in range(self.n_locations):
            start = int(edge_off[loc_id])
            end = int(edge_off[loc_id + 1])
            if np.any(self.edge_src_location_id[self.location_out_edge_id[start:end]] != loc_id):
                raise ValueError("location_out_edge CSR must group edges by matching source location")

        _len("location_has_behavior", self.location_has_behavior, self.n_locations)
        _len("location_verify_prob", self.location_verify_prob, self.n_locations)
        _len("location_decommission_prob", self.location_decommission_prob, self.n_locations)
        _len("location_reactivate_prob", self.location_reactivate_prob, self.n_locations)
