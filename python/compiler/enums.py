"""
Stable uint8 encodings for policy.models.OrgType and policy.models.PackState.

These values are part of the EngineInput ABI — keep them in sync with C++ ``enums.hpp`` for the native layer.
Bump ``compiler.types.ENGINE_INPUT_SCHEMA_VERSION`` if numeric assignments are changed.
``EngineInput`` schema is ``engine_input.v2`` (dense location behavior arrays; see ``compile.py``).
"""

from __future__ import annotations

import numpy as np

from policy.models import OrgType, PackState

__all__ = [
    "ORG_TYPE_TO_U8",
    "U8_TO_ORG_TYPE",
    "PACK_STATE_TO_U8",
    "U8_TO_PACK_STATE",
    "org_type_u8",
    "pack_state_u8",
    "decode_org_type_u8",
    "decode_pack_state_u8",
]

# explicit integer codes. Document order for C++ mirror.
ORG_TYPE_TO_U8: dict[OrgType, int] = {
    OrgType.OBP: 0,
    OrgType.WHOLESALER: 1,
    OrgType.LOCAL_ORG: 2,
    OrgType.NMVO: 3,
    OrgType.EMVO: 4,
}

PACK_STATE_TO_U8: dict[PackState, int] = {
    PackState.UPLOADED: 0,
    PackState.ACTIVE: 1,
    PackState.DECOMISSIONED: 2,
}


def _invert_org_type_map(m: dict[OrgType, int]) -> dict[int, OrgType]:
    inv: dict[int, OrgType] = {}
    for k, v in m.items():
        if v < 0 or v > 255:
            raise ValueError(f"enum code out of uint8 range: {k!r} -> {v}")
        if v in inv:
            raise ValueError(f"duplicate uint8 code {v} for {k!r} and {inv[v]!r}")
        inv[v] = k
    return inv


def _invert_pack_state_map(m: dict[PackState, int]) -> dict[int, PackState]:
    inv: dict[int, PackState] = {}
    for k, v in m.items():
        if v < 0 or v > 255:
            raise ValueError(f"enum code out of uint8 range: {k!r} -> {v}")
        if v in inv:
            raise ValueError(f"duplicate uint8 code {v} for {k!r} and {inv[v]!r}")
        inv[v] = k
    return inv


U8_TO_ORG_TYPE: dict[int, OrgType] = _invert_org_type_map(ORG_TYPE_TO_U8)
U8_TO_PACK_STATE: dict[int, PackState] = _invert_pack_state_map(PACK_STATE_TO_U8)


def _assert_maps_cover_enums() -> None:
    if set(ORG_TYPE_TO_U8.keys()) != set(OrgType):
        missing = set(OrgType) - set(ORG_TYPE_TO_U8.keys())
        extra = set(ORG_TYPE_TO_U8.keys()) - set(OrgType)
        raise AssertionError(f"ORG_TYPE_TO_U8 must match OrgType: missing={missing!r} extra={extra!r}")
    if set(PACK_STATE_TO_U8.keys()) != set(PackState):
        missing = set(PackState) - set(PACK_STATE_TO_U8.keys())
        extra = set(PACK_STATE_TO_U8.keys()) - set(PackState)
        raise AssertionError(
            f"PACK_STATE_TO_U8 must match PackState: missing={missing!r} extra={extra!r}"
        )
    if len({ORG_TYPE_TO_U8[k] for k in ORG_TYPE_TO_U8}) != len(ORG_TYPE_TO_U8):
        raise AssertionError("ORG_TYPE_TO_U8 values must be unique")
    if len({PACK_STATE_TO_U8[k] for k in PACK_STATE_TO_U8}) != len(PACK_STATE_TO_U8):
        raise AssertionError("PACK_STATE_TO_U8 values must be unique")


_assert_maps_cover_enums()


def org_type_u8(o: OrgType) -> np.uint8:
    try:
        return np.uint8(ORG_TYPE_TO_U8[o])
    except KeyError as e:
        raise KeyError(f"unknown OrgType: {o!r}") from e


def pack_state_u8(p: PackState) -> np.uint8:
    try:
        return np.uint8(PACK_STATE_TO_U8[p])
    except KeyError as e:
        raise KeyError(f"unknown PackState: {p!r}") from e


def decode_org_type_u8(code: int | np.uint8) -> OrgType:
    c = int(code)
    if c not in U8_TO_ORG_TYPE:
        raise ValueError(f"invalid OrgType u8 code: {c}")
    return U8_TO_ORG_TYPE[c]


def decode_pack_state_u8(code: int | np.uint8) -> PackState:
    c = int(code)
    if c not in U8_TO_PACK_STATE:
        raise ValueError(f"invalid PackState u8 code: {c}")
    return U8_TO_PACK_STATE[c]
