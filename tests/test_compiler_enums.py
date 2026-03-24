"""Sanity checks for compiler enum tables (ABI stability)."""

import numpy as np
import pytest

from compiler.enums import (
    ORG_TYPE_TO_U8,
    PACK_STATE_TO_U8,
    U8_TO_ORG_TYPE,
    U8_TO_PACK_STATE,
    decode_org_type_u8,
    decode_pack_state_u8,
    org_type_u8,
    pack_state_u8,
)
from policy.models import OrgType, PackState


def test_org_type_maps_cover_all_members() -> None:
    assert set(ORG_TYPE_TO_U8) == set(OrgType)
    assert len({ORG_TYPE_TO_U8[k] for k in ORG_TYPE_TO_U8}) == len(ORG_TYPE_TO_U8)


def test_pack_state_maps_cover_all_members() -> None:
    assert set(PACK_STATE_TO_U8) == set(PackState)
    assert len({PACK_STATE_TO_U8[k] for k in PACK_STATE_TO_U8}) == len(PACK_STATE_TO_U8)


def test_org_type_roundtrip() -> None:
    for o in OrgType:
        u = org_type_u8(o)
        assert isinstance(u, np.uint8)
        assert decode_org_type_u8(int(u)) == o
        assert U8_TO_ORG_TYPE[int(u)] == o


def test_pack_state_roundtrip() -> None:
    for p in PackState:
        u = pack_state_u8(p)
        assert isinstance(u, np.uint8)
        assert decode_pack_state_u8(int(u)) == p
        assert U8_TO_PACK_STATE[int(u)] == p


def test_decode_invalid_org_type_raises() -> None:
    bad = max(U8_TO_ORG_TYPE) + 1
    with pytest.raises(ValueError, match="invalid OrgType"):
        decode_org_type_u8(bad)


def test_decode_invalid_pack_state_raises() -> None:
    bad = max(U8_TO_PACK_STATE) + 1
    with pytest.raises(ValueError, match="invalid PackState"):
        decode_pack_state_u8(bad)
