from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


class OrgType(StrEnum):
    """ defines agent types in the simulation"""
    OBP = "OBP"                         # On-Board Partners
    WHOLESALER = "WHOLESALER"           # wholesalers, disributors
    LOCAL_ORG = "LOCAL_ORG"             # terminals: pharmacies, hospitals
    NMVO = "NMVO"
    EMVO = "EMVO"

class PackState(StrEnum):
    """ defines possible package states in sim"""
    UPLOADED = "UPLOADED"               # package manufactured and uploaded to EMVO, but has not reached NMVO (cannot be sold)
    ACTIVE = "ACTIVE"                   # package in NMVO, can be sold (decomissioned)
    DECOMISSIONED = "DECOMISSIONED"

class ProductCodeScheme(StrEnum):
    """ EMVO supports both, so we must support both as well"""
    GTIN = "GTIN"
    PPN = "PPN"

class DecommissionReason(StrEnum):
    """EMVS decommission status reasons (FMD Art. 40 set, pragmatic subset).

    A pack leaves the Active state for one of these reasons; ``DISPENSED`` is the normal
    point-of-dispense outcome at a pharmacy/hospital, the rest model loss/diversion/recall.
    """
    DISPENSED = "DISPENSED"
    STOLEN = "STOLEN"
    EXPORTED = "EXPORTED"
    DESTROYED = "DESTROYED"
    SAMPLED = "SAMPLED"
    LOCKED = "LOCKED"
    RECALLED = "RECALLED"

class AlertCode(StrEnum):
    """EMVS verification alerts raised by the national system on a rule violation.

    Maps to the EMVO alert space: UNKNOWN_SERIAL ~ A2 (serial not in repository),
    ALREADY_DECOMMISSIONED ~ A7 (pack already inactive), DUPLICATE_SERIAL (same serial
    active in >1 physical pack = clone), OUT_OF_MARKET (scanned in an NMVS the pack was
    never provisioned to = parallel trade / smuggling).
    """
    UNKNOWN_SERIAL = "UNKNOWN_SERIAL"
    OUT_OF_MARKET = "OUT_OF_MARKET"
    ALREADY_DECOMMISSIONED = "ALREADY_DECOMMISSIONED"
    DUPLICATE_SERIAL = "DUPLICATE_SERIAL"

@dataclass(frozen=True)
class Organization:
    ext_id: str                         # external organization id
    org_type: OrgType

@dataclass(frozen=True)
class Location:
    ext_id: str
    org_ext_id: str
    market_code: str
    postal_code: str


@dataclass(frozen=True)
class LocationEdge:
    """Directed location graph edge with transport cost and throughput capacity."""
    src_location_ext_id: str
    dst_location_ext_id: str
    cost: float
    capacity: int
    lead_time_ticks: int = 0      # transit ticks before a shipment arrives (0 = same tick)

@dataclass(frozen=True)
class ProductCode:
    scheme: ProductCodeScheme
    value: str
    is_primary: bool=False

@dataclass(frozen=True)
class Product:
    ext_id: str
    codes: tuple[ProductCode, ...]                           # global trade item number (preferred over PPN/IFA standard according to EMVO website. However may need to be changed)
    atc_code: str | None = None                              # WHO ATC classification, drives substitutability (e.g. "C10AA05")
    name: str | None = None
    pack_size: int | None = None                             # units per pack (dose/form granularity not modeled)

@dataclass(frozen=True)
class Batch:
    ext_id: str
    product_ext_id: str
    manufacturer_org_ext_id: str 
    intended_markets: tuple[str, ...]

@dataclass(frozen=True)
class Pack:
    ext_id: str 
    product_ext_id: str
    batch_ext_id: str
    serial: str 
    initial_market_code: str 
    initial_location_ext_id: str
    initial_state: PackState=PackState.UPLOADED

""" Model Behavior"""
@dataclass(frozen=True)
class LocationBehavior:
    verify_prob: float
    decomission_prob: float 
    reactivate_prob: float

@dataclass(frozen=True)
class Scenario:
    organizations: list[Organization]
    locations: list[Location]
    products: list[Product]
    batches: list[Batch]
    packs: list[Pack]
    location_edges: list[LocationEdge] = field(default_factory=list)
    behavior_by_location: dict[str, LocationBehavior] = field(default_factory=dict)
    seed: int = 42
