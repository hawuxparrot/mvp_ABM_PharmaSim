# EMVO / NMVS data request

One-page ask to turn the PharmaSim HITL AlertLabeler demo from a plausible prototype into
a calibrated tool. It is generated to mirror the dashboard's *EMVO data request* tab
(`python/learning/data_request.py`) and the data map in [`TODO.md`](../TODO.md) sections A-F.

## Why we are asking

The demo already runs end to end: the C++ engine emits real EMVS-typed alerts, a hidden
operator (encoding BDA's rules) labels a queue, and an online learner recovers that policy
from feedback alone (final F1 ~0.99 on a held-out seed). What it uses today is **synthetic**:

- Scan rates are hardcoded (`location_verify_prob = 0.12`, `location_decommission_prob = 0.03`
  for every local organization), so the simulated alert **rate** is far from the real
  &lt;0.5% and cannot be steered toward the 0.05% target.
- The distribution network is a **k-NN haversine proxy** (`python/canonical/routing.py`),
  not the licensed wholesaler-pharmacy supply relationships.
- There is no software-vendor tag, no intra-day clock, no medicine value/form, and no
  real bad-actor list.

We cannot credibly claim the meeting's operational numbers (Bulgaria ~3000 alerts/week,
Germany ~35000/week, ~160M scans/year, target 0.05% alert rate, 95% sold within 3 months)
until these inputs are real. Each request below maps a meeting figure to the exact engine
parameter it would calibrate.

## The ask (anonymized aggregates are sufficient)

| Priority | Topic | Meeting signal | Simulation today | Engine parameter | Requested aggregate |
|---|---|---|---|---|---|
| 1 | Verify / decommission rates by site type | Dispense scans near-mandatory; alert rate target 0.05% | `verify_prob=0.12`, `decommission_prob=0.03` for every local org | `location_verify_prob`, `location_decommission_prob` | Verifications & decommissions per site per day, split community pharmacy vs hospital (p50/p95) |
| 1 | Alert incidence and outcomes | BG ~3000 alerts/week, most errors, ~70% from one vendor | Alerts emitted by EMVS rules; no real incidence baseline | detector thresholds, candidate rates | Alert incidence per 10^4 scans by code, and alert -> investigation outcome (real fraud vs error) |
| 1 | Licensed supply network | Realistic routes / graph of who supplies whom | k-NN haversine proxy edges, not real links | `location_edges` (src/dst/capacity/lead_time) | Wholesaler service areas: which wholesalers supply which pharmacies/hospitals, and shipment volume per lane |
| 2 | Software vendor identity | ~70% of BG alerts come from one software company | No vendor tag on locations; cannot reproduce the 70% | new per-location `software_vendor_id` | Alert counts grouped by originating software vendor (anonymized vendor codes) |
| 2 | Time-of-day / day-of-week | Wholesaler scanning a valuable medicine at 23:00 is suspicious | `tick == 1 day`; no intra-day clock or weekday seasonality | `ticks_per_day`, scan scheduling multipliers | Scan volume by hour-of-day and day-of-week (seasonality multipliers) |
| 2 | Medicine value and form | Risk depends on medicine type and form (tablets, syringes) | Single synthetic product; no value or form attribute | product value, product form, `product_demand_weight` | ATC-level volume shares, pack value bands, and dosage form per product |
| 2 | OBP -> wholesaler timing | Benchmark: 95% sold within 3 months from OBP to wholesalers | Delivery timing is a synthetic MOVE proxy | `edge_lead_time_ticks`, OBP supply capacity | Distribution of time from release to wholesaler receipt by organization and medicine type |
| 3 | Bad-actor / blacklist | Certain bad actors -> trust-score debuff, a priori high trust | Trust score = geocode quality; no runtime blacklist | location trust/quality score, blacklist flag | List (or risk tier) of sanctioned or previously implicated actors to seed a trust debuff |

## What we return with the data

- A Bulgaria simulation whose alert **rate** and **weekly volume** match the real system,
  so detector thresholds and the 0.05% target become meaningful.
- The learned operator policy expressed in readable terms (feature weights and rule
  firing rates), validated against real alert-outcome labels.
- A network view on the true supply graph instead of a geographic proxy.

## Cross-references

- Full parameter map: [`TODO.md`](../TODO.md) sections A (NMVS transaction stream),
  B (physical distribution), C (demand & inventory), D (supply & production),
  E (network structure), F (anomaly & fraud baselines).
- Descriptive-statistics skeleton to be filled with the aggregates:
  [`calibration_stats.md`](calibration_stats.md).
- Operator rules encoding BDA's examples: [`configs/hitl/operator_rules.v1.json`](../configs/hitl/operator_rules.v1.json).
