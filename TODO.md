# TODO — Bulgaria EMVO realism & fraud/anomaly detection

Scope: single-country (BG) simulation calibrated for plausible baselines and detector thresholds.
Tick convention: 1 tick ≈ 1 day unless noted.

## Model gaps (implement before calibration pays off)

- [ ] EMVO lifecycle: `UPLOADED` → NMVS provisioning → `ACTIVE` → decommission (packs currently seed as `ACTIVE` at wholesalers; no hub upload/import phase)
- [ ] Explicit NMVO/EMVO org types on locations (registry is implicit per `market_code` today)
- [ ] Real wholesaler–pharmacy/hospital edges (replace haversine k-NN graph with observed or licensed supply links)
- [ ] Multi-product Bulgaria scenario (`data/products.csv` + volume weights; currently one synthetic product)
- [ ] MAH/OBP nodes from SPOR manufacturers, not promoted wholesalers
- [ ] Per-tick or event-driven scan model (replace independent Bernoulli verify/decommission per pack per tick)
- [ ] Detector baselines: fit thresholds from calibrated normal runs, not hand-tuned z-scores

## Data to collect — map to engine parameters

### A. NMVS / FMD transaction stream (highest priority for fraud)

Source target: BDA NMVS aggregates or anonymized transaction logs (verify / decommission / reactivate).

| Statistic | Maps to | Notes |
|-----------|---------|-------|
| Verifications per site per day (p50/p95) | `location_verify_prob` | Split community pharmacy vs hospital; dispense should be ~mandatory, not 0.12 |
| Decommissions per site per day by reason | `location_decommission_prob`, `location_decommission_reason_id` | Reason mix: DISPENSED, STOLEN, EXPORTED, DESTROYED, SAMPLED, LOCKED, RECALLED |
| Reactivation attempts & acceptance rate | `location_reactivate_prob`, `reactivation_window_ticks` | FMD window ≈ 10 days |
| Alert incidence rates (per 10⁴ scans) | detector baselines | `UNKNOWN_SERIAL`, `OUT_OF_MARKET`, `ALREADY_DECOMMISSIONED`, `DUPLICATE_SERIAL` |
| Cross-border verify share | parallel-trade baseline | Drives `OUT_OF_MARKET` false-positive budget |
| Time-of-day / DOW seasonality multipliers | demand & scan scheduling | Currently absent |

### B. Physical distribution & logistics

Source target: wholesaler shipment records, BG pharmacy registry topology, optional carrier SLAs.

| Statistic | Maps to | Notes |
|-----------|---------|-------|
| Lane distance & lead time by role | `edge_cost`, `edge_lead_time_ticks` | OBP→wholesaler, wholesaler→pharmacy, wholesaler→hospital |
| Shipment volume per edge per day (mean, variance, peak) | `edge_capacity` | Steady vs peak; currently hardcoded in `routing.py` |
| Multihop path length distribution | route CSR validation | % single-hop vs multi-hop to terminal |
| Preferred-supplier concentration | `location_preferred_supplier_edge_id` | Share on primary vs alternate upstream edge |
| Initial stock by tier | `location_initial_on_hand`, pack seed counts | `packs_per_obp`, `packs_per_wholesaler` |

### C. Demand & inventory (terminals & wholesalers)

Source target: reimbursement/Rx statistics (NHIF), wholesaler order data, or published market volumes.

| Statistic | Maps to | Notes |
|-----------|---------|-------|
| Dispensing rate per pharmacy/hospital (λ or constant) | `location_demand_poisson_lambda`, `location_demand_const_rate` | By site type and region |
| Product mix shares | `product_demand_weight` | ATC-level volumes; use `apply_product_demand_weights` |
| Substitute groups | `product_substitute_*` CSR | ATC level 4+ clinical interchangeability |
| Reorder policy (s, S), batch size, min interval | `location_reorder_point_s`, `location_order_up_to_S`, `location_min_order_interval_ticks` | Wholesaler (s,S) today is default 20 |
| Stockout / unfulfilled rate | `location_unfulfilled_unit_penalty` calibration | Implicit shortage cost |
| On-hand, backlog, pipeline time series | `location_*` aggregates | Validate tier dynamics |

### D. Supply & production

Source target: MAH release reports, wholesaler throughput, import statistics.

| Statistic | Maps to | Notes |
|-----------|---------|-------|
| Packs released per MAH/OBP per period | OBP `location_supply_capacity_per_tick` | Push policy id 2 after `calibrate_supply_policies` |
| Wholesaler outbound throughput | wholesaler capacity | Currently 3–10 packs/tick ad hoc |
| Downstream allocation fairness | fan-out policy design | Engine ships one preferred edge; measure concentration |
| Batch size & intended-market provisioning | `batch_intended_market_*` | BG-only vs re-imported parallel-trade SKUs |

### E. Network structure (BG registry + SPOR — partially implemented)

| Statistic | Use | Current gap |
|-----------|-----|-------------|
| Active pharmacy / hospital / wholesaler counts by region | node sampling, `max_locations` | Registry loaded; roles partly inferred |
| Wholesaler service areas (actual customers) | real edges | k-NN geographic proxy only |
| Pharmacy degree (suppliers per terminal) | graph validation | Synthetic |
| Hospital vs community pharmacy split | per-type behavior & demand | Partial via `a_type` in registry |
| MAH locations (SPOR) | OBP placement | Not wired into Bulgaria builder |

### F. Anomaly & fraud baselines (detector calibration)

| Statistic | Use |
|-----------|-----|
| Normal verify/decommission volume per location over time | z-score / rate detectors (`detect_volume_change`) |
| Typical pack journey (location sequence → decommission) | route-trajectory anomalies |
| Documented incident rates (theft, counterfeit, diversion) | ground-truth prior for injected vs observed |
| Alert → investigation outcome (if available) | precision/recall targets for rule baselines |

## Calibration workflow

1. Build BG scenario from registry + SPOR + geocode cache → `build_bulgaria_simulation_input`
2. Replace synthetic routing/behavior defaults with tables from sections A–E
3. Run baseline `run_ticks` → export event log → compute descriptive stats (section F)
4. Set detector thresholds at p99 of baseline metrics
5. Validate perturbations (`COUNTERFEIT`, `SMUGGLING`, `THEFT`, `VOLUME_SPIKE`) sit above baseline tails

## Thought experiments (post-calibration)

- Anomalous routes (pack visits unexpected location before decommission)
- Demand spikes decoupled from verify volume (legitimate surge vs diversion)
- Clone timing: second physical pack on serial after NMVS retirement
- Wholesaler `STOLEN` decommission rate vs sector norm

## Decided / in repo

- BG pharmacy registry + EMA SPOR loaders, OSM geocode cache, connectivity-first scenario build
- EMVS verify/decommission/reactivate + typed alerts in C++ event log
- Fraud perturbations + log-based detectors + precision/recall/F1 (`analytics/experiment.py`)
- Product catalog stub (`data/products.csv`) — not yet used in Bulgaria builder
