# PharmaSim (MVP)

### Problem overview

Model EU-wide pharmaceutical distribution to study cross-border theft and counterfeiting, starting from synthetic scenarios compatible in spirit with EMVO/NMVO-style data.

### MVP definition

Python defines policy and initial state; a C++ SoA kernel runs the simulation and an append-only event log. One EMVO hub, OBPs, wholesalers, local orgs (pharmacies/hospitals), and NMVOs appear as organization types on locations.

### Architecture

> **Note (2026-06-02):** The `canonical` data pipeline and transaction/experiment scaffolding are currently implemented entirely in Python for fast iteration. These modules may later be partially migrated into the C++ engine if per-tick injection of anomalies or richer event semantics is needed. The native `engine_input.v6` ABI was intentionally left unchanged during this pass to keep risk low.

Data sources → Canonical pipeline → Policy/Scenario → Compiler → Engine → Analytics → I/O

1. **Canonical** (`python/canonical/`): ETL loaders (BG pharmacy registry JSON, EMA SPOR CSV), geocoding with OSM cache, geography-aware routing, Bulgaria scenario assembly, synthetic transaction generation, experiment bundle orchestration. Currently Python-only; may partially migrate to C++ later.
2. **Policy** (`python/policy/`): `Scenario`, locations, edges, packs, transaction lifecycle contracts (`TransactionIntent`, `TransactionPlan`, `AnomalyLabel`).
3. **Compiler** (`python/compiler/`): validate, dense IDs, `EngineInput` columnar payload + route CSR. Edges are always compiled from dense columns; default ingress converts `scenario.location_edges`, or callers pass optional `edge_*` numpy arrays for large builders.
4. **Engine** (`cpp/engine/`): pack/location state, phased `run_ticks`, events.
5. **Runtime** (`python/runtime/`): `native_bridge`, `simulation_viz`.
6. **Analytics** (`python/analytics/`): anomaly injection (volume spikes, cross-market), baseline detectors (z-score, market mismatch), evaluation metrics (precision/recall/F1).

Users typically compile and run via `python/runtime` without hand-rolling the C++ ABI.

### End-to-end flow

**Data-driven workflow (new):**

`data/data.json` + `data/spor_locations.csv` → `bulgaria_registry_scenario()` / `bulgaria_registry_experiment_bundle()` → `compile_scenario` → `create_native_simulator` → `run_ticks` → event log and reports.

**Synthetic workflow (existing):**

`two_markets_demo()` → `compile_scenario` → same engine path.

**Large synthetic workflow (performance path):**

`multi_market_sparse_scenario_precomputed()` → `compile_scenario(..., edge_src=..., edge_dst=..., edge_cost=..., edge_capacity=...)` → same engine path. Skips building millions of `LocationEdge` Python objects; use for stress-scale graphs.

After changing C++ or the `EngineInput` layout: rebuild the extension (e.g. `scripts/setup.sh` or CMake + `nanobind` target) and reinstall the editable package (`uv pip install -e .`) so tests load a matching `_pharmasim_native`.

### EngineInput and schema

Stable ABI string `engine_input.v6`: keep `python/compiler/types.py` (`ENGINE_INPUT_SCHEMA_VERSION`) and `cpp/engine/enums.hpp` in sync when adding or reordering columns. Bindings must load every new field (`cpp/bindings/`).

**Compile API:**

Single entry point: `compile_scenario(scenario, *, edge_src_location_id=..., edge_dst_location_id=..., edge_cost=..., edge_capacity=...)`.

| Edge source | How |
| ----------- | --- |
| Default | Convert `scenario.location_edges` to columns inside the compiler |
| Large builders | Pass all four `edge_*` arrays; keep `scenario.location_edges` empty |

Both modes share the same internal columnar compile path (no ABI change).

### Simulation kernel (current)

- **Static inventory**: fixed `n_packs`; no runtime pack creation.
- **Per tick** (`Simulator::run_ticks`): (1) location phase — demand policies; (2) supply phase — supply policies and scheduled first hops; (3) shipment phase — due moves along edges (multihop via precomputed route CSR, `edge_lead_time_ticks`); (4) pack behavior — verify / decommission / reactivate probabilities per location. Stochastic per-pack graph walks exist in code but are **not** invoked from this tick loop.
- **Policies** (column `policy_id` + params): demand `0/1/2` (none, constant rate, Poisson on `location_demand_poisson_lambda`); supply `0/1/2` (none, wholesaler cap toward preferred edge if downstream backlog, OBP pool activation then ship). First-hop moves respect per-edge `edge_capacity` within the tick.
- **Registry**: mirrored from physical state in v1; not full NMVS/EMVO API fidelity.

Design notes: `simulation_realism.md`.

```mermaid
flowchart TB
  subgraph data [Data_sources]
    BG_REG[BGBulgaria_pharmacy_registry_JSON]
    SPOR[EMASPOR_locations_CSV]
  end
  subgraph canonical [Python_canonical_pipeline]
    LOAD[Loaders_normalize]
    GEO[Geocoding_OSM_cache]
    ROUTE[Routing_graph]
    TXGEN[Synthetic_transaction_generation]
  end
  subgraph compile [Python_compiler]
    SC[Scenario]
    EI[EngineInput_SoA]
  end
  subgraph kernel [C++_Simulator_run_ticks]
    PHY[Physical_pack_columns]
    REG[Registry_mirror_columns]
    CSR[Graph_and_route_CSR]
    RNG[mt19937_64]
    LOG[EventLog_append]
  end
  subgraph analytics [Python_analytics]
    INJECT[Anomaly_injectors]
    DETECT[Baseline_detectors]
    EVAL[Eval_metrics]
  end
  BG_REG --> LOAD
  SPOR --> LOAD
  LOAD --> GEO
  GEO --> ROUTE
  ROUTE --> SC
  TXGEN --> SC
  SC --> EI
  EI --> PHY
  EI --> CSR
  PHY --> REG
  RNG --> PHY
  CSR --> PHY
  PHY --> LOG
  LOG --> DETECT
  INJECT --> TXGEN
  INJECT --> SC
  DETECT --> EVAL
```

### End-to-end Bulgaria fraud/smuggling loop

A single reproducible loop now ties the data pipeline, the C++ engine, and detection
together so anomalies appear in the **real engine event log** (not a parallel synthetic
plan) and detectors are scored on that log:

```
data (BG registry + SPOR)
  -> batch geocode cache            scripts/geocode_bg.py  (resumable, OSM + city fallback)
  -> connectivity-first scenario    python/canonical/scenario_builder.py
  -> compile + calibrate            python/runtime/experiments.py  (push policies)
  -> baseline run_ticks             ground-truth event log
  -> perturbation (Python driver)   python/policy/perturbations.py  -> perturbed EngineInput + labels
  -> perturbed run_ticks            observable event log
  -> log detectors + scoring        python/analytics/{fraud,experiment}.py
```

Run it:

```bash
# one-time, resumable; populates data/geocode_cache_bg.json
uv run python scripts/geocode_bg.py

# build -> compile -> simulate -> export ground-truth log + map plot
TICKS=300 ./scripts/run_bulgaria.sh

# one-call baseline-vs-perturbed experiment (precision/recall/F1 vs labels)
uv run python -c "from analytics.experiment import run_fraud_experiment; \
from policy.perturbations import PerturbationSpec, PerturbationType as T; \
r=run_fraud_experiment(PerturbationSpec(ptype=T.SMUGGLING)); print(r.metrics)"
```

Perturbation types (`PerturbationType`): `VOLUME_SPIKE`, `SMUGGLING` (cross-market
diversion), `THEFT` (distributor shrinkage), `COUNTERFEIT` (cloned-serial injection). Each
rewrites calibrated engine parameters and emits ground-truth `PerturbationLabel`s; the
matching log detector recovers the injected sites. Map + baseline-vs-perturbed flow plots
live in `python/runtime/simulation_viz.py` (`plot_location_graph`, `plot_flow_comparison`).

> Engine-fit notes: the supply phase ships along each node's single *preferred* edge, so the
> calibration disperses import hubs geographically and stocks every wholesaler to get broad,
> realistic OBP -> wholesaler -> pharmacy coverage; routing precompute now uses
> `scipy.sparse.csgraph` (compile time dropped from the former bottleneck to a few ms).

### Next steps:
- Enrich event log with transaction/order semantics for richer observability
- Per-node fan-out distribution (engine currently ships along a single preferred edge)
- Push anomaly injection into per-tick engine loop (currently pre-computed in Python)
- Vectorized pack generation / compiled artifact caching for remaining build-time overhead
- Migrate stable canonical pipeline pieces into C++ EngineInput columns if scalability demands it
- different order policies by different medicine types -- substitutability of medicines by `ATC (Anatomical Therapeutic Classification)` (prefix tree data structure)
- More complex/realistic `WHOLESALER` logic for allocation, fairness
- More complex/realistic `OBP` production behavior, perhaps with production scheduling
- More realistic/complex penalty (modular, so can be switched for experimentation)
- Advanced detector methods beyond z-score / rule baselines
#### Decided
- Real actor location data
  - SPOR (European Medicines Agency) — **implemented** in `python/canonical/loaders.py`
  - Bulgarian pharmacy registry — **implemented** in `python/canonical/loaders.py`
- Geocoding via OpenStreetMap — **implemented** with cache + resumable batch CLI (`scripts/geocode_bg.py`) and city-centroid fallback in `python/canonical/geocoding.py`
- Connectivity-first scenario build (no more 0-edge trim) — **implemented** in `python/canonical/scenario_builder.py`
- Fast `scipy.sparse.csgraph` route precompute — **implemented** in `python/compiler/compile.py`
- Fraud as a Python policy-driver on the real engine log — **implemented** in `python/policy/perturbations.py`
- Log-based detectors + precision/recall/F1 scoring + one-call experiment — **implemented** in `python/analytics/{fraud,experiment}.py`
- Synthetic transaction lifecycle with anomaly injection — **implemented** in `python/canonical/transactions.py`
- Columnar edge compile in single `compile_scenario` API — **implemented** in `python/compiler/compile.py`
- Visualizations: geographic graph + baseline-vs-perturbed flow — **implemented** in `python/runtime/simulation_viz.py`
- --> argue based on plausibility of simulation





### Directory structure

| Path                | Role                                                         |
| ------------------- | ------------------------------------------------------------ |
| `python/canonical/` | BG data loaders, OSM geocoding, routing, scenario assembly, synthetic transaction generation, experiment bundles |
| `python/policy/`    | Scenarios, models, transaction lifecycle contracts (`transactions.py`) |
| `python/compiler/`  | AoS → SoA compile; edges always ingested as dense columns |
| `python/runtime/`   | Native bridge, `simulation_viz.py`                           |
| `python/analytics/` | Reports, anomaly injection & detection (`fraud.py`)          |
| `cpp/engine/`       | Kernel                                                       |
| `cpp/bindings/`     | nanobind module                                              |
| `schemas/`          | Version and enum docs                                        |
| `tests/`            | pytest: canonical pipeline, transactions, compiler, validate, dynamics, bridge, viz |


Tooling: **uv** (Python), **CMake** + **Ninja** (C++ / extension).

### Logs and visualization

See `python/runtime/native_bridge.py` and `python/runtime/simulation_viz.py`: `events_as_records`, `export_run_report`, `run_ticks_with_hook`, optional plots (`uv sync --extra viz`).

### Usage

- `scripts/setup.sh` — uv env + CMake build of `_pharmasim_native`
- `scripts/run_simulation.sh` — example scripted run
- `uv run pytest` — full test suite (`pyproject.toml` sets `pythonpath` for `python/`)

### Performance benchmarks

Benchmarks time the full pipeline in four explicit phases that sum to total wall clock:

1. `build_scenario`
2. `compile_scenario`
3. `create_simulator`
4. `run_ticks`

Phase breakdown is printed at the end of the pytest run (`tests/conftest.py`).

```bash
# Fast sanity (~sub-second)
PHARMASIM_BENCH_PROFILE=quick uv run pytest tests/bench.py -v

# Bulgaria registry scale
PHARMASIM_BENCH_PROFILE=bulgaria uv run pytest tests/bench.py -v

# Full stress profile (can take hours; 4M packs by default)
PHARMASIM_BENCH_PROFILE=stress uv run pytest tests/bench.py -v
```

Optional env vars:

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `PHARMASIM_BENCH_PROFILE` | `stress` | `quick`, `bulgaria`, or `stress` |
| `PHARMASIM_BENCH_TICKS` | profile-specific | Tick count for `run_ticks` |
| `PHARMASIM_BENCH_MARKETS` | `DE,FR,IT,UK` | Stress profile markets |
| `PHARMASIM_BENCH_LOCATIONS` | `10000` | Locations per market (stress) |
| `PHARMASIM_BENCH_PACKS` | `1000000` | Packs per market (stress) |
| `PHARMASIM_BENCH_PRECOMPUTED_EDGES` | `1` | Use precomputed-edge compile path in stress profile; set `0` for legacy list-edge path |

**Build-time note:** scenario construction and compile still dominate wall clock at large scale (especially millions of `Pack` objects and multihop route precompute). Passing `edge_*` columns avoids materializing `LocationEdge` lists; see `changes_31_05.md` §12–§13.

@Han Wu [hanwuh@ethz.ch](mailto:hanwuh@ethz.ch)