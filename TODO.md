1. refine pack status transitions to accurately reflect EMVO, allow for constant rate of failure/anomaly
2. descriptive metrics/stats for simulation realism (parameters to collect or estimate from real/historical data)
  - **Routing & logistics**
  - average / distribution of route distance (km) and travel time by lane type (OBP→wholesaler, wholesaler→pharmacy, wholesaler→hospital, cross-border)
  - `edge_cost` (transport €/pack or €/km) and `edge_lead_time_ticks` (order-to-delivery latency)
  - `edge_capacity` (packs shipped per tick) — mean, variance, and peak vs steady-state by edge role - multihop route length and frequency (how often goods traverse >1 hop before terminal), perhaps as histogram - preferred-supplier concentration (share of volume on primary vs alternate upstream edges)
    - **Inventory & ordering (wholesalers / terminals)**
      - stock build-up and draw-down time series: on-hand, backlog, pipeline-in-transit at each tier
      - demand arrival patterns at pharmacies/hospitals: constant rate vs Poisson (`location_demand_const_rate`, `location_demand_poisson_lambda`), including seasonality and day-of-week effects
      - reorder policy stats: `(s, S)` levels (`location_reorder_point_s`, `location_order_up_to_S`), `location_base_stock_level`, `location_ewma_alpha` for demand forecasting
      - `location_min_order_interval_ticks` and typical order batch sizes
      - stockout / unfulfilled-demand rate and `location_unfulfilled_unit_penalty` (implicit cost of shortage)
      - initial conditions: `location_initial_on_hand`, `location_initial_backlog`, `location_initial_pipeline_outstanding`
    - **Supply & production (OBPs / wholesalers)**
      - `location_supply_capacity_per_tick` by org type (release/shipment throughput)
      - allocation fairness across downstream customers (currently single preferred edge — useful baseline for fan-out redesign)
      - packs released per OBP per period (`packs_per_obp`), wholesaler intake (`packs_per_wholesaler`)
      - product mix and substitutability (`product_demand_weight`, ATC substitute groups)
    - **EMVS / NMVS scan behavior**
      - verify rate at dispense (`location_verify_prob`) by site type (community pharmacy vs hospital)
      - decommission rate and reason mix (`location_decommission_prob`, `location_decommission_reason_id`: dispensed, stolen, exported, destroyed, sampled, locked, recalled)
      - reactivation attempts and success within FMD window (`location_reactivate_prob`, `reactivation_window_ticks`)
      - alert baselines: `UNKNOWN_SERIAL`, `OUT_OF_MARKET`, `ALREADY_DECOMMISSIONED`, `DUPLICATE_SERIAL` incidence per market
    - **Network & actor structure**
      - org-type counts and geographic dispersion (OBP hubs, wholesalers, pharmacies, hospitals per region)
      - degree distribution and clustering of the supply graph (how many suppliers per terminal, how many customers per wholesaler)
      - cross-border / parallel-trade flow share between markets
    - **Transaction & anomaly baselines** (for later calibration and detection thresholds)
      - normal transaction volume per location and per market over time
      - typical pack journey (sequence of locations before decommission)
      - baseline shrinkage/theft/counterfeit rates to compare against injected anomalies
3. Thought experiment -- implant anomalous events (ex: anomalous routes, unsual demands, unusual package travel routes)
4. (Once real data available) calibrate simulation parameters to historical data, diff with current data for anomaly detection

