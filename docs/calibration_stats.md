# Descriptive Statistics for Simulation Parameter Calibration

## NMVS/FMD transaction statistics
- Verifications per site per day (average)
- Decomissions per site per day by reason (average)
- Reactivation attempts
- Reactivation acceptance rate
- Import/Export number
- Seasonality pattenrs

## Physical distribution
- Shipment times, distance, cost metrics
- Shipment volume per day between nodes
- How concentrated are suppliers (does a terminal concentrate orders on a single supplier or is it spread out?)

## Demand and Inventory
- Dispensing rate per pharmacy or hospital
- Product mix shares, perhaps ATC-level volumes
- On-hand inventory, backlog, pipeline amount time series data (or averages)

## Supply and Production
- Number of packs entering Bulgaria per time period
- Batch size

## More Detailed Network Structure
- Wholesaler service areas, service volume
- Differences in demand behavior between hospitals and pharmacies
- Average number of suppliers (degree) for hospitals/pharmacies

## Anomaly & Fruad baselines
- Normal verify/decomission volume per location over time, or per node type over time
- Typical pack journey (location sequence -> decomission) to detect route anomalies
- Documented incident rates (theft, counterfeit, etc)

Feel free to add to or modify the above!