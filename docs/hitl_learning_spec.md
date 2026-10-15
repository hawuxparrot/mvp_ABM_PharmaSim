# PharmaSim HITL Alert Learning Specification

## 1. Purpose

Build an alert triage loop on top of the current PharmaSim event log.

The loop must learn a hidden operator policy from operator feedback.

The first demo must use the existing simulator without a C++ ABI change.

The demo must show that the learner approaches the hidden operator policy.

The demo must also show the simulation and the learning process.

## 2. Main decision

The first demo is an online supervised learning system.

The learner predicts whether an alert case is unsafe.

The operator returns a safe or unsafe label.

The learner updates after each returned label.

This process is not full reinforcement learning.

Full reinforcement learning needs actions that change later simulation states.

Active label selection is optional for the first milestone.

It becomes required for the final demo.

## 3. Existing repository contract

Use these existing components.

1. Use `build_bulgaria_simulation_input()` to build the scenario.
2. Use `run_simulation()` for replay experiments.
3. Use `run_ticks_with_hook()` for the live tick loop.
4. Use `events_as_records()` as the event source.
5. Use `apply_perturbation()` to create unsafe behavior.
6. Use `PerturbationLabel` only for scoring.
7. Use the current Matplotlib dependency for plots.

Do not change `engine_input.v8` for this work.

Do not add a dashboard framework during the first implementation.

## 4. System boundary

The system has five parts.

1. The simulator produces events.
2. The case builder converts events into alert cases.
3. The hidden operator labels selected alert cases.
4. The learner predicts labels and updates its parameters.
5. The evaluator records metrics and plots.

The hidden operator and the learner must remain separate.

The learner must not read perturbation labels.

The learner must not read hidden operator rule identifiers.

The learner must not read future events.

## 5. Alert case definition

An alert case is the unit that the learner classifies.

Create `AlertCase` as an immutable dataclass.

Use these fields.

```python
@dataclass(frozen=True)
class AlertCase:
    case_id: str
    tick: int
    pack_id: int | None
    event_type: str
    alert_code: str | None
    decommission_reason: str | None
    from_location_id: int | None
    to_location_id: int | None
    from_role: str | None
    to_role: str | None
    from_market: str | None
    to_market: str | None
    same_location: bool
    same_organization: bool
    same_city: bool
    same_postal_code: bool
    distance_km: float | None
    market_mismatch: bool
    prior_pack_events: int
    prior_pack_alerts: int
    source_recent_event_count: int
    destination_recent_event_count: int
    source_trust_score: float
    destination_trust_score: float
```

The case builder must derive these fields from observable data.

The case builder may use the scenario and canonical location data.

The case builder must not use perturbation labels.

## 6. Candidate generation

The current engine emits true EMVS alerts.

A learner needs both safe and unsafe cases.

Therefore, the case builder must create a broader candidate stream.

Create a case for each event that matches one condition below.

1. The event type is `ALERT`.
2. The event type is `DECOMMISSION` with a nonstandard reason.
3. The event type is `MOVE` and the market changes.
4. The event type is `MOVE` and both endpoints are local organizations.
5. The event type is `MOVE` and the recent source count exceeds the candidate threshold.
6. The safe-event sampler selects the event.

The safe-event sampler prevents a positive-only data set.

The sampler must use a fixed seed.

The sampler must expose `safe_sample_rate` in the demo configuration.

The first default value is `0.10`.

## 7. Hidden operator

The hidden operator simulates the human in the loop.

Load its ordered rules from a JSON file.

The learner must not read this file.

The operator returns three values.

```python
OperatorFeedback(
    label: bool,
    confidence: float,
    rule_id: str,
)
```

Use `label=True` for unsafe.

Use `label=False` for safe.

The first rule set must include these rules.

### 7.1 Unsafe rules

Apply these rules before the safe rules.

1. Label `UNKNOWN_SERIAL` as unsafe.
2. Label `DUPLICATE_SERIAL` as unsafe.
3. Label `OUT_OF_MARKET` as unsafe.
4. Label a `STOLEN` decommission as unsafe.
5. Label a hospital-to-hospital move as unsafe.
6. Label a high recent source count as unsafe.
7. Label a low source or destination trust score as unsafe.

Load the count and trust thresholds from the operator JSON file.

The operator must not read perturbation labels.

### 7.2 Safe rules

Apply these rules after the unsafe rules.

1. Label an event within the same location as safe.
2. Label an event within the same organization as safe.
3. Label a pharmacy-to-pharmacy move in one postal code as safe.
4. Label a pharmacy-to-pharmacy move below the distance limit as safe.

Use `short_distance_km` from the operator JSON file.

The first default value is `5.0` kilometers.

### 7.3 Default rule

Label all remaining cases as safe.

This default produces a conservative operator policy.

### 7.4 Ground-truth validation

Keep perturbation labels outside the operator.

The evaluator may use them after all case predictions are complete.

Use them to test whether unsafe cases identify injected target locations.

This metric is separate from operator imitation.

## 8. Deferred operator fields

Do not implement these fields in the first milestone.

1. Medicine value.
2. Medicine form.
3. Real clock time.
4. The three-month sales benchmark.
5. External software vendor identity.
6. A persistent real-world blacklist.

The current model does not provide all required fields.

Add these fields only after the first demo works.

For time rules, first define `ticks_per_day` and the start time.

For medicine rules, first add product value and product form.

## 9. Learner features

Use named features that match observable case fields.

The first feature vector must contain these values.

1. A bias value.
2. One-hot event type values.
3. One-hot alert code values.
4. One-hot source role values.
5. One-hot destination role values.
6. `same_location`.
7. `same_organization`.
8. `same_city`.
9. `same_postal_code`.
10. Normalized distance.
11. `market_mismatch`.
12. Normalized prior pack event count.
13. Normalized prior pack alert count.
14. Normalized recent source event count.
15. Normalized recent destination event count.
16. Normalized source trust score.
17. Normalized destination trust score.

Use a fixed feature order.

Write the feature names into each output report.

Do not use the operator rule identifier as a feature.

Do not use the final operator label as a feature.

## 10. Online model

Implement `OnlineLogisticLearner` with NumPy.

Do not add a machine learning dependency.

The learner predicts this value.

```text
p_unsafe = sigmoid(weight dot feature_vector)
```

Use binary cross-entropy loss.

Use one stochastic gradient update for each queried case.

Add L2 regularization.

Expose these values in the demo configuration.

1. `learning_rate`.
2. `l2_strength`.
3. `decision_threshold`.
4. `initial_bias`.

Use these first defaults.

```json
{
  "learning_rate": 0.05,
  "l2_strength": 0.0001,
  "decision_threshold": 0.5,
  "initial_bias": -1.0
}
```

Clip the sigmoid input to a safe numeric range.

Save the model weights after each run.

## 11. Query policy

Implement two query policies.

### 11.1 Query all

Query the operator for every case.

Use this policy for the first correctness test.

### 11.2 Uncertainty query

Query cases near a probability of `0.5`.

Also query a small random sample.

This random sample prevents permanent blind spots.

Use these configuration values.

1. `query_budget_per_tick`.
2. `uncertainty_margin`.
3. `exploration_rate`.

Rank uncertain cases by the smallest distance from `0.5`.

Apply the per-tick budget after this ranking.

## 12. Execution loop

Implement replay mode first.

Replay mode uses a completed event log.

Live mode uses `run_ticks_with_hook()`.

Both modes must call the same case builder and learner code.

Use this loop.

```python
for tick in ordered_ticks:
    events = events_for_tick(tick)
    cases = case_builder.build(events, history)

    for case in cases:
        features = feature_encoder.encode(case)
        probability = learner.predict_proba(features)
        prediction = probability >= decision_threshold
        query = query_policy.select(case, probability)

        if query:
            feedback = operator.label(case)
            learner.update(features, feedback.label)

        recorder.append_step(...)

    history.update(events)
```

Update history after the current tick.

This order prevents same-tick future leakage.

## 13. Replay experiment

Create a deterministic replay experiment first.

Use the Bulgaria scenario with cached geocodes.

Run these perturbation types.

1. `COUNTERFEIT`.
2. `SMUGGLING`.
3. `THEFT`.
4. `VOLUME_SPIKE`.

Use at least five seeds.

Use one baseline and one perturbed run for each seed.

Build one combined case stream for training.

Build a separate fixed case stream for evaluation.

Do not update the learner on the evaluation stream.

## 14. CLI contract

Create `scripts/run_hitl_demo.py`.

The script must support this command.

```bash
uv run python scripts/run_hitl_demo.py \
  --scenario bulgaria \
  --ticks 300 \
  --seeds 1,2,3,4,5 \
  --perturbations COUNTERFEIT,SMUGGLING,THEFT,VOLUME_SPIKE \
  --operator configs/hitl/operator_rules.v1.json \
  --query-policy all \
  --output artifacts/hitl_demo
```

Add these modes.

```text
--mode oracle
--mode human
```

The first implementation must support `oracle` mode.

The final demo must support `human` mode.

Human mode must show one case at a time.

Use these commands in human mode.

```text
u = unsafe
s = safe
k = skip
q = quit and save
```

The CLI must support resume from a checkpoint.

## 15. Output contract

Write all outputs under the selected output directory.

Create these files.

```text
config_resolved.json
cases.jsonl
feedback.jsonl
learning_steps.csv
metrics.json
model_weights.json
feature_names.json
checkpoint.json
```

Create these plots.

```text
learning_curve.png
false_positive_rate.png
query_efficiency.png
model_weights.png
alert_timeline.png
simulation_map.png
```

The output files must include the repository commit hash.

The output files must include all random seeds.

## 16. Metrics

Record metrics after each fixed label interval.

Use an interval of 25 labels.

Record these metrics.

1. Accuracy.
2. Precision.
3. Recall.
4. F1 score.
5. False positive rate.
6. Log loss.
7. Labels requested.
8. Cases processed.
9. Query rate.

Report metrics on the fixed evaluation stream.

Also report metrics for each perturbation type.

Report target-location precision and recall against perturbation labels.

Keep this metric separate from operator imitation metrics.

Do not report only one final score.

## 17. Visualizations

### 17.1 Learning curve

Plot evaluation F1 against the number of operator labels.

Also plot the false positive rate.

This plot shows whether the learner approaches the operator.

### 17.2 Query efficiency

Plot evaluation F1 against the number of requested labels.

Compare query-all and uncertainty-query modes.

### 17.3 Model weights

Plot the largest positive and negative feature weights.

This plot shows the learned policy in readable terms.

### 17.4 Simulation map

Reuse the current geographic graph plot.

Highlight unsafe cases and learner predictions.

Use different markers for true positives, false positives, and false negatives.

### 17.5 Alert timeline

Reuse the current event timeline structure.

Add one row for operator labels and one row for learner decisions.

## 18. Acceptance criteria

The first demo passes only when all criteria hold.

1. The same configuration produces the same output files.
2. The case stream contains safe and unsafe cases.
3. The learner never reads the evaluation labels before prediction.
4. The final evaluation F1 is at least `0.85` in query-all mode.
5. The final false positive rate is at most `0.10`.
6. The F1 score improves by at least `0.20` from the initial model.
7. At least four of five seeds meet the F1 target.
8. Uncertainty-query mode uses at most half of the labels.
9. Uncertainty-query F1 stays within `0.05` of query-all F1.
10. The CLI creates all required output files.
11. The quick integration test finishes without manual input.
12. The demo does not change the C++ ABI.

If the data set cannot meet the class-count requirement, fail the run.

Do not hide this failure with resampling.

## 19. Required tests

Create these unit tests.

1. Test same-location feature extraction.
2. Test same-postal-code feature extraction.
3. Test distance feature extraction.
4. Test hospital-to-hospital operator labeling.
5. Test unsafe rule priority over safe rules.
6. Test deterministic safe-event sampling.
7. Test learner update direction.
8. Test checkpoint save and load.
9. Test that evaluation labels do not update the learner.
10. Test deterministic output from fixed seeds.

Create these integration tests.

1. Run a small synthetic scenario in replay mode.
2. Run one counterfeit perturbation in oracle mode.
3. Create all required report files.
4. Compare replay mode and live mode case identifiers.

Mark the Bulgaria ensemble test as slow.

## 20. File plan

Add these files.

```text
python/learning/__init__.py
python/learning/models.py
python/learning/case_builder.py
python/learning/features.py
python/learning/operator.py
python/learning/online_model.py
python/learning/query_policy.py
python/learning/runner.py
python/learning/metrics.py
python/learning/visualization.py
scripts/run_hitl_demo.py
configs/hitl/operator_rules.v1.json
configs/hitl/demo.v1.json
tests/test_learning_case_builder.py
tests/test_learning_operator.py
tests/test_learning_model.py
tests/test_learning_runner.py
tests/test_hitl_demo.py
```

Keep learning code outside `python/analytics/`.

The learner changes state during a run.

Analytics code should remain focused on reports and scoring.

## 21. Implementation order

### Work package 1: Data contract

1. Add the learning dataclasses.
2. Add the operator JSON schema.
3. Add the feature name contract.
4. Add unit tests for serialization.

Done means all new models round-trip through JSON.

### Work package 2: Case builder

1. Build observable alert cases from event records.
2. Add canonical location lookups.
3. Add safe-event sampling.
4. Add history counters.
5. Add unit tests for each feature.

Done means one event log produces a deterministic case stream.

### Work package 3: Hidden operator

1. Load ordered rules from JSON.
2. Apply rule priority.
3. Return labels and rule identifiers.
4. Add tests for each rule.

Done means the operator labels a fixed fixture deterministically.

### Work package 4: Online learner

1. Add the fixed feature encoder.
2. Add online logistic regression.
3. Add model save and load.
4. Add learner unit tests.

Done means repeated feedback reduces loss on a fixed fixture.

### Work package 5: Replay runner

1. Generate event logs from current experiments.
2. Build training and evaluation streams.
3. Run the query-all loop.
4. Save step records and metrics.
5. Add the quick integration test.

Done means the CLI produces a complete report directory.

### Work package 6: Learning plots

1. Add the learning curve.
2. Add the false positive plot.
3. Add the model weight plot.
4. Add the map overlay.
5. Add the alert timeline overlay.

Done means each plot builds from saved report files.

### Work package 7: Active label selection

1. Add uncertainty ranking.
2. Add random exploration.
3. Add the per-tick query budget.
4. Compare query efficiency against query-all mode.

Done means active mode meets the label-efficiency criterion.

### Work package 8: Live and human modes

1. Connect the runner to `run_ticks_with_hook()`.
2. Add the terminal label prompt.
3. Add checkpoint resume.
4. Compare live and replay case identifiers.

Done means a person can label cases and resume later.

## 22. Iteration rules

Use one work package per implementation iteration.

Before each iteration, read this specification and the current tests.

Do not change the system boundary without a written specification change.

Do not add hidden access from the learner to the operator.

Do not add a new dependency before NumPy becomes insufficient.

Run the focused tests after each code change.

Run the full test suite before each pull request.

Record any acceptance criterion that does not pass.

Do not lower a target without a written reason.

## 23. Full reinforcement learning extension

Add this extension only after the first demo passes.

The learner must choose an action that changes later simulation data.

Possible actions include these examples.

1. Allow the transaction.
2. Hold the transaction.
3. Request an audit.
4. Add the actor to a watchlist.
5. Change the future scan rate for the actor.

The simulator must model the action result.

The reward must include fraud loss, operator cost, and false alert cost.

At that point, the problem becomes a contextual bandit or reinforcement learning problem.

Do not call the first demo reinforcement learning in project claims.
