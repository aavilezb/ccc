# Function/input classifiers

The three model families are independent modules:

```text
models/
├── common/             # trace parsing, split assignment, evaluation only
├── inception_time/
│   ├── preprocess.py
│   ├── model.py
│   └── train.py
├── transformer/
│   ├── preprocess.py
│   ├── model.py
│   └── train.py
└── minirocket/
    ├── preprocess.py
    ├── model.py
    └── train.py
```

Every model owns its preprocessing artifact. The shared loader performs only
schema parsing, correctness filtering, bundling, labeling, and split assignment.
The learned models never receive sampling period, runtime, golden count, target
sample count, PID, CPU, or other metadata that could shortcut classification.

The class label is:

```text
<collection name>::<executable name>::<input identifier>
```

Thus `baseline::json_scan::1000` and `patched::json_scan::1000` are separate
classes. Use the same collection name when extending an existing implementation
dataset with additional trace indices.

## Splitting

The default `--split-unit trace` keeps every repetition from a trace JSON file
in one split. This is preferable because sequentially collected repetitions can
share acquisition conditions. It requires at least three trace files for every
class. For 40 observations per class, a collection such as `--runs 10 --total 4`
provides four acquisition blocks and supports this split.

If all 40 repetitions were written to one trace file, preprocessing cannot make
a trace-held-out train/validation/test split. `--split-unit run` is available in
that case; it still keeps each complete 13-group bundle intact, but gives a less
strict estimate of generalization across acquisition sessions.

All preprocessors calculate the physically meaningful interval signal:

```text
delta(member event) / delta(retired-instruction leader)
```

This is unit normalization rather than hand-designed feature extraction. Means,
variances, peaks, frequency coefficients, and similar summary features are not
constructed.

Each training output directory contains model state, `metrics.json`,
`predictions.csv`, and `confusion_matrix.png`. PyTorch models additionally save
loss history; MiniROCKET saves its fitted transform and linear classifier. A
`run.json` records the complete preprocessing metadata and training arguments.
All classifiers use training-split-derived class weights so a partially missing
class does not silently dominate the loss.

## Run all three models

From the `function_classification` directory, the aggregate runner applies one
deterministic split to all models. Its default holds out exactly five trace
files per class for testing and five other files for validation:

```sh
./run_all_models.sh --input-root outputs/baseline --name baseline_test5
```

The summary is written to `models/benchmarks/baseline_test5/summary.csv`, while
each model retains its preprocessed data and results inside its own directory.

## Restrict the input events

Pass an adaptive-format event CSV with `--event-subset` to train and test using
only its member events. The event indices in the subset must match those stored
in the collected traces. Filtering preserves each original PMU group's sampling
timeline; counters from independently sampled groups are never merged.

The canonical deterministic event list is
`../../events/pmc_group_events_deterministic.csv`. From the
`experiments/function_classification` directory, run all models on that subset
with:

```sh
./run_all_models.sh \
  --input-root outputs/baseline \
  --name baseline_deterministic_test5 \
  --event-subset ../../events/pmc_group_events_deterministic.csv \
  --device cuda
```

To train on selected workload arguments when a collection root contains
multiple inputs, pass their stored input IDs explicitly. For example, the
single-invocation patch stress test uses:

```sh
./run_all_models.sh \
  --input-root ../cross_machine_collection/results/patch-training/baseline \
  --input-ids 1 \
  --name patch_iteration1_full_test5 \
  --device cuda
```

The selected IDs are recorded in preprocessing metadata. The external runner
accepts the same `--input-ids` option.

The preprocessed dataset's `run.json` records both the subset path and selected
member indices, so full-event and restricted-event results remain auditable.

## Frozen inference on another machine

Use the external runner after bringing back a cross-machine trace tree:

```sh
./run_external_inference.sh \
  --input-root /mnt/linuxstorage/testing_data_cross-os \
  --name latitude_ubuntu \
  --device cuda
```

It evaluates the full-event and deterministic-event versions of all three
models. External preprocessing uses the sequence length, event layout, class
order, normalization mean, and normalization scale saved with the training
dataset. Neural checkpoints are loaded without optimization. MiniROCKET reuses
its training-fitted kernels, dilations, quantile biases, feature scaler, and
logistic classifier. No external observation is used to fit or select state.

Each model writes its prepared external tensor under `external_data/` and its
metrics, predictions, and confusion matrix under `external_results/`. The
combined summary is stored under `models/benchmarks/external/<name>/summary.csv`.
