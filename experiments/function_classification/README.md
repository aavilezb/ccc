# Function-classification experiment

This directory contains the reusable learning pipeline imported from:

```text
libpmc_new/experiments/function_classification/
```

The import intentionally includes source code, documentation, runners, and
tests only. Historical trace datasets, preprocessed `.npz` files, checkpoints,
plots, logs, and benchmark outputs remain in `libpmc_new` and are not duplicated
here.

## Boundary with trace collection

Trace collection is owned by the canonical tools at the repository root:

```text
tools/collect_pmc_groups.py
tools/collect_all.sh
collect.sh
```

The learning pipeline consumes their schema-version-2 trace JSON files. Model
code must not import collector internals; the trace schema is the interface
between acquisition and learning.

## Models

The imported model families are:

- MiniROCKET;
- InceptionTime-style convolutional network;
- Transformer.

Shared trace parsing, split assignment, evaluation, and training utilities are
under `models/common/`. Each model retains its own preprocessing module so its
input representation remains explicit and reproducible.

Install the Python dependencies in an isolated environment:

```sh
python3 -m pip install -r experiments/function_classification/models/requirements.txt
```

Run the model tests from the `libpmc_m1` root:

```sh
make test-models
```

`make test` remains the lightweight library/collector suite. Run both suites
with `make test-all`.

## Training

Point the aggregate runner at a collected class directory. For example:

```sh
experiments/function_classification/run_all_models.sh \
  --input-root /path/to/results/MACHINE_ID/baseline \
  --name experiment_name \
  --event-subset events/pmc_group_events_deterministic.csv \
  --device cuda
```

Generated tensors, checkpoints, metrics, plots, and benchmark summaries are
written below `models/` and ignored by Git.

## External inference

After training artifacts exist, evaluate an external trace tree with:

```sh
experiments/function_classification/run_external_inference.sh \
  --input-root /path/to/external/results/MACHINE_ID/baseline \
  --name target_domain \
  --device cuda
```

The external preprocessor reuses the class order, sequence layout, and
normalization parameters fitted on the training data. It must not fit state on
the external dataset.

## Historical reference

The historical result named
`latitude_ubuntu18_to_debian_30class_test5_deterministic` and its generated
artifacts remain under `libpmc_new`. This directory is the maintained source
copy for future experiments, not a relocation of historical results.
