# libpmc m2 milestone

This repository is a source snapshot of `libpmc_dev` after the second
measurement and classification milestone. It preserves the complete Git
history inherited from `libpmc_m1` and adds the implementation used for the
dynamic-repetition and context-mixer experiment.

## Measurement milestone

- Ten workload programs support `PMC_MEASUREMENT_REPETITIONS` and repeat only
  the target function inside the `pmc_group_begin_csv()` / `pmc_group_end()`
  measurement region.
- The adaptive collector first measures one invocation, selects the desired
  sample count, and increases the repetition count when necessary:

  ```text
  repetitions = ceil(minimum_period * desired_samples / one_call_golden_count)
  ```

- A repeated scout run verifies the effective golden count before sampling.
- Period calibration enforces the retained-sample floor for every PMU group.
- CPU affinity and the optional `mixed_v1` context mixer are applied to the
  measured target execution.
- `tools/find_least_busy_cpu.py` provides an SMT-aware CPU-selection aid.

The milestone campaign used a minimum period of 1000 and a fixed target of 20
samples for input length 1. These are explicit campaign settings rather than
changes to the collector's general defaults.

## Classification milestone

MiniRocket was trained on the Ubuntu 22 host and evaluated without target
retraining on the Latitude Ubuntu 18 machine. One measured run is one model
input trace; a JSON trace file is only a container for ten runs.

Using deterministic events across ten function classes:

| Condition | Host test accuracy | Latitude accuracy | Latitude macro-F1 |
| --- | ---: | ---: | ---: |
| No context mixer | 100.0% | 95.8% | 95.72% |
| `mixed_v1` | 100.0% | 96.8% | 96.78% |

Training used 3,000 traces, host validation and test used 500 traces each, and
Latitude inference used 500 traces. Invalid measurements were not included.

Generated trace datasets, tensors, trained models, and plots remain excluded
by `.gitignore`; the source snapshot does not duplicate the multi-gigabyte
artifacts stored under `libpmc_dev`.

## Review order

1. `pmc.h`
2. `pmc.c`
3. `programs/measurement_repetition.h` and `programs/`
4. `tools/collect_pmc_groups.py`
5. `context/`
6. `tools/find_least_busy_cpu.py`
7. `collect.sh` and `tools/collect_all.sh`
8. `automation/`
9. `experiments/function_classification/models/`

Build and run the source-level regression tests with:

```sh
make rebuilt
make test
```
