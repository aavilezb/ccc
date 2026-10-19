# libpmc development tree

The reproducible Ubuntu 22 training to Ubuntu 18.04 inference workflow,
including portable-runtime troubleshooting and host-GPU instructions, is in
[`automation/README.md`](automation/README.md).

```bash
sudo sysctl -w kernel.nmi_watchdog=0
sudo sysctl -w kernel.perf_event_paranoid=0
```
## Find a quiet CPU

`tools/find_least_busy_cpu.py` observes Linux per-CPU accounting and selects a
best-effort least-busy CPU from the caller's allowed affinity mask. It includes
SMT-sibling activity in the score by default:

```bash
./tools/find_least_busy_cpu.py --interval 1
```

Use its CPU-only output with another program:

```bash
CPU=$(./tools/find_least_busy_cpu.py --interval 1 --cpu-only)
taskset -c "$CPU" ./rebuilt/bin/example_group 100
```

The selected CPU is not reserved; unrelated tasks and interrupts can begin
using it after the observation interval.

## Fixed context-mixer experiment

The collector can pin every measured target to CPU 7 and run the optional
`mixed_v1` preconditioner outside the counted region:

```bash
./tools/collect_all.sh \
  --bin-dir "$PWD/rebuilt/bin" \
  --events "$PWD/events/pmc_group_events_full.csv" \
  --output-root "$PWD/results/context-mixer/mixed_v1" \
  --name baseline \
  --programs example_group \
  --lengths 1 \
  --runs 1 \
  --total 1 \
  --cpu 7 \
  --context-mixer mixed_v1
```

Use an otherwise identical collection under a different output root with
`--context-mixer none` as the CPU-pinned control. Keep `--name baseline` in both
datasets so the model class labels remain identical. See `context/README.md`
for the measurement boundary and the exact `mixed_v1` operations.
sudo sysctl -w kernel.nmi_watchdog=0
sudo sysctl -w kernel.perf_event_paranoid=0earning_output" \
    --name learning_demo \
    --programs example_group \
    --lengths 100 \
    --member-indices 1,2,3 \
    --runs 1 \
    --total 1 \
    --start 1 \
    --min-samples 20 \
    --calibration-runs 1 \
    --max-calibration-attempts 3 \
    --ring-buffer-pages 128 \
    --overwrite

  # This should require approximately three target executions:

  # 1 scout run
  # 1 calibration run, assuming the first period succeeds
  # 1 actual sampling run

  # Generated files:

  # learning_output/
  # └── learning_demo/
  #     ├── example_group/
  #     │   └── 100/
  #     │       ├── traces/
  #     │       │   └── learning_demo_1.json
  #     │       └── manifests/
  #     │           └── collection_1_1.json
  #     └── campaigns/
  #         └── <timestamp>/
  #             ├── combo_times.csv
  #             └── logs/
  #                 └── example_group__100.log
```
Terminal Output:

```bash
Programs:     1
Lengths:      100
Combinations: 1
Runs/combo:   1 trace file(s) x 1 run(s) per event group

[1/1] example_group + input 100
Class name:     learning_demo
Target:         /home/lizeren/Desktop/libpmc/libpmc_dev/rebuilt/bin/example_group 100
Input ID:       100
Event groups:   1 (3 member events)
Output:         /home/lizeren/Desktop/libpmc/libpmc_dev/learning_output/learning_demo/example_group/100
Pilot:          counting retired instructions ...
Pilot result:   golden=978, desired=20, target=20, initial period=48, wall=0.06s
Calibration:    attempt 1/3, period=48, 1 run(s) x 1 group(s)
Calibration:    accepted period=48; minimum retained=21
Trace 1/1: learning_demo_1.json
  group 1/1 run 1/1: 21 samples, 0.01s; ETA 0.00s

Collected 1/1 sampling runs in 0.09s
Target execution time: 0.01s
Quality: coverage 104.09-104.09%, retention 99.02-99.02%, lost=0
Traces: /home/lizeren/Desktop/libpmc/libpmc_dev/learning_output/learning_demo/example_group/100/traces
Manifest: /home/lizeren/Desktop/libpmc/libpmc_dev/learning_output/learning_demo/example_group/100/manifests/collection_1_1.json
Combination time: 0.123474s (success)

Campaign finished in 0.140s
Timing CSV: /home/lizeren/Desktop/libpmc/libpmc_dev/learning_output/learning_demo/campaigns/20260923T195729Z/combo_times.csv
Logs:       /home/lizeren/Desktop/libpmc/libpmc_dev/learning_output/learning_demo/campaigns/20260923T195729Z/logs
```

 **to mimic scouting phase and observe the output:**

```bash
mkdir -p code_study/scout

env \
    -u PMC_SAMPLE_PERIOD \
    -u PMC_RING_BUFFER_PAGES \
    PMC_EVENTS_CSV="$PWD/events/pmc_group_events_full.csv" \
    PMC_MEMBER_INDICES="1,2,3" \
    PMC_RUN_MODE=scout \
    PMC_OUTPUT_FILE="$PWD/code_study/scout/measurement.json" \
    ./rebuilt/bin/example_group 100

```

**to mimic calibration phase and observe the output:**

```bash
mkdir -p code_study/calibration

  env \
    -u PMC_SAMPLE_PERIOD \
    -u PMC_RING_BUFFER_PAGES \
    PMC_EVENTS_CSV="$PWD/events/pmc_group_events_full.csv" \
    PMC_MEMBER_INDICES="1,2,3" \
    PMC_RUN_MODE=sampling \
    PMC_SAMPLE_PERIOD=100 \
    PMC_RING_BUFFER_PAGES=128 \
    PMC_OUTPUT_FILE="$PWD/code_study/calibration/measurement.json" \
    ./rebuilt/bin/example_group 100

```

# Cross-machine PMU collection bundle

This directory is a self-contained Linux/x86-64 train/test collection bundle.
It builds native training workloads, freezes those exact ELF files and their
runtime, and executes the frozen payload unchanged on test machines.

Runtime requirements are Bash, coreutils, and Python 3.8 or newer. GCC, Make,
`readelf`, and `ldd` are required in train mode and are not required in test
mode.

## Shipping the trainer

The trainer is a source bundle intended for the machine that will establish a
new training domain. It contains the workloads, PMC library source, event
definitions, collection scripts, and build rules. It deliberately does not
contain `training_payload/`: train mode rebuilds that payload from the training
machine's own compiler and runtime.

After extracting the trainer into an empty directory, run:

```sh
./collect.sh --mode train --machine-id desktop-training --archive
```

## Mode 1: training collector

On the training machine, run:

```sh
./collect.sh --mode train --machine-id desktop-training --archive
```

Train mode always rebuilds from source with the local compiler and system
runtime. It then creates `training_payload/` containing:

- every newly built workload ELF;
- the matching `libpmc_group.so`;
- the ELF loader and every runtime library reported by the native linkage;
- a provenance manifest and SHA-256 checksums.

The collection defaults are 40 trace files x 10 runs = 400 examples per class.
There are ten workloads and seven input lengths, producing 70 classes and
28,000 examples when the complete default campaign is requested. The full
39-event configuration is used, allowing deterministic-event selection later.

With `--archive`, train mode produces three artifacts:

```text
exports/desktop-training.tar.gz                 # training results
exports/desktop-training_trainer.tar.gz         # reusable source/build collector
exports/desktop-training_test_collector.tar.gz  # collector + frozen training payload
```

The trainer archive can be sent to another machine to create a separate native
training dataset. Its train run will generate a new trainer archive and a new
test collector tied to that machine's freshly built binaries and runtime.

## Mode 2: test collector

Move the generated test-collector archive to a test machine and extract it into
an empty directory:

```sh
mkdir cross_machine_collection
tar -xzf desktop-training_test_collector.tar.gz -C cross_machine_collection
cd cross_machine_collection
./collect.sh --mode test --machine-id latitude-test --archive
```

Test mode never invokes GCC or Make. It verifies `training_payload/checksums.sha256`
and runs the frozen training ELFs through
`training_payload/runtime/ld-linux-x86-64.so.2`, with libraries resolved only
from the payload's `runtime/` directory.

Both modes default to:

- class namespace `baseline`;
- all ten programs and seven input lengths;
- 40 trace files per function/input class;
- 10 repetitions per trace file;
- all 39 member events.

Before writing any trace files, the collector chooses the leader period with
`floor(golden_count / target_samples)` and calibrates it across every selected
PMU event group. The default calibration performs three sampling runs per group.
If any run retains fewer than the default 20-sample floor, the period is lowered
and all groups are calibrated again. A production measurement below the floor
is rejected rather than silently stored. The hard period floor defaults to 1.
When the one-invocation golden count is too small to support the desired sample
count at that floor, the collector sets `PMC_MEASUREMENT_REPETITIONS` and runs a
second counting scout over repeated target-function calls. For example, a
golden count of 100 with 20 desired samples and `--min-period 1000` starts with
`ceil(20 * 1000 / 100) = 200` calls. The repeated scout supplies the actual
golden count used to choose the final period. `--max-repetitions` bounds this
automatic expansion (default: 1000000).

The bundled workload harnesses read `PMC_MEASUREMENT_REPETITIONS` immediately
before `pmc_group_begin_csv()` and repeat only the target function between the
existing begin/end calls. Custom targets must implement the same environment
variable contract to use automatic repetition.

Results are written under `results/ubuntu24-host2/baseline/`. Bring back the
whole `results/ubuntu24-host2/` directory; its `machine.json` records CPU, OS,
kernel, PMU limits, event-file hash, and binary hashes. The class namespace is
kept as `baseline` so returned ground-truth labels match the trained models;
the machine ID provides the separate domain/provenance boundary.

Use `--archive` to create `exports/<machine-id>.tar.gz`. Because the full
JSON dataset can be several gigabytes, archiving may take time and disk space.

## Short validation run

Before a full campaign, either check PMU access only:

```sh
./collect.sh --mode test --machine-id ubuntu24-host2 --preflight-only
```

or collect one small trace:

```sh
./collect.sh \
  --mode test \
  --machine-id ubuntu24-host2-smoke \
  --programs example_group \
  --lengths 10000 \
  --runs 1 \
  --total 1
```

The command-line values after the bundle options are passed to the campaign
runner. Run `./collect.sh --help` for the common controls.

### Older target glibc

An error such as `GLIBC_2.34 not found` means the target distribution's loader
tried to start the ELF before PMU setup. Test mode avoids this by explicitly
invoking the loader and libraries captured by train mode under
`training_payload/runtime/`.

If replacing the runtime manually, replace the loader and libraries together.
Merely setting `LD_LIBRARY_PATH` while retaining an older system loader can mix
incompatible glibc components.

## PMU compatibility

This implementation uses Linux `perf_event_open`, so native macOS and Windows
are unsupported. The raw event encodings originated on the reference Intel
Core i9-11900KF (family 6, model 167). Another Intel microarchitecture may
interpret or support some encodings differently; compare the target metadata
with `reference_machine.json` before treating features as semantically equal.

The wrapper rejects non-Intel CPUs by default. `--allow-pmu-mismatch` should
only be used with an event CSV whose encodings have been verified for that PMU.
The preflight opens every configured group. Insufficient perf permissions or
unsupported events therefore fail before the long campaign instead of silently
producing a partial dataset.

To collect only the legacy deterministic list, use `--events deterministic`.
For inference with both the full-event and deterministic-subset models, collect
the default full list once; the deterministic channels can be selected later.

## Intra-function patch stress suite

Six workloads form three version pairs built from shared implementation files.
Within a pair, the measured function has the same name, input, deterministic
data, and result. Only the security-relevant implementation changes:

- `bounds_check_v0` / `bounds_check_v1`: adds an explicit untrusted-index
  bounds check;
- `overflow_check_v0` / `overflow_check_v1`: replaces a potentially wrapping
  addition check with subtraction-based validation;
- `auth_compare_v0` / `auth_compare_v1`: replaces early-exit token comparison
  with a constant-time comparison.

The executable version is part of the class label, so each patch level remains
a distinct class. The focused stress campaign uses one invocation (`--lengths
1`) and therefore creates six deliberately similar classes:

```sh
./collect.sh \
  --mode train \
  --machine-id patch-training \
  --programs auth_compare_v0,auth_compare_v1,bounds_check_v0,bounds_check_v1,overflow_check_v0,overflow_check_v1 \
  --lengths 1 \
  --archive
```

Use the resulting `_test_collector.tar.gz` with the same `--programs` and
`--lengths` arguments for cross-machine inference collection.
