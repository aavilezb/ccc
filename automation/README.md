# Ubuntu 22 training to Ubuntu 18 inference

This directory records the commands and portability fixes used for the
September 2026 context-mixer experiment:

- training host: Ubuntu 22, logical CPU 7;
- inference host: Latitude running Ubuntu 18.04, logical CPU 0;
- ten programs at input length 1;
- 40 training trace files and 10 runs per event group;
- 5 inference trace files and 10 runs per event group;
- control (`none`) and fixed context-mixer (`mixed_v1`) conditions;
- deterministic-event MiniRocket with 10,000 features.

No password is stored in these scripts. Use interactive SSH authentication or
an SSH agent.

## 1. Prepare PMU access

Record the existing values before changing them:

```bash
cat /proc/sys/kernel/nmi_watchdog
cat /proc/sys/kernel/perf_event_paranoid
cat /proc/sys/kernel/perf_event_mlock_kb
ulimit -l
getconf PAGESIZE
```

For the collection session, both hosts used:

```bash
sudo sysctl -w kernel.nmi_watchdog=0
sudo sysctl -w kernel.perf_event_paranoid=0
```

Restore the original values after the campaign if the machine should not keep
these settings. Do not put a sudo or SSH password in a script.

Select and then consistently use one CPU for each host:

```bash
./tools/find_least_busy_cpu.py --interval 10
```

The successful A/B run used CPU 7 locally and CPU 0 on Latitude. The finder is
a snapshot, not a reservation; avoid other work, especially on the selected
CPU and its SMT sibling, during collection.

## 2. Collect Ubuntu 22 training traces

Run from the `libpmc_dev` root:

```bash
./automation/collect_training_ubuntu22.sh \
  none desktop-affinity7-none-length1 7

./automation/collect_training_ubuntu22.sh \
  mixed_v1 desktop-affinity7-mixed-v1-length1 7
```

Each invocation rebuilds natively and stages `training_payload/`. The mixer is
compiled into the library but selected at runtime, so either resulting payload
supports both conditions.

Expected local result roots:

```text
results/desktop-affinity7-none-length1/baseline
results/desktop-affinity7-mixed-v1-length1/baseline
```

Package a lightweight test collector outside the source tree:

```bash
./automation/package_test_collector.sh /tmp/libpmc_test_collector.tar.gz
scp /tmp/libpmc_test_collector.tar.gz Latitude:~/Desktop/
```

## 3. Collect Ubuntu 18.04 inference traces

On Latitude, unpack into a new directory:

```bash
mkdir -p ~/Desktop/libpmc_context_inference
tar -xzf ~/Desktop/libpmc_test_collector.tar.gz \
  -C ~/Desktop/libpmc_context_inference
cd ~/Desktop/libpmc_context_inference
```

After applying the PMU sysctls, verify the frozen payload before a long run:

```bash
./collect.sh \
  --mode test \
  --machine-id latitude-preflight \
  --class-name baseline \
  --events full \
  --cpu 0 \
  --context-mixer none \
  --preflight-only
```

Then collect the two matched conditions:

```bash
./automation/collect_inference_ubuntu18.sh \
  none latitude-affinity0-none-length1 0

./automation/collect_inference_ubuntu18.sh \
  mixed_v1 latitude-affinity0-mixed-v1-length1 0
```

Archive only the result directories:

```bash
tar -czf /tmp/latitude-context-results.tar.gz \
  -C results \
  latitude-affinity0-none-length1 \
  latitude-affinity0-mixed-v1-length1
```

Back on Ubuntu 22:

```bash
scp Latitude:/tmp/latitude-context-results.tar.gz /tmp/
tar -xzf /tmp/latitude-context-results.tar.gz -C results
```

## 4. Train and infer with MiniRocket

The following command preprocesses both conditions, trains two independent
models, runs matched external inference, and writes metrics and confusion
matrices. Run it from `libpmc_dev`:

```bash
./automation/run_minirocket_ab.sh \
  results/desktop-affinity7-none-length1/baseline \
  results/desktop-affinity7-mixed-v1-length1/baseline \
  results/latitude-affinity0-none-length1/baseline \
  results/latitude-affinity0-mixed-v1-length1/baseline \
  context_mixer_replay \
  cuda
```

Outputs are placed under:

```text
experiments/function_classification/models/minirocket/data/<experiment-id>
experiments/function_classification/models/minirocket/results/<experiment-id>
experiments/function_classification/models/minirocket/external_results/<experiment-id>
```

### Requesting the host GPU from Codex

CUDA may be available on the host while intentionally hidden inside the Codex
sandbox. In that case, explicitly ask:

> Run `automation/run_minirocket_ab.sh ... cuda` outside the sandbox with
> escalated host execution because the GPU is not visible inside the sandbox.

Codex should present an execution-approval dialog. Approve the specific command
or the narrow MiniRocket train/infer command prefix. GPU access does not require
`sudo`. Verify host visibility with:

```bash
nvidia-smi
python3 -c 'import torch; ok=torch.cuda.is_available(); print(ok, torch.cuda.get_device_name(0) if ok else "no CUDA device")'
```

Do not conclude that the host lacks a GPU solely because these checks fail
inside the sandbox.

## Portability problems and their fixes

### Frozen glibc must use its DT_SONAME

The loader requests `libc.so.6`, even when the source file is physically named
`libc-2.27.so`. An old payload copied only the physical basename, causing the
training loader to fall back to the inference host's `/lib/.../libc.so.6` and
produce errors such as `GLIBC_2.35 not found`.

`tools/stage_training_payload.py` now reads each ELF `DT_SONAME` and copies the
same bytes under both required names. A valid payload contains at least:

```text
training_payload/runtime/ld-linux-x86-64.so.2
training_payload/runtime/libc.so.6
```

The portable launcher explicitly uses the frozen loader with:

```text
--inhibit-cache --library-path training_payload/runtime:training_payload
```

Useful checks are:

```bash
(cd training_payload && sha256sum --check checksums.sha256)
readelf -d training_payload/runtime/libc.so.6 | grep SONAME
```

### Ubuntu 18.04 `mawk` vendor parsing

The original colon-splitting trim left a leading space on Ubuntu 18.04 and
misreported `GenuineIntel` as a PMU mismatch. `collect.sh` now reads the third
whitespace-delimited field from `vendor_id`. Do not use
`--allow-pmu-mismatch` to hide this problem; raw encodings should only be used
on a compatible PMU.

### Python 3.8 runtime generic expressions

Postponed annotations are accepted by Python 3.8, but expressions such as
`enumerate[tuple[int, ...]](...)` and `tuple[Event, ...](...)` execute generic
syntax at runtime and fail there. The collector now uses ordinary
`enumerate(...)` and `tuple(...)`. The packaged collector requires Python 3.8
or newer.

### Shell line continuation

Every continued command line must end in `\`. Without it, the next option is
executed as a separate command, producing messages such as:

```text
bash: --allow-pmu-mismatch: command not found
```

The wrappers in this directory avoid manually repeating long commands.

### Interrupted and resumed collections

The collector refuses to overwrite existing trace indices. Resume at the next
index and adjust the number of new files, for example:

```bash
./automation/collect_inference_ubuntu18.sh \
  none latitude-affinity0-none-length1 0 \
  --start 3 --total 3
```

Use `--overwrite` only when the existing requested files are intentionally
being replaced.

### Length-1 quality warning

Forcing at least 20 samples from these extremely short functions selected
periods of roughly 1 to 4 retired instructions. The experiment had no lost
records and a running ratio of 1.0, but counter coverage was about 2.3x the
counting-only golden value and record retention was about 0.29. These traces
therefore fail the normal 95--105% coverage gate.

`run_minirocket_ab.sh` deliberately supplies `--include-invalid` so this
stress-test remains reproducible. Do not silently reuse that option for a
measurement-valid experiment. With longer functions or less aggressive
sampling, remove `--include-invalid` and enforce the normal quality checks.

## Recorded A/B result

For the dataset above, both local tests were 100%. Latitude inference improved
from 62.6% accuracy / 56.05% macro-F1 without the mixer to 70.8% accuracy /
65.27% macro-F1 with `mixed_v1`. This is one matched campaign and should be
repeated before treating the 8.2-point improvement as a stable estimate.
