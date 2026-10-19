# Fixed microarchitectural preconditioner

`mixed_v1` is an optional, deterministic sequence used to test whether a
repeatable initial microarchitectural context improves classification accuracy.
It is a preconditioner, not an architectural reset and not a guarantee that two
different processors finish in identical cache, TLB, predictor, or frequency
states.

The sequence performs fixed amounts of:

- integer arithmetic;
- sequential and pseudo-random access to an initialized 8 MiB buffer;
- pseudo-random page access;
- indirect calls to distinct no-inline functions;
- data-dependent conditional branches.

Buffer allocation, initialization, and page prefaulting occur during PMU setup.
The final sequence contains no allocation or system calls. `pmc.c` runs it after
`PERF_EVENT_IOC_RESET` and before `PERF_EVENT_IOC_ENABLE`, while the whole perf
group remains disabled. Mixer instructions therefore do not belong to the
measured trace.

Enable it through the collector:

```sh
python3 tools/collect_pmc_groups.py \
  --target "$PWD/rebuilt/bin/example_group 10000" \
  --events "$PWD/events/pmc_group_events_full.csv" \
  --member-indices 1,2,3 \
  --name baseline \
  --cpu 7 \
  --context-mixer mixed_v1
```

Use `--context-mixer none` for the control condition. Both the mixer name and
CPU affinity are written into every raw measurement, trace configuration, and
collection manifest.

`mixed_v1` is versioned behavior. Do not silently change its operations or
constants after collecting data; introduce `mixed_v2` for a different sequence.
