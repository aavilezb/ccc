# Patch-stress workloads

The `*_v0` and `*_v1` executables in each family are compiled from the same
implementation header. They exercise the same deterministic valid inputs and
produce the same logical result; only the measured function implementation is
changed.

| Family | v0 | v1 | Expected difficulty |
|---|---|---|---|
| `bounds_check` | direct use of a parser-supplied index | adds one bounds guard | very subtle |
| `overflow_check` | validates `offset + length` after addition | validates with `length <= size - offset` | subtle |
| `auth_compare` | exits on the first unequal byte | accumulates all byte differences | moderate |

The generated authentication mismatch is in the final byte. This intentionally
makes the early-exit and constant-time paths much closer than a first-byte
mismatch would be.

The class identity remains `<namespace>::<executable>::<input>`, for example:

```text
baseline::bounds_check_v0::100000
baseline::bounds_check_v1::100000
```

This allows confusion matrices to show exactly which patch levels the model
can or cannot distinguish.

The intended stress configuration is `--lengths 1`: every model example is one
invocation of the measured function. Repeated `--runs` and `--total` values
create independent observations; they do not increase the work performed
inside an individual invocation.
