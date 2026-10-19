# Transformer classifier

This module preserves the measured sequence length whenever it is at most
`--max-length`. Short sequences are right-padded with an explicit attention
mask; only longer sequences are linearly downsampled. Normalization statistics
come exclusively from the training split.

Each group has its own input projection, followed by a small shared temporal
transformer. By default, the learned projection maps four adjacent intervals to
one token, reducing attention cost without constructing hand-designed features.
Masked mean pooling produces one embedding per independently collected PMU
group, and classification happens after late fusion.

```sh
python3 models/transformer/preprocess.py \
  --input-root outputs/baseline \
  --output models/transformer/data/baseline.npz \
  --max-length 512

python3 models/transformer/train.py \
  --data models/transformer/data/baseline.npz \
  --output-dir models/transformer/results/baseline
```

The defaults deliberately use a small two-layer, four-head transformer because
the initial trace corpus is expected to be modest.
