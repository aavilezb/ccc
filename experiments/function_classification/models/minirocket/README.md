# MiniROCKET classifier

This module provides a self-contained channel-wise MiniROCKET-style transform,
so it does not require `sktime` or `aeon`. It uses the canonical 84 length-nine
kernels, dilation and padding variation, training-fitted quantile biases, and
proportion-of-positive-values features. Kernels operate on each PMU event
channel independently; a linear classifier combines the automatic features.

The preprocessor only creates normalized, fixed-length interval series. The
MiniROCKET biases and linear-feature scaling are fitted later using the training
split, preventing validation/test leakage.

```sh
python3 models/minirocket/preprocess.py \
  --input-root outputs/baseline \
  --output models/minirocket/data/baseline.npz \
  --length 128

python3 models/minirocket/train.py \
  --data models/minirocket/data/baseline.npz \
  --output-dir models/minirocket/results/baseline \
  --features 10000
```
