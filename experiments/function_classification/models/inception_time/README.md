# InceptionTime-style classifier

This module performs end-to-end learned temporal feature extraction. Its
preprocessor converts cumulative sample reads to interval event/instruction
ratios, linearly maps each group to a fixed normalized execution timeline, and
applies training-split-only z-score normalization. It does not calculate
hand-designed statistical features.

Each PMU group is processed by a separate compact InceptionTime-style encoder.
The resulting group embeddings are concatenated only after temporal encoding,
because the groups came from independent executions.

```sh
python3 models/inception_time/preprocess.py \
  --input-root outputs/baseline \
  --output models/inception_time/data/baseline.npz \
  --length 128

python3 models/inception_time/train.py \
  --data models/inception_time/data/baseline.npz \
  --output-dir models/inception_time/results/baseline
```
