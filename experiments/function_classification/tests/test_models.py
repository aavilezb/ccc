import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.common.trace_dataset import (  # noqa: E402
    SPLIT_IDS,
    assign_splits,
    fixed_length_tensor,
    load_examples,
    padded_tensor,
    standardize_fixed,
    standardize_padded,
)
from models.inception_time.model import GroupInceptionClassifier  # noqa: E402
from models.minirocket.model import MiniRocketTransform  # noqa: E402
from models.transformer.model import GroupTransformerClassifier  # noqa: E402


def measurement(class_index, batch, run):
    members = [1, 2, 3] if batch == 1 else [4, 5, 6]
    cumulative = {0: 0, **{member: 0 for member in members}}
    samples = []
    for interval in range(1, 5):
        cumulative[0] += 100
        for member in members:
            cumulative[member] += class_index * 7 + batch * 3 + member + interval
        samples.append(
            {
                "counts": {str(index): value for index, value in cumulative.items()},
                "time": interval,
            }
        )
    return {
        "label": "work",
        "run_mode": "sampling",
        "leader_total_count": 400,
        "time_enabled": 1000,
        "time_running": 1000,
        "lost_samples": 0,
        "num_samples": 4,
        "collection": {
            "file_index": 0,
            "batch": batch,
            "run": run,
            "member_indices": members,
        },
        "adaptive_validation": {
            "counter_coverage": 1.0,
            "record_retention": 1.0,
        },
        "samples": samples,
    }


def write_fixture(root):
    for class_index in range(3):
        for file_index in range(1, 4):
            path = root / f"impl{class_index}" / "prog" / "100" / "traces" / f"trace_{file_index}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            measurements = [
                measurement(class_index, batch, run)
                for batch in (1, 2)
                for run in (1, 2)
            ]
            document = {
                "schema_version": 2,
                "configuration": {
                    "class_name": f"impl{class_index}",
                    "target_executable": "/tmp/prog",
                    "input_id": "100",
                    "leader": {"index": 0},
                },
                "collection": {"file_index": file_index},
                "measurements": measurements,
            }
            path.write_text(json.dumps(document), encoding="utf-8")


class ModelPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        write_fixture(cls.root)
        cls.result = load_examples(cls.root)
        cls.examples = cls.result.examples
        cls.splits = assign_splits(cls.examples, 0.2, 0.2, 17, "trace")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_loader_bundles_groups_and_splits_by_trace(self):
        self.assertEqual(len(self.examples), 18)
        self.assertEqual(len(self.examples[0].groups), 2)
        for split_id in SPLIT_IDS.values():
            self.assertEqual(np.count_nonzero(self.splits == split_id), 6)
        by_file = {}
        for example, split in zip(self.examples, self.splits):
            by_file.setdefault(example.source_file, set()).add(int(split))
        self.assertTrue(all(len(values) == 1 for values in by_file.values()))

    def test_exact_split_unit_counts(self):
        splits = assign_splits(
            self.examples, 0.2, 0.2, 17, "trace", validation_units=1, test_units=1
        )
        for split_id in SPLIT_IDS.values():
            self.assertEqual(np.count_nonzero(splits == split_id), 6)

    def test_loader_selects_members_without_merging_group_timelines(self):
        selected = load_examples(self.root, member_indices=(2, 5))
        self.assertEqual(len(selected.examples), 18)
        self.assertEqual(selected.examples[0].group_members, ((2,), (5,)))
        self.assertEqual(selected.examples[0].groups[0].shape, (4, 1))
        self.assertEqual(selected.examples[0].groups[1].shape, (4, 1))

    def test_loader_filters_and_validates_input_ids(self):
        selected = load_examples(self.root, input_ids=("100",))
        self.assertEqual(len(selected.examples), 18)
        self.assertEqual(selected.trace_files, 9)
        self.assertTrue(all(example.input_id == "100" for example in selected.examples))
        with self.assertRaisesRegex(ValueError, "no complete valid examples"):
            load_examples(self.root, input_ids=("missing",))

    def test_loader_filters_and_validates_programs(self):
        selected = load_examples(self.root, programs=("prog",))
        self.assertEqual(len(selected.examples), 18)
        self.assertEqual(selected.trace_files, 9)
        self.assertTrue(all(example.program == "prog" for example in selected.examples))
        with self.assertRaisesRegex(ValueError, "no complete valid examples"):
            load_examples(self.root, programs=("missing",))

    def test_fixed_and_padded_preprocessing(self):
        fixed, _, members = fixed_length_tensor(self.examples, 16)
        fixed, _, _ = standardize_fixed(fixed, self.splits, members)
        self.assertEqual(fixed.shape, (18, 2, 16, 3))
        self.assertTrue(np.isfinite(fixed).all())
        padded, mask, _, members = padded_tensor(self.examples, 8)
        padded, _, _ = standardize_padded(padded, mask, self.splits, members)
        self.assertEqual(padded.shape, (18, 2, 8, 3))
        self.assertEqual(mask.sum(axis=2).min(), 4)

    def test_neural_models_accept_grouped_tensors(self):
        x = torch.randn(4, 2, 16, 3)
        mask = torch.ones(4, 2, 16, dtype=torch.bool)
        inception = GroupInceptionClassifier(2, 3, 3, filters=4, depth=1)
        transformer = GroupTransformerClassifier(
            2, 3, 3, model_dimension=16, heads=4, layers=1, feedforward_dimension=32
        )
        self.assertEqual(tuple(inception(x).shape), (4, 3))
        self.assertEqual(tuple(transformer(x, mask).shape), (4, 3))

    def test_minirocket_transform(self):
        fixed, _, members = fixed_length_tensor(self.examples, 16)
        fixed, _, _ = standardize_fixed(fixed, self.splits, members)
        transform = MiniRocketTransform(num_features=32, seed=9, batch_size=4)
        train = fixed[self.splits == SPLIT_IDS["train"]]
        transform.fit(train, members)
        features = transform.transform(fixed, members)
        self.assertEqual(features.shape, (18, 32))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(((features >= 0) & (features <= 1)).all())


if __name__ == "__main__":
    unittest.main()
