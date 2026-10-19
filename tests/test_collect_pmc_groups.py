import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import collect_pmc_groups as collector  # noqa: E402


class AdaptivePeriodTests(unittest.TestCase):
    def test_period_floor_does_not_round_expected_samples_below_target(self):
        self.assertEqual(collector.choose_period(7_551, 20, 1), (20, 377))
        self.assertEqual(collector.choose_period(9_156, 20, 1), (20, 457))

    def test_period_floor_reports_when_hard_minimum_limits_sample_count(self):
        self.assertEqual(collector.choose_period(9_156, 20, 1_000), (9, 1_017))

    def test_repetitions_make_desired_samples_feasible_at_period_floor(self):
        self.assertEqual(
            collector.choose_measurement_repetitions(100, 20, 1_000, 10_000),
            (200, 20_000),
        )
        self.assertEqual(
            collector.choose_measurement_repetitions(25_000, 20, 1_000, 10_000),
            (1, 20_000),
        )

    def test_repetition_limit_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "max-repetitions"):
            collector.choose_measurement_repetitions(100, 20, 1_000, 199)

    def test_reduced_period_has_guard_margin(self):
        self.assertEqual(
            collector.reduce_period_for_sample_floor(100, 19, 20, 1), 90
        )
        self.assertIsNone(
            collector.reduce_period_for_sample_floor(1, 19, 20, 1)
        )

    def test_calibration_retries_every_group_at_reduced_period(self):
        calls = []

        def fake_run_target(
            _command,
            _working_directory,
            _events,
            member_indices,
            _mode,
            _raw_output,
            _timeout,
            _measurement_label,
            sample_period,
            _ring_buffer_pages,
            context_mixer,
            cpu_affinity,
            measurement_repetitions,
        ):
            calls.append(
                (
                    tuple(member_indices),
                    sample_period,
                    context_mixer,
                    cpu_affinity,
                    measurement_repetitions,
                )
            )
            samples = 19 if sample_period == 100 else 20
            return {
                "num_samples": samples,
                "leader_total_count": 2_000,
                "lost_samples": 0,
                "time_enabled": 100,
                "time_running": 100,
            }, 0.1

        with tempfile.TemporaryDirectory() as temporary, patch.object(
            collector, "run_target", side_effect=fake_run_target
        ):
            period, attempts, wall_seconds = collector.calibrate_sample_period(
                ["target"],
                Path(temporary),
                Path("events.csv"),
                [[1, 2, 3], [4, 5, 6]],
                Path(temporary) / "measurement.json",
                30.0,
                None,
                100,
                1,
                20,
                2,
                3,
                8192,
                "mixed_v1",
                7,
                230,
            )

        self.assertEqual(period, 90)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0]["minimum_observed_samples"], 19)
        self.assertEqual(attempts[1]["minimum_observed_samples"], 20)
        self.assertEqual(len(calls), 8)
        self.assertTrue(all(call[2:] == ("mixed_v1", 7, 230) for call in calls))
        self.assertAlmostEqual(wall_seconds, 0.8)


if __name__ == "__main__":
    unittest.main()
