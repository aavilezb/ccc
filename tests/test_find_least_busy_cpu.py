from pathlib import Path
import importlib.util
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "find_least_busy_cpu.py"
SPEC = importlib.util.spec_from_file_location("find_least_busy_cpu", str(MODULE_PATH))
cpu_finder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cpu_finder)


class CpuListTests(unittest.TestCase):
    def test_parse_and_format_cpu_list(self):
        self.assertEqual(cpu_finder.parse_cpu_list("0-3,8,10-11"), {0, 1, 2, 3, 8, 10, 11})
        self.assertEqual(cpu_finder.format_cpu_list({0, 1, 2, 4, 7, 8}), "0-2,4,7-8")

    def test_rejects_descending_range(self):
        with self.assertRaisesRegex(ValueError, "descending"):
            cpu_finder.parse_cpu_list("4-2")


class ActivityTests(unittest.TestCase):
    def test_calculates_busy_fraction(self):
        before = {0: (100, 80), 1: (200, 120)}
        after = {0: (200, 170), 1: (300, 160)}
        activity = cpu_finder.calculate_activity(before, after)
        self.assertAlmostEqual(activity[0]["busy_fraction"], 0.10)
        self.assertAlmostEqual(activity[1]["busy_fraction"], 0.60)

    def test_smt_score_avoids_busy_sibling(self):
        activity = {
            0: {"total_delta": 100, "busy_delta": 10, "busy_fraction": 0.10},
            1: {"total_delta": 100, "busy_delta": 90, "busy_fraction": 0.90},
            2: {"total_delta": 100, "busy_delta": 30, "busy_fraction": 0.30},
            3: {"total_delta": 100, "busy_delta": 30, "busy_fraction": 0.30},
        }
        topology = {
            0: {"siblings": {0, 1}},
            1: {"siblings": {0, 1}},
            2: {"siblings": {2, 3}},
            3: {"siblings": {2, 3}},
        }
        smt_aware = cpu_finder.rank_cpus(activity, {0, 1, 2, 3}, topology)
        logical_only = cpu_finder.rank_cpus(
            activity, {0, 1, 2, 3}, topology, account_smt=False
        )
        self.assertEqual(smt_aware[0]["cpu"], 2)
        self.assertEqual(logical_only[0]["cpu"], 0)


if __name__ == "__main__":
    unittest.main()
