"""Behavioral checks for the paired experiment, not timing-dependent tests."""
import importlib.util
import math
from pathlib import Path
import statistics
import unittest

spec = importlib.util.spec_from_file_location("comparison", Path(__file__).resolve().parents[1] / "tools/compare_benchmarks.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def rows(ratio, blocks=8, noise=None):
    result = []
    for block in range(blocks):
        order = "ABBA" if block % 2 else "BAAB"
        for position, letter in enumerate(order):
            # Slowdown drifts linearly within a block and varies strongly across blocks.
            baseline = (1 + block * .15) * (1 + position * .02)
            value = baseline * (ratio if letter == "B" else 1)
            if noise: value *= noise[block] if letter == "B" else 1
            result.append(dict(block=block, order=order, variant="baseline" if letter == "A" else "candidate",
                               mean_ms=value, cpu_ms=value, p99_ms=value, max_ms=value,
                               over_10ms=0, late=0, involuntary_switches=0, major_faults=0))
    return result


class StatisticsTest(unittest.TestCase):
    def test_student_against_analytic_cauchy(self):
        for value in (.1, 1, 4, 20):
            self.assertAlmostEqual(comparison.student_t_cdf(value, 1), .5 + math.atan(value) / math.pi, places=10)

    def test_student_against_nist_table(self):
        for df, expected in ((1, 12.706), (7, 2.365), (11, 2.201), (23, 2.069)):
            self.assertAlmostEqual(comparison.student_t_quantile(.975, df), expected, delta=.00051)

    def test_abba_cancels_linear_drift(self):
        result = comparison.summarize(rows(.95), resamples=2000)
        self.assertAlmostEqual(result["time_reduction_pct"], 5, places=10)
        self.assertGreater(result["time_reduction_ci_pct"][0], 4.99)

    def test_no_change_not_declared_gain(self):
        result = comparison.summarize(rows(1), resamples=2000)
        self.assertAlmostEqual(result["time_reduction_pct"], 0)
        self.assertEqual(result["two_sided_sign_randomization_p"], 1)

    def test_noisy_comparison_preserves_uncertainty(self):
        result = comparison.summarize(rows(.99, noise=[.9, 1.1] * 4), resamples=4000)
        self.assertLess(result["time_reduction_ci_pct"][0], 0)
        self.assertGreater(result["time_reduction_ci_pct"][1], 0)

    def test_incomplete_blocks_not_counted(self):
        result = comparison.summarize(rows(.95)[:-1], resamples=1000)
        self.assertEqual(result["complete_blocks"], 7)

    def test_sample_sd_of_run_means(self):
        result = comparison.distribution([1, 2, 3])
        self.assertEqual(result["sd"], 1)

    def test_regression_has_negative_effect(self):
        result = comparison.summarize(rows(1.1), resamples=2000)
        self.assertAlmostEqual(result["time_reduction_pct"], -10)
        self.assertLess(result["time_reduction_ci_pct"][1], 0)


if __name__ == "__main__": unittest.main()
