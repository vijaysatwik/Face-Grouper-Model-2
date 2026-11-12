"""Synthetic unit tests for threshold calibration math; no photos used."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from calibrate_threshold import select_threshold, summarize_at_threshold  # noqa: E402


class ThresholdTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"actual_identity": "a", "predicted_identity": "a", "score": .91},
            {"actual_identity": "b", "predicted_identity": "a", "score": .61},
            {"actual_identity": None, "predicted_identity": "a", "score": .72},
            {"actual_identity": None, "predicted_identity": "b", "score": .30},
        ]

    def test_summary_counts_far_frr_and_identity(self):
        result = summarize_at_threshold(self.rows, .7)
        self.assertEqual(result["far"], .5)
        self.assertEqual(result["frr"], .5)
        self.assertEqual(result["known_wrong_accept_count"], 0)

    def test_selector_respects_target_far(self):
        threshold, report = select_threshold(self.rows, 0)
        self.assertGreaterEqual(threshold, .72)
        self.assertEqual(report["selected"]["far"], 0)

    def test_empty_population_has_no_division_error(self):
        result = summarize_at_threshold([], .5)
        self.assertIsNone(result["far"])
        self.assertIsNone(result["frr"])

    def test_unknowns_and_knowns_are_required(self):
        with self.assertRaisesRegex(ValueError, "unknown calibration"):
            select_threshold([self.rows[0]], .05)
        with self.assertRaisesRegex(ValueError, "known calibration"):
            select_threshold([self.rows[2]], .05)

    def test_invalid_threshold_and_target_rejected(self):
        with self.assertRaises(ValueError):
            summarize_at_threshold(self.rows, float("nan"))
        with self.assertRaises(ValueError):
            select_threshold(self.rows, 1.1)


if __name__ == "__main__":
    unittest.main()
