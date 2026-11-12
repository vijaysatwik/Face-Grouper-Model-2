"""Synthetic recognition tests; vectors contain no personal data."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import recognize_face  # noqa: E402
from recognize_face import CalibrationThresholdError, load_latest_threshold, recognize_embedding  # noqa: E402


def unit(score):
    vector = np.zeros(512, dtype=np.float32)
    vector[0] = score
    vector[1] = np.sqrt(1 - score * score)
    return vector


class RecognitionTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.stack([unit(.95), unit(.80)])
        self.items = [
            {"identity": "kowshik", "relative_path": "dataset/processed/kowshik/a.jpg", "filename": "a.jpg"},
            {"identity": "murthy", "relative_path": "dataset/processed/murthy/b.jpg", "filename": "b.jpg"},
        ]

    def test_accepts_candidate_at_inclusive_threshold(self):
        # Matcher aggregates the top two reference scores here: (1.00 + 0.80) / 2.
        result = recognize_embedding(unit(1), self.matrix, self.items, .90)
        self.assertEqual(result["label"], "kowshik")
        self.assertEqual(result["status"], "known")

    def test_rejects_low_score_as_unknown(self):
        result = recognize_embedding(unit(0), self.matrix, self.items, .90)
        self.assertEqual(result["label"], "Unknown Person")
        self.assertIsNone(result["identity"])

    def test_no_threshold_file_does_not_fall_back_to_default(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.object(recognize_face, "THRESHOLD_DIR", Path(temporary_directory)):
                with self.assertRaises(CalibrationThresholdError):
                    load_latest_threshold()

    def test_bad_embedding_or_threshold_rejected(self):
        with self.assertRaises(ValueError):
            recognize_embedding(np.zeros(512), self.matrix, self.items, .5)
        with self.assertRaises(ValueError):
            recognize_embedding(unit(1), self.matrix, self.items, float("inf"))


if __name__ == "__main__":
    unittest.main()
