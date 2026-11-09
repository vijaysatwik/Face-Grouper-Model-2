"""Unit tests for cosine matching using synthetic vectors only."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from match_faces import cosine_similarity, match_query_embedding  # noqa: E402


def vector_with_first_coordinate(score: float) -> np.ndarray:
    """Return a unit vector with the requested dot product against [1, 0, 0]."""
    vector = np.zeros(512, dtype=np.float64)
    vector[0] = score
    vector[1] = np.sqrt(max(0.0, 1.0 - score * score))
    return vector


def query_vector() -> np.ndarray:
    vector = np.zeros(512, dtype=np.float64)
    vector[0] = 1.0
    return vector


def reference(identity: str, name: str, vector: np.ndarray) -> dict:
    return {
        "identity": identity,
        "filename": name,
        "relative_path": f"dataset/processed/{identity}/{name}",
    }


class CosineSimilarityTests(unittest.TestCase):
    def test_identical_unit_vectors_score_one(self):
        vector = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(vector, vector), 1.0, places=7)

    def test_orthogonal_vectors_score_zero(self):
        self.assertAlmostEqual(
            cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 0.0, places=7
        )

    def test_opposite_vectors_score_negative_one(self):
        self.assertAlmostEqual(
            cosine_similarity(np.array([1.0, 0.0]), np.array([-1.0, 0.0])), -1.0, places=7
        )

    def test_non_unit_vectors_are_handled(self):
        self.assertAlmostEqual(
            cosine_similarity(np.array([2.0, 0.0]), np.array([3.0, 4.0])), 0.6, places=7
        )

    def test_zero_nan_infinite_and_mismatched_vectors_are_rejected(self):
        bad_vectors = [
            np.zeros(3),
            np.array([1.0, np.nan]),
            np.array([1.0, np.inf]),
        ]
        for bad in bad_vectors:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    cosine_similarity(bad, np.ones(bad.size))
        with self.assertRaises(ValueError):
            cosine_similarity(np.ones(2), np.ones(3))


class IdentityRankingTests(unittest.TestCase):
    def test_top_three_mean_ranks_identity_not_single_best_photo(self):
        query = query_vector()
        vectors = np.stack(
            [
                vector_with_first_coordinate(0.90),
                vector_with_first_coordinate(0.80),
                vector_with_first_coordinate(0.70),
                vector_with_first_coordinate(0.10),
                vector_with_first_coordinate(0.85),
                vector_with_first_coordinate(0.84),
                vector_with_first_coordinate(0.82),
            ]
        )
        items = [
            reference("kowshik", f"k{index}.jpg", vectors[index])
            for index in range(4)
        ] + [
            reference("murthy", f"m{index}.jpg", vectors[index + 4])
            for index in range(3)
        ]
        result = match_query_embedding(query, vectors, items, top_k=3)
        self.assertEqual(result["best_candidate"], "murthy")
        self.assertGreater(
            result["ranked_candidates"][0]["aggregate_similarity"],
            result["ranked_candidates"][1]["aggregate_similarity"],
        )
        self.assertEqual(result["ranked_candidates"][0]["references_used"], 3)

    def test_top_k_handles_identity_with_fewer_references(self):
        query = query_vector()
        vectors = np.stack([query])
        items = [reference("kowshik", "only.jpg", vectors[0])]
        result = match_query_embedding(query, vectors, items, top_k=3)
        candidate = result["ranked_candidates"][0]
        self.assertEqual(candidate["references_used"], 1)
        self.assertAlmostEqual(candidate["aggregate_similarity"], 1.0)

    def test_duplicate_reference_vectors_are_not_counted_twice(self):
        query = query_vector()
        repeated = vector_with_first_coordinate(0.9)
        vectors = np.stack([repeated, repeated.copy(), vector_with_first_coordinate(0.6)])
        items = [
            reference("kowshik", "first.jpg", vectors[0]),
            reference("kowshik", "duplicate.jpg", vectors[1]),
            reference("kowshik", "second.jpg", vectors[2]),
        ]
        result = match_query_embedding(query, vectors, items, top_k=3)
        candidate = result["ranked_candidates"][0]
        self.assertEqual(result["deduplicated_reference_count"], 1)
        self.assertEqual(len(candidate["individual_scores"]), 2)
        self.assertEqual(candidate["references_used"], 2)

    def test_query_source_is_excluded_from_reference_gallery(self):
        query = query_vector()
        vectors = np.stack([query, vector_with_first_coordinate(0.5)])
        items = [
            reference("kowshik", "query.jpg", vectors[0]),
            reference("kowshik", "other.jpg", vectors[1]),
        ]
        result = match_query_embedding(
            query,
            vectors,
            items,
            exclude_relative_path="dataset\\processed\\kowshik\\query.jpg",
        )
        scores = result["ranked_candidates"][0]["individual_scores"]
        self.assertEqual([score["filename"] for score in scores], ["other.jpg"])

    def test_invalid_query_dimension_is_rejected(self):
        vectors = np.zeros((0, 512), dtype=np.float32)
        with self.assertRaises(ValueError):
            match_query_embedding(np.ones(8), vectors, [])


if __name__ == "__main__":
    unittest.main()
