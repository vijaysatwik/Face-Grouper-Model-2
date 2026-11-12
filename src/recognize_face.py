"""Recognize one local face crop using a calibrated cosine threshold."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from calibrate_threshold import THRESHOLD_DIR  # noqa: E402
from match_faces import discover_latest_bundle, load_embedding_bundle, match_query_embedding  # noqa: E402


class CalibrationThresholdError(RuntimeError):
    """Raised when no valid calibration result is available."""


def validate_threshold(value: object) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("threshold must be a number") from error
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("threshold must be finite and between -1 and 1")
    return threshold


def recognize_embedding(query: np.ndarray, matrix: np.ndarray, items: list[dict], threshold: float,
                        include_paths: set[str] | None = None, top_k: int = 3) -> dict:
    """Return the top candidate only when its aggregate score clears threshold."""
    threshold = validate_threshold(threshold)
    result = match_query_embedding(query, matrix, items, include_paths=include_paths, top_k=top_k)
    best = result["ranked_candidates"][0]
    score = float(best["aggregate_similarity"])
    accepted = score >= threshold
    return {
        "status": "known" if accepted else "unknown",
        "identity": best["identity"] if accepted else None,
        "label": best["identity"] if accepted else "Unknown Person",
        "score": score,
        "threshold": threshold,
        "ranked_candidates": result["ranked_candidates"],
    }


def load_latest_threshold(path: Path | None = None) -> tuple[float, dict]:
    """Load latest local threshold metadata; never supply a default threshold."""
    if path is None:
        files = sorted(THRESHOLD_DIR.glob("threshold_*.json")) if THRESHOLD_DIR.is_dir() else []
        if not files:
            raise CalibrationThresholdError(
                "No calibrated threshold is available. Run dataset calibration first; no default is used."
            )
        path = files[-1]
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        threshold = validate_threshold(metadata["threshold"])
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as error:
        raise CalibrationThresholdError(f"Could not load a valid calibration file: {error}") from error
    if metadata.get("model") != "InceptionResnetV1/VGGFace2" or "target_far" not in metadata:
        raise CalibrationThresholdError("Calibration metadata is missing model or target FAR provenance")
    return threshold, metadata


def embed_local_image(image_path: Path) -> np.ndarray:
    """Detect exactly one face locally and embed its crop using cached Day 6 weights."""
    try:
        import cv2
        import torch
    except ImportError as error:
        raise RuntimeError(f"Required package {error.name!r} is not installed in this Python") from error
    from extract_embeddings import get_weights_path, load_model, make_tensor
    from prepare_dataset import make_square_crop, OUTPUT_SIZE
    from query_face_detection import detect_one_query_face

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image is unreadable: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_dir = Path(cv2.data.haarcascades)
    frontal = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_frontalface_alt2.xml"))
    profile = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_profileface.xml"))
    if frontal.empty() or profile.empty():
        raise RuntimeError("OpenCV bundled face detector could not be loaded")
    face, _method, reason = detect_one_query_face(gray, frontal, profile)
    if face is None:
        raise ValueError(f"Cannot process query: {reason}")
    crop = make_square_crop(image, face)
    crop = cv2.resize(crop, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = get_weights_path(torch)
    if not weights.is_file():
        raise RuntimeError(f"Local FaceNet weights are missing: {weights}; no download was attempted")
    model = load_model(torch, weights, device)
    tensor = make_tensor(torch, crop).to(device)
    with torch.no_grad():
        vector = model(tensor).squeeze(0).detach().cpu().numpy().astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (512,) or not np.isfinite(vector).all() or not math.isfinite(norm) or norm <= 0:
        raise ValueError("FaceNet returned an invalid embedding")
    return vector / norm


def load_enrollment_paths() -> set[str]:
    """Use only Day 7 enrollment references, matching the calibration gallery."""
    split_path = PROJECT_ROOT / "reports" / "evaluation_split_proposal.csv"
    try:
        with split_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error) as error:
        raise ValueError(f"Could not read Day 7 enrollment split: {error}") from error
    paths = {
        Path(row["processed_local_path"]).resolve().relative_to(PROJECT_ROOT).as_posix()
        for row in rows if row.get("proposed_role", "").strip() == "enrollment"
    }
    if not paths:
        raise ValueError("Day 7 split contains no enrollment references")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="local image containing exactly one face")
    args = parser.parse_args()
    try:
        threshold, metadata = load_latest_threshold()
        npz_path, json_path = discover_latest_bundle()
        matrix, items = load_embedding_bundle(npz_path, json_path)
        result = recognize_embedding(embed_local_image(args.image), matrix, items, threshold,
                                     include_paths=load_enrollment_paths(),
                                     top_k=int(metadata.get("top_k", 3)))
    except (CalibrationThresholdError, FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"STOP: {error}")
        return 2
    print(f"Result: {result['label']}")
    print(f"Best cosine score: {result['score']:.4f}; calibrated threshold: {threshold:.4f}")
    print(f"Calibration target FAR: {metadata['target_far']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
