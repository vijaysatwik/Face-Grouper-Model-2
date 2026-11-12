"""Calibrate Unknown Person rejection from separate local photo partitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
REPORT_DIR = PROJECT_ROOT / "reports"
THRESHOLD_DIR = PROJECT_ROOT / "artifacts" / "threshold"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IDENTITIES = ("kowshik", "murthy", "surekha", "vijay")
MODEL_NAME = "InceptionResnetV1/VGGFace2"
CALIBRATION_TOP_K = 1


def summarize_at_threshold(records: Iterable[dict], threshold: float) -> dict:
    """Compute empirical FAR/FRR for score records using inclusive >= acceptance."""
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("threshold must be finite and between -1 and 1")
    rows = list(records)
    unknown = [row for row in rows if row.get("actual_identity") is None]
    known = [row for row in rows if row.get("actual_identity") is not None]
    for row in rows:
        score = float(row["score"])
        if not math.isfinite(score) or not -1 <= score <= 1:
            raise ValueError("all scores must be finite and between -1 and 1")
    accepted_unknown = sum(float(row["score"]) >= threshold for row in unknown)
    rejected_known = sum(float(row["score"]) < threshold for row in known)
    accepted_known = [row for row in known if float(row["score"]) >= threshold]
    correct_known = sum(row["predicted_identity"] == row["actual_identity"] for row in accepted_known)
    return {
        "threshold": float(threshold),
        "unknown_count": len(unknown),
        "known_count": len(known),
        "far": accepted_unknown / len(unknown) if unknown else None,
        "frr": rejected_known / len(known) if known else None,
        "known_correct_accept_rate": correct_known / len(known) if known else None,
        "known_wrong_accept_count": len(accepted_known) - correct_known,
        "accepted_unknown_count": accepted_unknown,
        "rejected_known_count": rejected_known,
    }


def select_threshold(records: Iterable[dict], target_far: float) -> tuple[float, dict]:
    """Sweep observed score boundaries; maximize correct known acceptance under target FAR."""
    if not math.isfinite(target_far) or not 0 <= target_far <= 1:
        raise ValueError("target FAR must be finite and between 0 and 1")
    rows = list(records)
    if not any(row.get("actual_identity") is None for row in rows):
        raise ValueError("unknown calibration examples are required")
    if not any(row.get("actual_identity") is not None for row in rows):
        raise ValueError("known calibration examples are required")
    scores = sorted({float(row["score"]) for row in rows})
    if any(not math.isfinite(value) or not -1 <= value <= 1 for value in scores):
        raise ValueError("all scores must be finite and between -1 and 1")
    candidates = {-1.0, 1.0, *scores}
    # The next representable value above each score allows the >= classifier to
    # reject even the highest observed unknown sample when the target FAR is 0.
    candidates.update(float(np.nextafter(value, math.inf)) for value in scores if value < 1.0)
    results = [summarize_at_threshold(rows, t) for t in sorted(candidates) if t <= 1.0]
    feasible = [result for result in results if result["far"] is not None and result["far"] <= target_far]
    if not feasible:
        raise ValueError("no threshold met the requested empirical FAR")
    chosen = max(feasible, key=lambda result: (
        result["known_correct_accept_rate"],
        -result["known_wrong_accept_count"],
        result["threshold"],
    ))
    return chosen["threshold"], {"target_far": target_far, "selected": chosen, "sweep": results}


def _direct_images(folder: Path) -> list[Path]:
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                  key=lambda p: p.name.casefold()) if folder.is_dir() else []


def _subject_folders(root: Path) -> dict[str, list[Path]]:
    if not root.is_dir():
        return {}
    subjects = {}
    for folder in sorted(root.iterdir(), key=lambda p: p.name.casefold()):
        if folder.is_dir():
            photos = _direct_images(folder)
            if photos:
                subjects[folder.name] = photos
    return subjects


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_query_embeddings(paths: list[Path]) -> dict[Path, np.ndarray]:
    """Detect exactly one face per local image and embed it with cached Day 6 weights."""
    try:
        import cv2
        import torch
    except ImportError as error:
        raise RuntimeError(f"Required package {error.name!r} is missing from {sys.executable}") from error
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from extract_embeddings import get_weights_path, load_model, make_tensor
    from prepare_dataset import make_square_crop, OUTPUT_SIZE
    from query_face_detection import detect_one_query_face

    weights = get_weights_path(torch)
    if not weights.is_file():
        raise RuntimeError(f"Local FaceNet weights are missing: {weights}; no download was attempted")
    cascades = Path(cv2.data.haarcascades)
    frontal = cv2.CascadeClassifier(str(cascades / "haarcascade_frontalface_alt2.xml"))
    profile = cv2.CascadeClassifier(str(cascades / "haarcascade_profileface.xml"))
    if frontal.empty() or profile.empty():
        raise RuntimeError("Could not load OpenCV's bundled face detectors")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(torch, weights, device)
    vectors = {}
    skipped = {}
    for path in paths:
        try:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("image is unreadable or unsupported")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face, _method, reason = detect_one_query_face(gray, frontal, profile)
            if face is None:
                raise ValueError(reason)
            crop = cv2.resize(make_square_crop(image, face), OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
            tensor = make_tensor(torch, crop).to(device)
            with torch.no_grad():
                vector = model(tensor).squeeze(0).detach().cpu().numpy().astype(np.float32)
            norm = float(np.linalg.norm(vector))
            if vector.shape != (512,) or not np.isfinite(vector).all() or not math.isfinite(norm) or norm <= 0:
                raise ValueError("FaceNet returned an invalid embedding")
            vectors[path.resolve()] = vector / norm
            print(f"Embedded local photo: {path.relative_to(PROJECT_ROOT)}")
        except (cv2.error, OSError, ValueError, RuntimeError) as error:
            skipped[path.resolve()] = str(error)
            print(f"SKIP local photo: {path.relative_to(PROJECT_ROOT)} ({error})")
    return vectors, skipped


def _load_day7_data():
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from match_faces import discover_latest_bundle, load_embedding_bundle
    npz, metadata_path = discover_latest_bundle()
    matrix, metadata = load_embedding_bundle(npz, metadata_path)
    split_path = REPORT_DIR / "evaluation_split_proposal.csv"
    with split_path.open(encoding="utf-8-sig", newline="") as stream:
        split_rows = list(csv.DictReader(stream))
    if not split_rows:
        raise ValueError("Day 7 split proposal is missing or empty")
    by_path = {str((PROJECT_ROOT / row["relative_path"]).resolve()).casefold():
               (row["embedding_index"], row) for row in metadata}
    enrollment_paths: set[str] = set()
    reserves: dict[str, list[dict]] = defaultdict(list)
    role_groups = defaultdict(set)
    for row in split_rows:
        role = row["proposed_role"].strip()
        if role not in {"enrollment", "known_test", "unused_reserve"}:
            raise ValueError(f"Unexpected Day 7 split role: {role}")
        role_groups[row["near_duplicate_group"]].add(role)
        key = str(Path(row["processed_local_path"]).resolve()).casefold()
        if key not in by_path:
            raise ValueError(f"Day 7 crop is not in the current embedding bundle: {row['processed_local_path']}")
        item_index, item = by_path[key]
        if item["identity"].casefold() != row["identity"].casefold():
            raise ValueError("Day 7 split identity does not match the embedding index")
        if role == "enrollment":
            enrollment_paths.add(item["relative_path"])
        elif role == "unused_reserve":
            reserves[row["identity"].casefold()].append({
                "identity": row["identity"].casefold(),
                "group": row["near_duplicate_group"],
                "source": Path(row["original_local_path"]),
                "vector": matrix[item_index],
                "relative_path": item["relative_path"],
            })
    if any(len(roles) > 1 for roles in role_groups.values()):
        raise ValueError("Day 7 split has a near-duplicate group crossing enrollment, test, or reserve roles")
    return matrix, metadata, enrollment_paths, reserves, npz


def _unique_paths(*groups: list[Path]) -> None:
    roles: dict[str, Path] = {}
    for group in groups:
        for path in group:
            key = _sha256(path)
            previous = roles.get(key)
            if previous is not None:
                raise ValueError(f"Exact duplicate photo appears in multiple examples: {previous.name} and {path.name}")
            roles[key] = path


def _score_queries(queries: list[dict], matrix: np.ndarray, metadata: list[dict],
                   enrollment_paths: set[str], top_k: int = CALIBRATION_TOP_K) -> list[dict]:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from match_faces import match_query_embedding
    scores = []
    for query in queries:
        result = match_query_embedding(query["vector"], matrix, metadata,
                                       include_paths=enrollment_paths, top_k=top_k)
        candidate = result["ranked_candidates"][0]
        scores.append({
            "actual_identity": query["actual_identity"],
            "predicted_identity": candidate["identity"],
            "score": candidate["aggregate_similarity"],
        })
    return scores


def _save_json_unique(folder: Path, prefix: str, value: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = folder / f"{prefix}_{stamp}.json"
    suffix = 1
    while path.exists():
        path = folder / f"{prefix}_{stamp}_{suffix}.json"
        suffix += 1
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    return path


def run_calibration(target_far: float) -> tuple[dict, dict]:
    matrix, metadata, enrollment_paths, reserves, bundle_path = _load_day7_data()
    calibration_unknown = _subject_folders(DATASET_DIR / "unknown_calibration")
    final_unknown = _subject_folders(DATASET_DIR / "unknown_test")
    if not calibration_unknown or not final_unknown or any(not photos for photos in calibration_unknown.values()) or any(not photos for photos in final_unknown.values()):
        raise ValueError("Put photos in at least one subject folder under both unknown_calibration and unknown_test")
    if set(name.casefold() for name in calibration_unknown) & set(name.casefold() for name in final_unknown):
        raise ValueError("Unknown calibration and final-test folder labels must identify different people")
    for name, photos in {**{f"calibration/{k}": v for k, v in calibration_unknown.items()},
                         **{f"final/{k}": v for k, v in final_unknown.items()}}.items():
        if len(photos) < 5:
            print(f"WARNING: {name} has {len(photos)} photo(s); five or more are recommended.")

    new_known_cal = {name: _direct_images(DATASET_DIR / "known_calibration" / name) for name in IDENTITIES}
    new_known_test = {name: _direct_images(DATASET_DIR / "known_test" / name) for name in IDENTITIES}
    known_cal: list[dict] = []
    known_final: list[dict] = []
    reserve_sources: list[Path] = []
    # Divide at most two unused Day 7 reserves per identity into independent
    # calibration and final-test examples. Day 7 known_test rows are never reused.
    for identity in IDENTITIES:
        available = sorted(reserves.get(identity, []), key=lambda row: row["group"].casefold())
        if available:
            known_cal.append({"actual_identity": identity, "vector": available[0]["vector"], "path": available[0]["source"]})
            reserve_sources.append(available[0]["source"])
        if len(available) > 1:
            known_final.append({"actual_identity": identity, "vector": available[1]["vector"], "path": available[1]["source"]})
            reserve_sources.append(available[1]["source"])
        if new_known_cal[identity]:
            known_cal.extend({"actual_identity": identity, "path": path, "new": True} for path in new_known_cal[identity])
        if new_known_test[identity]:
            known_final.extend({"actual_identity": identity, "path": path, "new": True} for path in new_known_test[identity])
    for label, rows in (("known calibration", known_cal), ("known final test", known_final)):
        missing = [identity for identity in IDENTITIES if not any(row["actual_identity"] == identity for row in rows)]
        if missing:
            raise ValueError(f"{label} needs at least one independent example for each identity; missing: {', '.join(missing)}")
    unknown_cal_paths = [path for photos in calibration_unknown.values() for path in photos]
    unknown_final_paths = [path for photos in final_unknown.values() for path in photos]
    new_known_paths = [row["path"] for row in known_cal + known_final if row.get("new")]
    _unique_paths(unknown_cal_paths, unknown_final_paths, new_known_paths, reserve_sources)
    all_new_paths = unknown_cal_paths + unknown_final_paths + new_known_paths
    new_vectors, skipped_paths = _load_query_embeddings(all_new_paths) if all_new_paths else ({}, {})
    if skipped_paths:
        print(f"Skipped {len(skipped_paths)} unreadable, no-face, or ambiguous photo(s); see each SKIP reason above.")
    for row in known_cal + known_final:
        if row.get("new"):
            row["vector"] = new_vectors.get(row["path"].resolve())
    unknown_cal_queries = [
        {"actual_identity": None, "vector": new_vectors[path.resolve()]}
        for path in unknown_cal_paths
        if path.resolve() in new_vectors
    ]
    unknown_final_queries = [
        {"actual_identity": None, "vector": new_vectors[path.resolve()]}
        for path in unknown_final_paths
        if path.resolve() in new_vectors
    ]
    for label, folders in (("calibration", calibration_unknown), ("final test", final_unknown)):
        for subject, photos in folders.items():
            usable = sum(path.resolve() in new_vectors for path in photos)
            print(f"{label} unknown/{subject}: {usable}/{len(photos)} usable photo(s)")
            if usable == 0:
                raise ValueError(f"No usable photos remain for {label} unknown subject {subject}")
    for label, rows in (("known calibration", known_cal), ("known final test", known_final)):
        missing = [identity for identity in IDENTITIES if not any(
            row["actual_identity"] == identity and row.get("vector") is not None for row in rows
        )]
        if missing:
            raise ValueError(f"{label} needs a usable independent example for each identity; missing: {', '.join(missing)}")

    cal_scores = _score_queries(known_cal + unknown_cal_queries, matrix, metadata,
                                enrollment_paths, top_k=CALIBRATION_TOP_K)
    threshold, sweep = select_threshold(cal_scores, target_far)
    final_scores = _score_queries(known_final + unknown_final_queries, matrix, metadata,
                                  enrollment_paths, top_k=CALIBRATION_TOP_K)
    final_metrics = summarize_at_threshold(final_scores, threshold)
    final_metrics["known_correct_identity_count"] = sum(
        row["actual_identity"] is not None
        and row["predicted_identity"] == row["actual_identity"]
        and row["score"] >= threshold
        for row in final_scores
    )
    final_metrics["known_wrong_identity_accepted_count"] = sum(
        row["actual_identity"] is not None
        and row["predicted_identity"] != row["actual_identity"]
        and row["score"] >= threshold
        for row in final_scores
    )
    cal_subject_count = len(calibration_unknown)
    test_subject_count = len(final_unknown)
    warnings = []
    prior_final_reports = list(REPORT_DIR.glob("day8_final_evaluation_*.json"))
    evaluation_status = "independent_first_evaluation" if not prior_final_reports else "diagnostic_after_prior_final_results"
    if prior_final_reports:
        warnings.append("A Day 8 final-test report already existed before this run. Treat these repeated final-set metrics as diagnostic only, not as a fresh independent evaluation.")
    if cal_subject_count < 5 or test_subject_count < 5:
        warnings.append("Only one unknown subject is represented in at least one partition; FAR is a small-sample demonstration, not a reliable population estimate.")
    if len(unknown_cal_queries) < 20 or len(unknown_final_queries) < 20:
        warnings.append("Fewer than 20 unknown query photos in a partition; reported FAR changes in large steps and has wide uncertainty.")
    audit_path = REPORT_DIR / "dataset_audit.txt"
    audit_text = audit_path.read_text(encoding="utf-8", errors="replace") if audit_path.is_file() else ""
    audit_status = next((line.partition(":")[2].strip() for line in audit_text.splitlines()
                         if line.startswith("Readiness result:")), "AUDIT_NOT_FOUND")
    stamp = datetime.now(timezone.utc).isoformat()
    threshold_meta = {
        "threshold": threshold,
        "target_far": target_far,
        "model": MODEL_NAME,
        "top_k": CALIBRATION_TOP_K,
        "aggregation": f"mean of each identity's top {CALIBRATION_TOP_K} enrollment cosine similarity",
        "created_utc": stamp,
        "embedding_bundle": bundle_path.name,
        "day5_audit_status": audit_status,
        "calibration_known_count": len(known_cal),
        "calibration_unknown_photos": len(unknown_cal_paths),
        "calibration_unknown_usable_photos": len(unknown_cal_queries),
        "calibration_unknown_subjects": cal_subject_count,
        "calibration_far": sweep["selected"]["far"],
        "calibration_frr": sweep["selected"]["frr"],
        "selection_rule": "best known correct-accept rate among thresholds meeting the explicit empirical target FAR",
        "limitations": warnings,
        "privacy": "Local biometric metadata. Do not commit, publish, or share.",
    }
    final_report = {
        "created_utc": stamp,
        "threshold": threshold,
        "target_far": target_far,
        "top_k": CALIBRATION_TOP_K,
        "model": MODEL_NAME,
        "day5_audit_status": audit_status,
        "final_known_count": len(known_final),
        "final_unknown_photos": len(unknown_final_paths),
        "final_unknown_usable_photos": len(unknown_final_queries),
        "final_unknown_subjects": test_subject_count,
        "evaluation_status": evaluation_status,
        "metrics": final_metrics,
        "warnings": warnings,
        "privacy": "Local biometric evaluation report. Do not commit, publish, or share.",
    }
    # Persist only after calibration and the independent final partition both ran.
    threshold_path = _save_json_unique(THRESHOLD_DIR, "threshold", threshold_meta)
    report_path = _save_json_unique(REPORT_DIR, "day8_final_evaluation", final_report)
    return threshold_meta | {"threshold_file": str(threshold_path.relative_to(PROJECT_ROOT))}, final_report | {
        "report_file": str(report_path.relative_to(PROJECT_ROOT))
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-far", required=True, type=float,
                        help="maximum empirical unknown acceptance fraction (for five calibration photos, 0.20 allows at most one)")
    args = parser.parse_args()
    if not math.isfinite(args.target_far) or not 0 <= args.target_far <= 1:
        parser.error("--target-far must be a finite number from 0 through 1")
    audit_path = REPORT_DIR / "dataset_audit.txt"
    audit_status = "AUDIT_NOT_FOUND"
    if audit_path.is_file():
        audit_status = next((line.partition(":")[2].strip() for line in
                             audit_path.read_text(encoding="utf-8", errors="replace").splitlines()
                             if line.startswith("Readiness result:")), "AUDIT_STATUS_UNAVAILABLE")
    print(f"Day 5 audit status: {audit_status}; it is reported but not changed by Day 8 calibration.")
    try:
        calibration, final = run_calibration(args.target_far)
    except (OSError, ValueError, RuntimeError, KeyError, csv.Error) as error:
        print(f"STOP: {error}")
        return 2
    print("\nThreshold calibrated from calibration data only.")
    print(f"Threshold: {calibration['threshold']:.6f}; target FAR: {calibration['target_far']:.3f}")
    print(f"Calibration: known={calibration['calibration_known_count']}, unknown={calibration['calibration_unknown_photos']} photos across {calibration['calibration_unknown_subjects']} subject(s), empirical FAR={calibration['calibration_far']:.3f}, FRR={calibration['calibration_frr']:.3f}")
    status_label = "Independent final-test results:" if final["evaluation_status"] == "independent_first_evaluation" else "Diagnostic final-test results (this set was evaluated previously):"
    print(f"\n{status_label}")
    print(f"Known queries: {final['final_known_count']}; correct accepted identities: {final['metrics']['known_correct_identity_count']}; rejected known: {final['metrics']['rejected_known_count']}; wrong known accepted: {final['metrics']['known_wrong_identity_accepted_count']}")
    print(f"Unknown queries: {final['final_unknown_photos']} across {final['final_unknown_subjects']} subject(s); false accepts: {final['metrics']['accepted_unknown_count']}; FAR={final['metrics']['far']:.3f}")
    print(f"Final known FRR: {final['metrics']['frr']:.3f}")
    for warning in calibration["limitations"]:
        print(f"WARNING: {warning}")
    print(f"Local threshold metadata: {calibration['threshold_file']}")
    print(f"Local final-test report: {final['report_file']}")
    print("All image processing was local. No photo was changed, copied, or uploaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
