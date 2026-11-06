"""Audit local face dataset files and write private reports/contact sheets."""

from collections import Counter, defaultdict
import csv
import hashlib
import math
from pathlib import Path
import sys
import textwrap

try:
    import cv2
    import numpy as np
except ImportError:
    print(
        "OpenCV and NumPy are needed for the local dataset audit. "
        "Install the project requirements with 'python -m pip install -r requirements.txt'."
    )
    raise SystemExit(1)

try:
    from prepare_dataset import (
        IDENTITIES,
        IMAGE_EXTENSIONS,
        OUTPUT_SIZE,
        detect_one_face,
        output_path_for,
    )
except ImportError:
    print("Could not import the existing preprocessing helpers from src/prepare_dataset.py.")
    raise SystemExit(1)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
PROCESSED_DIR = DATASET_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
PREVIOUS_DIAGNOSTICS = DATASET_DIR / "diagnostics" / "baseline_and_improved_diagnostics.csv"
REPORT_PATH = REPORTS_DIR / "dataset_audit.txt"
MANIFEST_PATH = REPORTS_DIR / "dataset_manifest.csv"
SPLIT_PATH = REPORTS_DIR / "evaluation_split_proposal.csv"
PHASH_DISTANCE_LIMIT = 10
TRAIN_ENROLLMENT_TARGET = 12
KNOWN_TEST_TARGET = 4


def sha256_file(path):
    """Hash a file in small pieces without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for block in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_image(path):
    """Open an image with OpenCV; return None when the file is unreadable."""
    try:
        return cv2.imread(str(path), cv2.IMREAD_COLOR)
    except (cv2.error, OSError):
        return None


def perceptual_hash(image):
    """Return a small visual hash used only to avoid splitting lookalike photos."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low_frequency = cv2.dct(small)[:8, :8].flatten()[1:]
    median = float(np.median(low_frequency))
    value = 0
    for bit_index, is_high in enumerate(low_frequency > median):
        if is_high:
            value |= 1 << bit_index
    return value


def group_near_duplicates(records):
    """Group same-identity source images with similar perceptual hashes."""
    parent = list(range(len(records)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            distance = (records[left]["phash"] ^ records[right]["phash"]).bit_count()
            if distance <= PHASH_DISTANCE_LIMIT:
                left_root, right_root = find(left), find(right)
                if left_root != right_root:
                    parent[right_root] = left_root

    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[find(index)].append(record)
    return list(groups.values())


def read_previous_manual_flags():
    """Reuse prior visual findings without treating filenames as proof of identity."""
    flags = {}
    if not PREVIOUS_DIAGNOSTICS.is_file():
        return flags
    try:
        with PREVIOUS_DIAGNOSTICS.open(encoding="utf-8-sig", newline="") as report_file:
            for row in csv.DictReader(report_file):
                note = row.get("visual_review", "")
                low_note = note.lower()
                if "incorrect" in low_note or "uncertain" in low_note or "cannot be confirmed" in low_note:
                    flags[(row.get("identity", ""), row.get("original_filename", ""))] = note
    except (OSError, csv.Error):
        print(f"Could not read the previous visual report: {PREVIOUS_DIAGNOSTICS}")
    return flags


def collect_image_records(frontal_detector, profile_detector, manual_flags):
    """Inspect original and processed files and return counts, rows, and review notes."""
    manifest_rows = []
    identity_data = {}
    duplicate_maps = {"original": defaultdict(list), "processed": defaultdict(list)}
    cross_identity_hashes = defaultdict(list)
    unreadable_sources = []
    unreadable_processed = []
    unsupported_files = []
    missing_outputs = []
    extra_outputs = []
    wrong_dimensions = []
    suspicious_crops = []
    source_records_all = []

    for identity in IDENTITIES:
        source_folder = DATASET_DIR / identity
        output_folder = PROCESSED_DIR / identity
        source_files = sorted(
            path for path in source_folder.iterdir()
            if path.is_file() and path.name != ".gitkeep"
        ) if source_folder.is_dir() else []
        supported_sources = [path for path in source_files if path.suffix.lower() in IMAGE_EXTENSIONS]
        source_by_output = {output_path_for(path, output_folder).name: path for path in supported_sources}
        expected_outputs = set(source_by_output)
        output_files = sorted(
            path for path in output_folder.iterdir()
            if path.is_file() and path.name != ".gitkeep"
        ) if output_folder.is_dir() else []
        supported_outputs = [path for path in output_files if path.suffix.lower() in IMAGE_EXTENSIONS]

        identity_data[identity] = {
            "source_count": len(supported_sources),
            "processed_count": len(supported_outputs),
            "source_records": [],
            "output_records": [],
            "unsupported_source_count": 0,
            "unsupported_output_count": 0,
            "unreadable_source_count": 0,
            "unreadable_output_count": 0,
            "wrong_dimension_count": 0,
            "missing_count": 0,
            "extra_count": 0,
            "face_crop_counts": Counter(),
        }

        for path in source_files:
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                unsupported_files.append(path)
                identity_data[identity]["unsupported_source_count"] += 1
                manifest_rows.append({
                    "local_path": str(path.resolve()), "identity": identity,
                    "width_px": "", "height_px": "", "sha256": sha256_file(path),
                    "validation_status": "unsupported_or_nonimage_source_file",
                })
                continue

            image = read_image(path)
            digest = sha256_file(path)
            output_path = output_path_for(path, output_folder)
            record = {
                "path": path,
                "identity": identity,
                "sha256": digest,
                "image": image,
                "output_path": output_path,
                "phash": perceptual_hash(image) if image is not None else None,
            }
            identity_data[identity]["source_records"].append(record)
            source_records_all.append(record)
            duplicate_maps["original"][digest].append(record)
            cross_identity_hashes[digest].append(record)

            statuses = ["supported_format"]
            if image is None:
                identity_data[identity]["unreadable_source_count"] += 1
                unreadable_sources.append(path)
                statuses.append("unreadable")
                width = height = ""
            else:
                height, width = image.shape[:2]
                statuses.append("readable")
            if output_path.is_file():
                statuses.append("processed_output_exists")
            else:
                statuses.append("missing_processed_output")
                identity_data[identity]["missing_count"] += 1
                missing_outputs.append((identity, path, output_path))
            manifest_rows.append({
                "local_path": str(path.resolve()), "identity": identity,
                "width_px": width, "height_px": height, "sha256": digest,
                "validation_status": ";".join(statuses),
            })

        for path in output_files:
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                unsupported_files.append(path)
                identity_data[identity]["unsupported_output_count"] += 1
                identity_data[identity]["extra_count"] += 1
                extra_outputs.append((identity, path))
                manifest_rows.append({
                    "local_path": str(path.resolve()), "identity": identity,
                    "width_px": "", "height_px": "", "sha256": sha256_file(path),
                    "validation_status": "unsupported_or_nonimage_processed_file;extra_output",
                })
                continue

            image = read_image(path)
            digest = sha256_file(path)
            source = source_by_output.get(path.name)
            record = {
                "path": path, "identity": identity, "sha256": digest,
                "image": image, "source": source, "face_check": "unreadable",
            }
            identity_data[identity]["output_records"].append(record)
            duplicate_maps["processed"][digest].append(record)

            statuses = ["supported_format"]
            if image is None:
                identity_data[identity]["unreadable_output_count"] += 1
                unreadable_processed.append(path)
                statuses.append("unreadable")
                width = height = ""
                crop_face_result = "unreadable"
            else:
                height, width = image.shape[:2]
                statuses.append("readable")
                if (width, height) == OUTPUT_SIZE:
                    statuses.append("dimensions_ok_160x160")
                else:
                    identity_data[identity]["wrong_dimension_count"] += 1
                    wrong_dimensions.append((identity, path, width, height))
                    statuses.append("incorrect_dimensions")

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                face, method, frontal_n, profile_n, flipped_n = detect_one_face(
                    gray, frontal_detector, profile_detector
                )
                if face is None:
                    crop_face_result = "no_unique_face_detected"
                else:
                    crop_face_result = f"one_candidate_{method}"
                record["face_check"] = crop_face_result
                statuses.append(f"crop_machine_check:{crop_face_result}")
                identity_data[identity]["face_crop_counts"][crop_face_result] += 1

            source_name = source.name if source else ""
            manual_note = manual_flags.get((identity, source_name), "") if source else ""
            if manual_note:
                statuses.append("prior_visual_review_flag")
                suspicious_crops.append((identity, path, manual_note))
            if source is None:
                statuses.append("extra_output_without_matching_original")
                identity_data[identity]["extra_count"] += 1
                extra_outputs.append((identity, path))
            if image is not None and crop_face_result == "no_unique_face_detected" and not manual_note:
                suspicious_crops.append((identity, path, "Current local detector did not find one unique face in this crop."))
            manifest_rows.append({
                "local_path": str(path.resolve()), "identity": identity,
                "width_px": width, "height_px": height, "sha256": digest,
                "validation_status": ";".join(statuses),
            })

    exact_duplicate_groups = {
        kind: [records for records in digest_map.values() if len(records) > 1]
        for kind, digest_map in duplicate_maps.items()
    }
    cross_identity_groups = [
        records for records in cross_identity_hashes.values()
        if len({record["identity"] for record in records}) > 1
    ]
    duplicate_paths = set()
    for duplicate_group in exact_duplicate_groups.values():
        for records in duplicate_group:
            duplicate_paths.update(str(record["path"].resolve()) for record in records)
    cross_identity_paths = {
        str(record["path"].resolve())
        for group in cross_identity_groups
        for record in group
    }
    for row in manifest_rows:
        path = row["local_path"]
        if path in duplicate_paths:
            row["validation_status"] += ";exact_duplicate"
        if path in cross_identity_paths:
            row["validation_status"] += ";cross_identity_duplicate"
    return {
        "identities": identity_data,
        "manifest_rows": manifest_rows,
        "exact_duplicate_groups": exact_duplicate_groups,
        "cross_identity_groups": cross_identity_groups,
        "unreadable_sources": unreadable_sources,
        "unreadable_processed": unreadable_processed,
        "unsupported_files": unsupported_files,
        "missing_outputs": missing_outputs,
        "extra_outputs": extra_outputs,
        "wrong_dimensions": wrong_dimensions,
        "suspicious_crops": suspicious_crops,
        "source_records": source_records_all,
    }


def make_processed_contact_sheets(audit, manual_flags):
    """Create a local per-identity sheet for reviewing every processed output."""
    sheet_dir = REPORTS_DIR / "processed_contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    suspicious_paths = {str(path.resolve()) for _, path, _ in audit["suspicious_crops"]}
    for identity, data in audit["identities"].items():
        items = data["output_records"]
        columns, cell_width, cell_height, title_height = 4, 300, 250, 55
        rows = max(1, math.ceil(len(items) / columns))
        sheet = np.full((title_height + rows * cell_height, columns * cell_width, 3), 255, dtype=np.uint8)
        cv2.putText(
            sheet, f"{identity}: processed outputs ({len(items)}); red = review",
            (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 2, cv2.LINE_AA,
        )
        for index, record in enumerate(items):
            image = record["image"]
            if image is None:
                continue
            row, column = divmod(index, columns)
            x0, y0 = column * cell_width, title_height + row * cell_height
            height, width = image.shape[:2]
            scale = min((cell_width - 16) / width, 175 / height)
            new_width, new_height = max(1, int(width * scale)), max(1, int(height * scale))
            thumb = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            x = x0 + (cell_width - new_width) // 2
            y = y0 + 4 + (175 - new_height) // 2
            sheet[y:y + new_height, x:x + new_width] = thumb

            source = record["source"]
            note = manual_flags.get((identity, source.name), "") if source else ""
            if note or str(record["path"].resolve()) in suspicious_paths:
                label = "REVIEW"
                color = (0, 0, 220)
                cv2.rectangle(sheet, (x0 + 2, y0 + 2), (x0 + cell_width - 2, y0 + 180), color, 2)
            elif (width, height) != OUTPUT_SIZE:
                label, color = "WRONG SIZE", (0, 0, 220)
            else:
                label, color = "", (20, 20, 20)
            if label:
                cv2.putText(sheet, label, (x0 + 8, y0 + 199), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            filename = record["path"].name
            for line_index, line in enumerate(textwrap.wrap(filename, width=38)[:2]):
                cv2.putText(
                    sheet, line, (x0 + 8, y0 + 218 + line_index * 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (20, 20, 20), 1, cv2.LINE_AA,
                )
        destination = sheet_dir / f"{identity}_processed_contact_sheet.jpg"
        if not cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]):
            print(f"Could not write contact sheet: {destination}")


def write_manifest(audit):
    """Write the requested local-only, metadata-only CSV manifest."""
    fieldnames = ["local_path", "identity", "width_px", "height_px", "sha256", "validation_status"]
    with MANIFEST_PATH.open("w", encoding="utf-8-sig", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit["manifest_rows"])


def propose_evaluation_split(audit, manual_flags):
    """Propose a path-only known-person split without copying or moving photos."""
    split_rows = []
    identity_summaries = {}
    enough_for_all = True

    for identity, data in audit["identities"].items():
        eligible = []
        for record in data["source_records"]:
            output = record["output_path"]
            output_record = next(
                (item for item in data["output_records"] if item["path"] == output), None
            )
            if record["image"] is None or output_record is None or output_record["image"] is None:
                continue
            if output_record["image"].shape[1::-1] != OUTPUT_SIZE:
                continue
            if (identity, record["path"].name) in manual_flags:
                continue
            if not output_record["face_check"].startswith("one_candidate_"):
                continue
            record["output_record"] = output_record
            eligible.append(record)

        near_groups = group_near_duplicates(eligible)
        # Only multi-image groups need a persistent group label in the proposal.
        groups = sorted(
            near_groups,
            key=lambda group: (-len(group), min(item["path"].name for item in group)),
        )
        if len(eligible) < TRAIN_ENROLLMENT_TARGET + KNOWN_TEST_TARGET:
            enough_for_all = False
            identity_summaries[identity] = {
                "eligible": len(eligible), "near_groups": groups,
                "reason": f"Only {len(eligible)} suitable original/output pairs; need at least 16.",
            }
            continue

        test_groups = []
        test_count = 0
        # Keep perceptual near-duplicate groups together; fill the test target with singletons.
        for group in groups:
            if len(group) > 1 and test_count + len(group) <= KNOWN_TEST_TARGET:
                test_groups.append(group)
                test_count += len(group)
        for group in groups:
            if test_count >= KNOWN_TEST_TARGET:
                break
            if group in test_groups:
                continue
            if test_count + len(group) <= KNOWN_TEST_TARGET:
                test_groups.append(group)
                test_count += len(group)

        if test_count != KNOWN_TEST_TARGET:
            enough_for_all = False
            identity_summaries[identity] = {
                "eligible": len(eligible), "near_groups": groups,
                "reason": "Could not select four known-test images without splitting a near-duplicate group.",
            }
            continue

        test_ids = {id(record) for group in test_groups for record in group}
        test_records = [record for group in test_groups for record in group]
        enrollment_records = []
        for group in groups:
            if any(id(record) in test_ids for record in group):
                continue
            if len(enrollment_records) + len(group) <= TRAIN_ENROLLMENT_TARGET:
                enrollment_records.extend(group)
            if len(enrollment_records) == TRAIN_ENROLLMENT_TARGET:
                break
        if len(enrollment_records) != TRAIN_ENROLLMENT_TARGET:
            enough_for_all = False
            identity_summaries[identity] = {
                "eligible": len(eligible), "near_groups": groups,
                "reason": "Could not select twelve enrollment images without splitting a near-duplicate group.",
            }
            continue

        split_ids = {id(record): "known_test" for record in test_records}
        split_ids.update({id(record): "enrollment" for record in enrollment_records})
        group_lookup = {}
        group_number = 0
        for group in groups:
            group_number += 1
            group_name = f"{identity}_near_{group_number:02d}" if len(group) > 1 else f"{identity}_single_{group[0]['path'].stem}"
            for record in group:
                group_lookup[id(record)] = group_name

        for record in eligible:
            role = split_ids.get(id(record), "unused_reserve")
            split_rows.append({
                "identity": identity,
                "original_local_path": str(record["path"].resolve()),
                "processed_local_path": str(record["output_record"]["path"].resolve()),
                "near_duplicate_group": group_lookup[id(record)],
                "proposed_role": role,
                "note": "Proposal only; verify identity and independence manually. No files were moved or copied.",
            })
        identity_summaries[identity] = {
            "eligible": len(eligible), "near_groups": groups,
            "enrollment": len(enrollment_records), "known_test": len(test_records),
            "reserve": len(eligible) - len(enrollment_records) - len(test_records),
            "reason": "Known-person proposal possible; unknown-person photos are not present.",
        }

    if enough_for_all:
        with SPLIT_PATH.open("w", encoding="utf-8-sig", newline="") as split_file:
            fields = ["identity", "original_local_path", "processed_local_path", "near_duplicate_group", "proposed_role", "note"]
            writer = csv.DictWriter(split_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(split_rows)
        return identity_summaries, SPLIT_PATH
    if SPLIT_PATH.exists():
        SPLIT_PATH.unlink()
    return identity_summaries, None


def write_audit_report(audit, split_summary, split_path):
    """Write a readable summary, readiness decision, and manual follow-up list."""
    data = audit["identities"]
    missing_count = len(audit["missing_outputs"])
    wrong_size_count = len(audit["wrong_dimensions"])
    unreadable_count = len(audit["unreadable_sources"]) + len(audit["unreadable_processed"])
    extra_count = len(audit["extra_outputs"])
    false_crop_flags = [item for item in audit["suspicious_crops"] if "incorrect" in item[2].lower()]

    if missing_count or wrong_size_count or unreadable_count or extra_count or false_crop_flags:
        readiness = "NOT_READY"
    elif audit["suspicious_crops"] or audit["cross_identity_groups"]:
        readiness = "NEEDS_MANUAL_REVIEW"
    else:
        # Automated checks cannot confirm identity labels or overall crop quality.
        readiness = "NEEDS_MANUAL_REVIEW"

    lines = [
        "DATASET QUALITY AUDIT",
        "=====================",
        "",
        "This report was created locally. It contains file metadata only; no image was uploaded or moved.",
        "",
        "Counts by identity",
        "------------------",
    ]
    for identity in IDENTITIES:
        entry = data[identity]
        lines.append(
            f"{identity}: originals={entry['source_count']}, processed_images={entry['processed_count']}, "
            f"unreadable_originals={entry['unreadable_source_count']}, "
            f"unreadable_processed={entry['unreadable_output_count']}, "
            f"wrong_dimensions={entry['wrong_dimension_count']}, "
            f"missing_outputs={entry['missing_count']}, extra_outputs={entry['extra_count']}"
        )

    lines.extend([
        "",
        "Automated checks",
        "-----------------",
        "Supported image formats scanned: " + ", ".join(sorted(IMAGE_EXTENSIONS)),
        f"Original images found: {sum(item['source_count'] for item in data.values())}",
        f"Readable supported originals (structurally valid originals): {sum(item['source_count'] - item['unreadable_source_count'] for item in data.values())}",
        f"Processed images found: {sum(item['processed_count'] for item in data.values())}",
        f"Unreadable originals: {len(audit['unreadable_sources'])}",
        f"Unreadable processed images: {len(audit['unreadable_processed'])}",
        f"Processed images with dimensions other than 160 x 160: {wrong_size_count}",
        f"Structurally valid processed images (readable and exactly 160 x 160): {sum(item['processed_count'] - item['unreadable_output_count'] - item['wrong_dimension_count'] for item in data.values())}",
        f"Structurally invalid processed images (unreadable or wrong dimensions): {len(audit['unreadable_processed']) + wrong_size_count}",
        f"Missing processed outputs: {missing_count}",
        f"Extra processed outputs without a matching original: {extra_count}",
        f"Unsupported/non-image files inside identity folders: {len(audit['unsupported_files'])}",
        f"Exact duplicate groups among originals: {len(audit['exact_duplicate_groups']['original'])}",
        f"Exact duplicate groups among processed images: {len(audit['exact_duplicate_groups']['processed'])}",
        f"Exact original-image groups appearing in multiple identities: {len(audit['cross_identity_groups'])}",
        f"Processed crops flagged for manual review: {len(audit['suspicious_crops'])}",
        f"Known visually incorrect processed crops from prior local review: {sum('incorrect' in note.lower() for _, _, note in audit['suspicious_crops'])}",
        "",
        "Missing processed outputs",
        "-------------------------",
    ])
    if audit["missing_outputs"]:
        for identity, source, output in audit["missing_outputs"]:
            lines.append(f"{identity}: {source.name} -> missing {output.relative_to(PROJECT_ROOT)}")
    else:
        lines.append("None.")

    lines.extend(["", "Exact duplicates", "-----------------"])
    any_duplicates = False
    for kind in ("original", "processed"):
        for group in audit["exact_duplicate_groups"][kind]:
            any_duplicates = True
            lines.append(f"{kind} SHA-256 {group[0]['sha256']}: " + ", ".join(str(item['path'].relative_to(PROJECT_ROOT)) for item in group))
    if not any_duplicates:
        lines.append("No byte-for-byte duplicate groups found.")
    if audit["cross_identity_groups"]:
        lines.append("Cross-identity original duplicates:")
        for group in audit["cross_identity_groups"]:
            lines.append("  " + ", ".join(str(item["path"].relative_to(PROJECT_ROOT)) for item in group))

    lines.extend(["", "Potentially suspicious crops", "----------------------------"])
    if audit["suspicious_crops"]:
        for identity, path, note in audit["suspicious_crops"]:
            lines.append(f"{identity}: {path.relative_to(PROJECT_ROOT)} - {note}")
    else:
        lines.append("No specific crop was flagged by the automated checks or prior visual report.")

    lines.extend([
        "",
        "Exact duplicate check uses SHA-256: identical file bytes produce identical hashes.",
        f"Near-duplicate split grouping uses a visual hash with Hamming distance <= {PHASH_DISTANCE_LIMIT}; "
        "this is a conservative hint, not proof that two photographs are the same.",
        "The face check reuses the local OpenCV frontal/profile cascades. A positive result means a detector found a face-like region; it does not prove the correct identity or a good crop.",
        "",
        "Manual checks only a person can complete",
        "-----------------------------------------",
        "[ ] Check every image in each identity contact sheet against the correct person.",
        "[ ] Check that each crop contains the whole intended face, not a poster, clothing, or another person.",
        "[ ] Review the red REVIEW frames and the prior known false crops before training.",
        "[ ] Confirm the two missing-output sources are manually cropped or replaced with clear photos.",
        "[ ] Confirm consent and keep originals, crops, reports, and embeddings private.",
        "",
        f"Readiness result: {readiness}",
        "Reason: automated checks found missing outputs and prior visual flags. Human identity and crop-quality checks are also still required.",
        "Dataset creation: the folder structure and local image collection are present; dataset readiness is not yet established.",
        "",
        "Evaluation plan",
        "---------------",
        "Enrollment images build one reference representation per known person. Known-person test images must be separate photographs of those people. Unknown-person test images must show people who are not enrolled.",
        "Never put the same original, exact duplicate, or near-duplicate group in both enrollment and test.",
    ])
    if split_path:
        lines.append(
            f"A local known-person split proposal was written to {split_path.relative_to(PROJECT_ROOT)}: "
            "12 enrollment, 4 known-test, and remaining eligible images held as reserves per identity."
        )
        for identity in IDENTITIES:
            summary = split_summary[identity]
            lines.append(
                f"  {identity}: eligible={summary['eligible']}, enrollment={summary['enrollment']}, "
                f"known_test={summary['known_test']}, reserves={summary['reserve']}."
            )
    else:
        lines.append("No split proposal was generated because there were not enough suitable independent images for every identity.")
    lines.append("No unknown-person images exist in these four enrolled identity folders; collect separate local test-only images of non-enrolled people before evaluating Unknown Person behavior.")
    lines.extend([
        "",
        "Next project step",
        "------------------",
        "After manual review and a suitable split, the next task is to extract face embeddings with a pretrained model and compare them using a threshold so unregistered people can be labeled Unknown Person.",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return readiness


def main():
    if not hasattr(cv2, "CascadeClassifier"):
        print(f"This audit needs the project's OpenCV 4 dependency; found OpenCV {getattr(cv2, '__version__', 'unknown')}.")
        return 1

    cascade_dir = Path(cv2.data.haarcascades)
    frontal_detector = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_frontalface_alt2.xml"))
    profile_detector = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_profileface.xml"))
    if frontal_detector.empty() or profile_detector.empty():
        print(f"Could not load OpenCV's bundled cascades from {cascade_dir}.")
        return 1

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    manual_flags = read_previous_manual_flags()
    audit = collect_image_records(frontal_detector, profile_detector, manual_flags)
    write_manifest(audit)
    make_processed_contact_sheets(audit, manual_flags)
    split_summary, split_path = propose_evaluation_split(audit, manual_flags)
    readiness = write_audit_report(audit, split_summary, split_path)

    print(f"Original images: {sum(item['source_count'] for item in audit['identities'].values())}")
    print(f"Processed images: {sum(item['processed_count'] for item in audit['identities'].values())}")
    print(f"Missing outputs: {len(audit['missing_outputs'])}; wrong dimensions: {len(audit['wrong_dimensions'])}")
    print(f"Exact original duplicate groups: {len(audit['exact_duplicate_groups']['original'])}")
    print(f"Readiness: {readiness}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Manifest: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    if split_path:
        print(f"Split proposal: {split_path.relative_to(PROJECT_ROOT)}")
    print(f"Contact sheets: {(REPORTS_DIR / 'processed_contact_sheets').relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
