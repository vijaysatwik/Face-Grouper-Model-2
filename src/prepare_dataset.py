"""Create local face crops without changing original photographs."""

import csv
from pathlib import Path
import sys

try:
    import cv2
except ImportError:
    print(
        "OpenCV is not installed. Install the project requirements with "
        "'python -m pip install -r requirements.txt'."
    )
    raise SystemExit(1)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
PROCESSED_DIR = DATASET_DIR / "processed"
DIAGNOSTICS_DIR = DATASET_DIR / "diagnostics"
IDENTITIES = ("kowshik", "murthy", "surekha", "vijay")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
OUTPUT_SIZE = (160, 160)
FACE_MARGIN = 1.4
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 7
MIN_FACE_SIZE = (60, 60)


def make_square_crop(image, face):
    """Return a square crop around a face, clipped to the image boundaries."""
    image_height, image_width = image.shape[:2]
    x, y, width, height = map(int, face)
    side = min(
        max(1, round(max(width, height) * FACE_MARGIN)),
        image_width,
        image_height,
    )

    center_x = x + width / 2
    center_y = y + height / 2
    left = max(0, min(round(center_x - side / 2), image_width - side))
    top = max(0, min(round(center_y - side / 2), image_height - side))
    return image[top : top + side, left : left + side]


def output_path_for(image_path, output_folder):
    """Include the source extension to avoid same-name collisions."""
    extension_label = image_path.suffix.lower().lstrip(".")
    return output_folder / f"{image_path.stem}_{extension_label}.jpg"


def detect_one_face(gray, frontal_detector, profile_detector):
    """Use strict frontal detection, then a single unambiguous profile result."""
    frontal_faces = frontal_detector.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=MIN_FACE_SIZE,
    )
    if len(frontal_faces) == 1:
        return frontal_faces[0], "frontal", len(frontal_faces), 0, 0

    profile_faces = profile_detector.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=5,
        minSize=MIN_FACE_SIZE,
    )
    flipped_faces = profile_detector.detectMultiScale(
        cv2.flip(gray, 1),
        scaleFactor=SCALE_FACTOR,
        minNeighbors=5,
        minSize=MIN_FACE_SIZE,
    )

    # A flipped-image box must be mapped back to the original image coordinates.
    if len(profile_faces) + len(flipped_faces) == 1:
        if len(profile_faces) == 1:
            face = profile_faces[0]
            method = "profile"
        else:
            x, y, width, height = map(int, flipped_faces[0])
            face = (gray.shape[1] - x - width, y, width, height)
            method = "flipped_profile"
        return face, method, len(frontal_faces), len(profile_faces), len(flipped_faces)

    return None, "ambiguous", len(frontal_faces), len(profile_faces), len(flipped_faces)


def process_identity(identity, frontal_detector, profile_detector, report_rows):
    input_folder = DATASET_DIR / identity
    output_folder = PROCESSED_DIR / identity
    output_folder.mkdir(parents=True, exist_ok=True)

    counts = {
        "found": 0,
        "processed": 0,
        "profile_processed": 0,
        "no_face": 0,
        "multiple_faces": 0,
        "unreadable": 0,
        "existing": 0,
        "errors": 0,
    }

    if not input_folder.is_dir():
        print(f"{identity}: source folder is missing: {input_folder}")
        return counts

    # Read only files directly inside an identity folder, never processed subfolders.
    image_paths = sorted(
        path
        for path in input_folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    counts["found"] = len(image_paths)

    for image_path in image_paths:
        destination = output_path_for(image_path, output_folder)
        row = {
            "identity": identity,
            "filename": image_path.name,
            "width": "",
            "height": "",
            "readable": "no",
            "frontal_regions": "",
            "profile_regions": "",
            "flipped_profile_regions": "",
            "decision": "",
            "status": "",
            "reason": "",
            "recommended_action": "",
            "output_file": str(destination.relative_to(PROJECT_ROOT)),
        }

        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        except cv2.error as error:
            image = None
            row["reason"] = f"OpenCV could not read this file: {error}"

        if image is None:
            counts["unreadable"] += 1
            row.update(
                status="unreadable",
                reason=row["reason"] or "OpenCV could not decode this image.",
                recommended_action="Check the file locally or replace it with a readable supported image.",
            )
            report_rows.append(row)
            print(f"SKIP (unreadable or unsupported image): {image_path.name}")
            continue

        height, width = image.shape[:2]
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face, method, frontal_count, profile_count, flipped_count = detect_one_face(
                gray, frontal_detector, profile_detector
            )
        except cv2.error as error:
            counts["unreadable"] += 1
            row.update(
                width=width,
                height=height,
                readable="yes",
                status="analysis_error",
                reason=f"OpenCV could read the file but failed while analyzing it: {error}",
                recommended_action="Review the image file and try again; the original was not changed.",
            )
            report_rows.append(row)
            print(f"SKIP (could not analyze image): {image_path.name} ({error})")
            continue
        row.update(
            width=width,
            height=height,
            readable="yes",
            frontal_regions=frontal_count,
            profile_regions=profile_count,
            flipped_profile_regions=flipped_count,
            decision=method,
        )

        if destination.exists():
            counts["existing"] += 1
            row.update(
                status="existing_output",
                reason=(
                    "A processed file already exists and was left untouched. "
                    f"Current detector decision: {method}; no output was overwritten."
                ),
                recommended_action="Review the existing crop in the processed contact sheet before using it.",
            )
            report_rows.append(row)
            print(f"SKIP (output already exists and was preserved): {image_path.name}")
            continue

        if face is None:
            candidates = frontal_count + profile_count + flipped_count
            if candidates == 0:
                counts["no_face"] += 1
                row.update(
                    status="no_face",
                    reason="Neither bundled frontal nor profile detector found a face at the chosen settings.",
                    recommended_action="Review pose, face size, blur, and lighting; manually review or use a better local detector.",
                )
                print(f"SKIP (no face detected): {image_path.name}")
            else:
                counts["multiple_faces"] += 1
                row.update(
                    status="ambiguous",
                    reason=(
                        f"Detectors returned {frontal_count} frontal, {profile_count} profile, "
                        f"and {flipped_count} flipped-profile region(s); no single candidate was safe to choose."
                    ),
                    recommended_action="Inspect the contact sheet and manually review; the script will not guess.",
                )
                print(f"SKIP (ambiguous face detections): {image_path.name}")
            report_rows.append(row)
            continue

        try:
            crop = make_square_crop(image, face)
            resized = cv2.resize(crop, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
            success, encoded = cv2.imencode(".jpg", resized)
            if not success:
                raise OSError("OpenCV could not encode the JPEG output")

            # Exclusive file creation makes it impossible to overwrite an existing output.
            with destination.open("xb") as output_file:
                output_file.write(encoded.tobytes())
            counts["processed"] += 1
            if method != "frontal":
                counts["profile_processed"] += 1
            row.update(
                status="processed",
                reason=f"Exactly one unambiguous {method.replace('_', ' ')} face region was selected.",
                recommended_action="Review the crop visually before using it for model training.",
            )
            print(f"SAVED ({method}): {destination.relative_to(PROJECT_ROOT)}")
        except FileExistsError:
            counts["existing"] += 1
            row.update(
                status="existing_output",
                reason="A processed file appeared before save; it was left untouched.",
                recommended_action="Review the existing crop visually.",
            )
            print(f"SKIP (output already exists): {destination.name}")
        except (OSError, cv2.error) as error:
            counts["errors"] += 1
            row.update(
                status="save_error",
                reason=str(error),
                recommended_action="Check local disk access and try again.",
            )
            print(f"ERROR (could not save output for {image_path.name}): {error}")
        report_rows.append(row)

    print(
        f"\n{identity} summary: images found={counts['found']}, "
        f"new crops={counts['processed']} (profile fallback={counts['profile_processed']}), "
        f"no face={counts['no_face']}, ambiguous/multiple={counts['multiple_faces']}, "
        f"unreadable={counts['unreadable']}, existing outputs preserved={counts['existing']}, "
        f"save errors={counts['errors']}\n"
    )
    return counts


def main():
    if not hasattr(cv2, "CascadeClassifier"):
        version = getattr(cv2, "__version__", "unknown")
        print(
            f"This script needs OpenCV 4, but OpenCV {version} is installed. "
            "Install the project dependency with "
            "'python -m pip install --force-reinstall -r requirements.txt'."
        )
        return 1

    cascade_dir = Path(cv2.data.haarcascades)
    frontal_path = cascade_dir / "haarcascade_frontalface_alt2.xml"
    profile_path = cascade_dir / "haarcascade_profileface.xml"
    frontal_detector = cv2.CascadeClassifier(str(frontal_path))
    profile_detector = cv2.CascadeClassifier(str(profile_path))
    if frontal_detector.empty() or profile_detector.empty():
        print(f"Could not load OpenCV's bundled face cascades from: {cascade_dir}")
        return 1

    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    report_rows = []
    for identity in IDENTITIES:
        process_identity(identity, frontal_detector, profile_detector, report_rows)

    report_path = DIAGNOSTICS_DIR / "face_detection_diagnostics.csv"
    fieldnames = [
        "identity", "filename", "width", "height", "readable", "frontal_regions",
        "profile_regions", "flipped_profile_regions", "decision", "status", "reason",
        "recommended_action", "output_file",
    ]
    with report_path.open("w", newline="", encoding="utf-8-sig") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"Diagnostic report saved locally: {report_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
