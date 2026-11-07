"""Extract local FaceNet embeddings from reviewed 160x160 face crops."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset" / "processed"
AUDIT_PATH = PROJECT_ROOT / "reports" / "dataset_audit.txt"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "embeddings"
IDENTITIES = ("kowshik", "murthy", "surekha", "vijay")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
WEIGHT_FILENAME = "20180402-114759-vggface2.pt"


def get_audit_status():
    """Read the audit status for information; it does not gate extraction."""
    if not AUDIT_PATH.is_file():
        return "AUDIT_NOT_FOUND"
    report = AUDIT_PATH.read_text(encoding="utf-8", errors="replace")
    readiness_lines = [line.strip() for line in report.splitlines() if line.startswith("Readiness result:")]
    if len(readiness_lines) != 1:
        return "AUDIT_STATUS_UNAVAILABLE"
    return readiness_lines[0].partition(":")[2].strip()


def get_weights_path(torch):
    """Return the expected local cache file without triggering a download."""
    configured_home = os.environ.get("TORCH_HOME")
    if configured_home:
        torch_home = Path(configured_home).expanduser()
    else:
        cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
        torch_home = cache_home / "torch"
    return torch_home / "checkpoints" / WEIGHT_FILENAME


def load_model(torch, weights_path, device):
    """Load only the locally cached VGGFace2 weights; never invoke auto-download."""
    from facenet_pytorch import InceptionResnetV1

    model = InceptionResnetV1(classify=True, num_classes=8631)
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError(
            "This installed PyTorch lacks safe weights_only loading. Update the Python/PyTorch setup first."
        ) from error
    model.load_state_dict(state)
    model.classify = False
    return model.to(device).eval()


def image_paths():
    """List direct children of processed identity folders only."""
    for identity in IDENTITIES:
        folder = DATASET_DIR / identity
        if not folder.is_dir():
            raise RuntimeError(f"Processed identity folder is missing: {folder.relative_to(PROJECT_ROOT)}")
        for path in sorted(folder.iterdir(), key=lambda item: item.name.casefold()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield identity, path


def make_tensor(torch, image):
    """Convert an OpenCV BGR crop to RGB, 160x160, float [-1, 1], NCHW."""
    import cv2
    import numpy as np

    if image is None:
        raise ValueError("OpenCV could not decode the image")
    if image.shape[:2] != (160, 160):
        raise ValueError(f"expected 160x160 crop, got {image.shape[1]}x{image.shape[0]}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    array = np.asarray(rgb, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def unique_output_paths():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"facenet_vggface2_{stamp}"
    npz_path = OUTPUT_DIR / f"{base}.npz"
    json_path = OUTPUT_DIR / f"{base}.json"
    suffix = 1
    while npz_path.exists() or json_path.exists():
        npz_path = OUTPUT_DIR / f"{base}_{suffix}.npz"
        json_path = OUTPUT_DIR / f"{base}_{suffix}.json"
        suffix += 1
    return npz_path, json_path


def main():
    audit_status = get_audit_status()
    if audit_status != "READY_FOR_PROTOTYPE":
        print(f"NOTE: Day 5 audit status is {audit_status}. Continuing with available processed crops as requested.")
        print("Missing processed crops are skipped because no file exists for them.")

    try:
        import cv2
        import numpy as np
        import torch
        import facenet_pytorch  # noqa: F401 - checked before loading any photographs
    except ImportError as error:
        print(f"STOP: Required package '{error.name}' is missing for this Python: {sys.executable}")
        print("Activate the project's compatible virtual environment, then install requirements there.")
        return 1

    weights_path = get_weights_path(torch)
    if not weights_path.is_file():
        print("STOP: The local VGGFace2 FaceNet weights are not installed.")
        print(f"Expected local file: {weights_path}")
        print("No download was attempted. Obtain the official 107 MB file only after explicit approval, then place it at this path.")
        return 1

    try:
        paths = list(image_paths())
    except RuntimeError as error:
        print(f"STOP: {error}")
        return 1
    if not paths:
        print("STOP: No supported processed images were found.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = load_model(torch, weights_path, device)
    except Exception as error:
        print(f"STOP: Could not load the local model weights safely: {error}")
        return 1

    rows = []
    vectors = []
    skipped = 0
    for identity, path in paths:
        try:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            tensor = make_tensor(torch, image).to(device)
            with torch.no_grad():
                vector = model(tensor).squeeze(0).detach().cpu().numpy().astype(np.float32)
            if vector.shape != (512,) or not np.isfinite(vector).all():
                raise ValueError(f"model returned invalid embedding shape/values: {vector.shape}")
            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                raise ValueError("model returned a zero vector")
            vector = vector / norm
            vectors.append(vector)
            rows.append({
                "identity": identity,
                "filename": path.name,
                "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "embedding_index": len(vectors) - 1,
                "dimensions": 512,
                "l2_normalized": True,
            })
            print(f"OK {identity}/{path.name}")
        except Exception as error:
            skipped += 1
            print(f"SKIP {identity}/{path.name}: {error}")

    if not vectors:
        print("STOP: No valid embeddings were produced; no output files were written.")
        return 1

    matrix = np.stack(vectors).astype(np.float32)
    metadata = {
        "model": "InceptionResnetV1 pretrained on VGGFace2",
        "embedding_dimensions": 512,
        "normalization": "RGB float32 scaled to [-1, 1]; output L2-normalized",
        "input_size": [160, 160],
        "device": str(device),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "count": len(rows),
        "skipped": skipped,
        "items": rows,
        "privacy": "Local biometric data. Do not commit, publish, or share.",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = unique_output_paths()
    npz_created = False
    json_created = False
    try:
        # Exclusive file creation preserves any prior local artifacts.
        with npz_path.open("xb") as output:
            npz_created = True
            np.savez_compressed(output, embeddings=matrix)
        with json_path.open("x", encoding="utf-8") as output:
            json_created = True
            json.dump(metadata, output, indent=2)
            output.write("\n")
    except Exception:
        if npz_created:
            npz_path.unlink(missing_ok=True)
        if json_created:
            json_path.unlink(missing_ok=True)
        raise

    print(f"\nFinished: {len(rows)} embeddings, {skipped} skipped, shape={matrix.shape}, device={device}")
    print(f"Local outputs: {npz_path.relative_to(PROJECT_ROOT)} and {json_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
