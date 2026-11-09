# Face Recognition Project

## Goal

Build a custom Python application that recognizes faces using a pretrained face embedding model.

## Planned stack

- Python for project code
- A pretrained face embedding model for face representations
- Streamlit for the user interface
- Git and GitHub for version control and collaboration

## Setup status

The repository includes local face preprocessing, FaceNet embedding extraction, and a cosine-similarity matching prototype. Unknown-person rejection and the Streamlit interface have not been built.

## Development plan

The project is being built incrementally: prepare and audit the local dataset, extract embeddings, compare query faces with identity references, then calibrate unknown-person rejection before building the Streamlit interface.

Keep private photographs, identity data, credentials, model weights, and generated embeddings out of Git. See `.gitignore` for local exclusions.

## Day 6: local FaceNet embeddings

`src/extract_embeddings.py` extracts one 512-number embedding per reviewed 160x160 crop using the `facenet-pytorch` InceptionResnetV1 model pretrained on VGGFace2. It reads only direct image files inside `dataset/processed/<identity>/`, processes them locally, and writes a compressed `.npz` matrix plus a JSON index under `artifacts/embeddings/`. These are private biometric data and are ignored by Git.

The audit is printed as an informational note and does not block extraction. The script reads only processed image files that exist, so missing crops are naturally skipped. Review the contact sheets and keep known non-face images out of `dataset/processed/<identity>/` before running; the two previously incorrect crops were manually replaced from their original photos. Other dataset concerns remain documented in the Day 5 audit.

The script also requires the VGGFace2 weight file in the PyTorch cache. `facenet-pytorch` can download weights automatically, so this script avoids that behavior and loads only an existing local file. The expected filename is `20180402-114759-vggface2.pt` under `%USERPROFILE%\.cache\torch\checkpoints\` by default (or `%TORCH_HOME%\checkpoints\` when `TORCH_HOME` is set). The official file is about 107 MB. Do not download it without approval; if authorized, use the official [facenet-pytorch VGGFace2 release](https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt).

The project environment uses Python 3.11 with CPU PyTorch 2.2.2 and torchvision 0.17.2. Keep using the project's `.venv` for these commands.

After the compatible environment is set up, run these commands from the repository root:

```powershell
$python311 = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'
& $python311 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python src/extract_embeddings.py
```

Package installation and model-weight setup are manual. The program does not download model weights, and it never changes the source or processed photographs. Each run uses new output filenames so existing embeddings are preserved.

## Private dataset and preprocessing

Place original photographs directly in the matching identity folder:

```text
dataset/
├── kowshik/   # Original photos for this identity
├── murthy/    # Original photos for this identity
├── surekha/   # Original photos for this identity
└── vijay/     # Original photos for this identity
```

The folder name is the identity label. Keep each person's photographs in the matching folder. The preprocessing script reads supported image files directly inside these folders and leaves originals untouched. It uses OpenCV's bundled `frontalface_alt2` Haar cascade, then tries the bundled profile cascade (including a flipped image) only when the frontal result is not unique. A crop is saved only when that process leaves one unambiguous candidate; uncertain photos are skipped for manual review. Successful crops are square 160-by-160 JPEGs in the corresponding `dataset/processed/` folder. The script never overwrites an existing output.

Install the listed dependency in your own environment and run the script from the repository root:

```powershell
python -m pip install -r requirements.txt
python src/prepare_dataset.py
```

Find processed images under `dataset/processed/kowshik/`, `dataset/processed/murthy/`, `dataset/processed/surekha/`, and `dataset/processed/vijay/`. The script writes a local CSV diagnostic report to `dataset/diagnostics/face_detection_diagnostics.csv`. Keep original and processed photographs and diagnostic files private on your computer. `.gitignore` excludes dataset contents while allowing `.gitkeep` placeholders to be tracked.

## Day 7: compare face embeddings

Run the held-out evaluation from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python src/match_faces.py
```

The script picks the newest matching NPZ/JSON pair in `artifacts/embeddings/`. It uses the existing Day 5 proposal: 12 enrollment photos and 4 separate known-person test photos per identity. It verifies that original paths, exact file hashes, and near-duplicate groups do not cross those roles. The two missing processed crops do not have embeddings and are not invented or included. Seven other available embeddings remain unused by the evaluation proposal.

Cosine similarity measures the angle between two numeric face vectors. A value near 1 means the vectors point in a similar direction, 0 means they are perpendicular, and -1 means they point in opposite directions. For example, `[1, 0]` and `[0.8, 0.6]` have cosine similarity `0.8`. Day 6 vectors are already L2-normalized, so their dot product gives cosine similarity without normalizing them again.

For each identity, matching reports the individual reference scores and uses the mean of that identity's three highest scores as its aggregate. This avoids choosing an identity based only on one unusually high photo score. A higher score ranks that identity as a closer candidate; it does not prove who the person is. The project has no acceptance threshold or Unknown Person result yet; those are Day 8 work.

The evaluation is a small local experiment, not production accuracy. Its test queries are separate from its enrollment references, so an image is never tested against its own stored embedding. To match a new query crop, pass an already face-cropped 160x160 image:

```powershell
python src/match_faces.py --query-image "C:\path\to\a\new_160x160_face_crop.jpg"
```

This reuses the Day 6 model loading and RGB/[-1, 1] preprocessing. It loads weights only from the local PyTorch cache and does not download them. Query-image mode returns candidates only; use the no-argument command for the leakage-checked evaluation. Keep query images, metadata, and embedding files private and out of Git.
