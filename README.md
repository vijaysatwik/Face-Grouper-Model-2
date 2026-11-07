# Face Recognition Project

## Goal

Build a custom Python application that recognizes faces using a pretrained face embedding model.

## Planned stack

- Python for project code
- A pretrained face embedding model for face representations
- Streamlit for the user interface
- Git and GitHub for version control and collaboration

## Setup status

The repository structure is initialized and a local OpenCV face preprocessing script is available. The face recognition model has not been built. Install the listed dependency when you are ready to run preprocessing.

## Development plan

The project will be built incrementally: establish the repository, add and validate the recognition pipeline, then build the Streamlit interface and refine the application. Dependencies and implementation will be introduced in later steps.

Keep private photographs, identity data, credentials, model weights, and generated embeddings out of Git. See `.gitignore` for local exclusions.

## Day 6: local FaceNet embeddings

`src/extract_embeddings.py` extracts one 512-number embedding per reviewed 160x160 crop using the `facenet-pytorch` InceptionResnetV1 model pretrained on VGGFace2. It reads only direct image files inside `dataset/processed/<identity>/`, processes them locally, and writes a compressed `.npz` matrix plus a JSON index under `artifacts/embeddings/`. These are private biometric data and are ignored by Git.

The audit is printed as an informational note and does not block extraction. The script reads only processed image files that exist, so missing crops are naturally skipped. Review the contact sheets and keep known non-face images out of `dataset/processed/<identity>/` before running; the two previously incorrect crops were manually replaced from their original photos. Other dataset concerns remain documented in the Day 5 audit.

The script also requires the VGGFace2 weight file in the PyTorch cache. `facenet-pytorch` can download weights automatically, so this script avoids that behavior and loads only an existing local file. The expected filename is `20180402-114759-vggface2.pt` under `%USERPROFILE%\.cache\torch\checkpoints\` by default (or `%TORCH_HOME%\checkpoints\` when `TORCH_HOME` is set). The official file is about 107 MB. Do not download it without approval; if authorized, use the official [facenet-pytorch VGGFace2 release](https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt).

The current computer uses Python 3.13. The pinned `facenet-pytorch` 2.6.0 release requires PyTorch 2.2.x and torchvision 0.17.x, while the official Windows PyTorch setup page currently documents Python 3.9–3.12. Create a separate Python 3.11 environment before installing dependencies; do not use Python 3.13 for this pinned stack.

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
