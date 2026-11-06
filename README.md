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
