# Face Recognition Project

## Goal

Build a custom Python application that recognizes faces using a pretrained face embedding model.

## Planned stack

- Python for project code
- A pretrained face embedding model for face representations
- Streamlit for the user interface
- Git and GitHub for version control and collaboration

## Setup status

Day 1 repository structure is initialized. No recognition code or dependencies have been added.

## Development plan

The project will be built incrementally: establish the repository, add and validate the recognition pipeline, then build the Streamlit interface and refine the application. Dependencies and implementation will be introduced in later steps.

Keep private photographs, identity data, credentials, model weights, and generated embeddings out of Git. See `.gitignore` for local exclusions.

## Private dataset

Put five real photographs of each person in that person's folder under `dataset/`:

```text
dataset/
├── Person_A/  # Add five photographs of Person A here
├── Person_B/  # Add five photographs of Person B here
├── Person_C/  # Add five photographs of Person C here
└── Person_D/  # Add five photographs of Person D here
```

The folder name is the identity label used by the project, so keep each person's photographs in the matching folder. Replace `Person_A` through `Person_D` with your chosen labels before adding photographs if desired. Keep photographs and identifying information private on your computer; the dataset files are excluded from Git. The `.gitkeep` files only preserve the empty folder structure in the repository.
