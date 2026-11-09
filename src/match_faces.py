"""Compare a query FaceNet embedding with a local identity gallery."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDING_DIR = PROJECT_ROOT / "artifacts" / "embeddings"
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "reports" / "evaluation_split_proposal.csv"
EMBEDDING_DIMENSIONS = 512
DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class ReferenceEmbedding:
    """One stored vector, its identity label, and its source crop path."""

    identity: str
    relative_path: str
    filename: str
    vector: np.ndarray


def _as_vector(value: np.ndarray, *, dimensions: int | None = None) -> np.ndarray:
    """Validate a vector without renormalizing vectors that are already unit length."""
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("embedding must contain numeric values") from error
    if vector.ndim != 1:
        raise ValueError("embedding must be a one-dimensional vector")
    if dimensions is not None and vector.size != dimensions:
        raise ValueError(f"embedding must contain {dimensions} values; got {vector.size}")
    if vector.size == 0:
        raise ValueError("embedding cannot be empty")
    if not np.isfinite(vector).all():
        raise ValueError("embedding contains NaN or infinite values")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("embedding must have a finite, non-zero length")
    # FaceNet already returns unit vectors. Leave them as-is; normalize only other inputs.
    if np.isclose(norm, 1.0, rtol=0.0, atol=1e-6):
        return vector
    return vector / norm


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Return cosine similarity in [-1, 1], rejecting invalid vectors."""
    a = _as_vector(vector_a)
    b = _as_vector(vector_b)
    if a.shape != b.shape:
        raise ValueError(f"vectors must have the same shape; got {a.shape} and {b.shape}")

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    dot = float(np.dot(a, b))
    # Unit FaceNet vectors have norm 1, so their dot product is already cosine similarity.
    if np.isclose(norm_a, 1.0, rtol=0.0, atol=1e-6) and np.isclose(
        norm_b, 1.0, rtol=0.0, atol=1e-6
    ):
        score = dot
    else:
        score = dot / (norm_a * norm_b)
    if not np.isfinite(score):
        raise ValueError("cosine similarity is not finite")
    if score < -1.0 - 1e-8 or score > 1.0 + 1e-8:
        raise ValueError(f"cosine similarity is outside its valid range: {score}")
    return float(np.clip(score, -1.0, 1.0))


def _relative_key(path: str | Path) -> str:
    """Normalize path separators and case for reliable metadata matching on Windows."""
    return os.path.normcase(str(path).replace("\\", "/")).casefold()


def _absolute_key(path: str | Path, project_root: Path = PROJECT_ROOT) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return os.path.normcase(str(candidate.resolve())).casefold()


def discover_latest_bundle(embedding_dir: Path = DEFAULT_EMBEDDING_DIR) -> tuple[Path, Path]:
    """Find the newest NPZ whose same-stem JSON metadata file also exists."""
    bundles = [
        (path, path.with_suffix(".json"))
        for path in embedding_dir.glob("facenet_vggface2_*.npz")
        if path.with_suffix(".json").is_file()
    ]
    if not bundles:
        raise FileNotFoundError(f"No matching FaceNet NPZ/JSON bundle found in {embedding_dir}")
    return max(bundles, key=lambda pair: pair[0].name)


def load_embedding_bundle(npz_path: Path, metadata_path: Path) -> tuple[np.ndarray, list[dict]]:
    """Load and validate the Day 6 NPZ matrix and its JSON index."""
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            if "embeddings" not in archive.files:
                raise ValueError("NPZ must contain an 'embeddings' array")
            matrix = np.asarray(archive["embeddings"])
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not load embeddings from {npz_path}: {error}") from error
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load metadata from {metadata_path}: {error}") from error

    if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Expected an N x {EMBEDDING_DIMENSIONS} matrix; got shape {matrix.shape}"
        )
    if not np.issubdtype(matrix.dtype, np.number) or not np.isfinite(matrix).all():
        raise ValueError("embedding matrix must contain only finite numeric values")
    row_norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if not np.allclose(row_norms, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("Day 6 embedding rows must already be L2-normalized")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("items"), list):
        raise ValueError("metadata must contain an 'items' list")
    items = metadata["items"]
    if len(items) != matrix.shape[0] or metadata.get("count") != matrix.shape[0]:
        raise ValueError("NPZ row count does not match metadata count/items")
    if metadata.get("embedding_dimensions") != EMBEDDING_DIMENSIONS:
        raise ValueError("metadata embedding_dimensions does not match the 512-value vectors")

    indexes = []
    seen_paths: dict[str, str] = {}
    for row in items:
        if not isinstance(row, dict):
            raise ValueError("each metadata item must be an object")
        identity = row.get("identity")
        relative_path = row.get("relative_path")
        index = row.get("embedding_index")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("metadata item is missing a valid identity label")
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError("metadata item is missing a valid relative_path")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(items):
            raise ValueError("metadata item has an invalid embedding_index")
        if row.get("dimensions") != EMBEDDING_DIMENSIONS:
            raise ValueError("metadata item does not declare a 512-value embedding")
        if row.get("l2_normalized") is not True:
            raise ValueError("metadata item is not marked as L2-normalized")
        parts = PurePosixPath(relative_path.replace("\\", "/")).parts
        if len(parts) != 4 or parts[:2] != ("dataset", "processed") or parts[2] != identity:
            raise ValueError(f"identity label does not match processed path: {relative_path}")
        if row.get("filename") != parts[-1]:
            raise ValueError(f"metadata filename does not match its relative_path: {relative_path}")
        if not (PROJECT_ROOT / Path(*parts)).is_file():
            raise ValueError(f"metadata points to a missing processed crop: {relative_path}")
        path_key = _relative_key(relative_path)
        prior_identity = seen_paths.setdefault(path_key, identity)
        if prior_identity != identity:
            raise ValueError(f"one source path has conflicting identity labels: {relative_path}")
        indexes.append(index)
    if sorted(indexes) != list(range(matrix.shape[0])):
        raise ValueError("metadata embedding_index values must cover each matrix row exactly once")
    if "model" not in metadata or "normalization" not in metadata:
        raise ValueError("metadata is missing the model or normalization description")
    return matrix, items


def _build_references(
    matrix: np.ndarray,
    items: list[dict],
    *,
    include_paths: set[str] | None = None,
    exclude_relative_path: str | None = None,
) -> tuple[list[ReferenceEmbedding], int]:
    """Build the gallery and drop duplicate paths/vectors within the same identity."""
    if matrix.ndim != 2 or matrix.shape[0] != len(items):
        raise ValueError("embedding rows and metadata items do not align")
    include_keys = {_relative_key(path) for path in include_paths} if include_paths is not None else None
    excluded_key = _relative_key(exclude_relative_path) if exclude_relative_path else None
    references: list[ReferenceEmbedding] = []
    seen_paths: dict[tuple[str, str], bytes] = {}
    seen_vectors: set[tuple[str, bytes]] = set()
    duplicates = 0

    for row_number, item in enumerate(items):
        row_index = item.get("embedding_index", row_number)
        if not isinstance(row_index, int) or not 0 <= row_index < matrix.shape[0]:
            raise ValueError("metadata contains an invalid embedding_index")
        identity = item.get("identity")
        relative_path = item.get("relative_path")
        if not isinstance(identity, str) or not isinstance(relative_path, str):
            raise ValueError("metadata identity and relative_path must be strings")
        path_key = _relative_key(relative_path)
        if include_keys is not None and path_key not in include_keys:
            continue
        if excluded_key is not None and path_key == excluded_key:
            continue

        vector = np.asarray(matrix[row_index])
        # Validate each stored vector but leave Day 6 unit vectors unchanged.
        _as_vector(vector, dimensions=EMBEDDING_DIMENSIONS)
        vector_bytes = np.asarray(vector, dtype=np.float32).tobytes()
        path_identity = (identity, path_key)
        prior_vector = seen_paths.get(path_identity)
        if prior_vector is not None:
            if prior_vector != vector_bytes:
                raise ValueError(f"duplicate metadata path has different vectors: {relative_path}")
            duplicates += 1
            continue
        seen_paths[path_identity] = vector_bytes

        vector_identity = (identity, vector_bytes)
        if vector_identity in seen_vectors:
            duplicates += 1
            continue
        seen_vectors.add(vector_identity)
        references.append(
            ReferenceEmbedding(
                identity=identity,
                relative_path=relative_path,
                filename=str(item.get("filename") or Path(relative_path).name),
                vector=vector,
            )
        )
    return references, duplicates


def match_query_embedding(
    query_embedding: np.ndarray,
    embedding_matrix: np.ndarray,
    metadata_items: list[dict],
    *,
    top_k: int = DEFAULT_TOP_K,
    include_paths: set[str] | None = None,
    exclude_relative_path: str | None = None,
) -> dict:
    """Rank identity candidates using the mean of each identity's top-k references."""
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    query = _as_vector(query_embedding, dimensions=EMBEDDING_DIMENSIONS)
    references, duplicate_count = _build_references(
        embedding_matrix,
        metadata_items,
        include_paths=include_paths,
        exclude_relative_path=exclude_relative_path,
    )
    if not references:
        raise ValueError("no reference embeddings remain after filtering")

    by_identity: dict[str, list[tuple[float, ReferenceEmbedding]]] = defaultdict(list)
    for reference in references:
        score = cosine_similarity(query, reference.vector)
        by_identity[reference.identity].append((score, reference))

    ranked = []
    for identity, scored_references in by_identity.items():
        scored_references.sort(key=lambda pair: (-pair[0], pair[1].relative_path.casefold()))
        used = scored_references[: min(top_k, len(scored_references))]
        aggregate = float(np.mean([score for score, _ in used]))
        if not np.isfinite(aggregate):
            raise ValueError(f"aggregate score is not finite for {identity}")
        ranked.append(
            {
                "identity": identity,
                "aggregate_similarity": float(np.clip(aggregate, -1.0, 1.0)),
                "aggregation": f"mean of the top {len(used)} of {len(scored_references)} reference scores",
                "references_used": len(used),
                "individual_scores": [
                    {
                        "filename": reference.filename,
                        "relative_path": reference.relative_path,
                        "similarity": score,
                    }
                    for score, reference in scored_references
                ],
            }
        )
    ranked.sort(key=lambda result: (-result["aggregate_similarity"], result["identity"].casefold()))
    return {
        "best_candidate": ranked[0]["identity"],
        "ranked_candidates": ranked,
        "top_k": top_k,
        "deduplicated_reference_count": duplicate_count,
        "decision": "candidate only; no acceptance threshold or unknown-person decision is applied",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_known_split(
    embedding_matrix: np.ndarray,
    metadata_items: list[dict],
    split_path: Path = DEFAULT_SPLIT_PATH,
    *,
    top_k: int = DEFAULT_TOP_K,
    project_root: Path = PROJECT_ROOT,
) -> dict:
    """Evaluate held-out photos using only enrollment references after leakage checks."""
    try:
        with split_path.open(encoding="utf-8-sig", newline="") as split_file:
            rows = list(csv.DictReader(split_file))
    except (OSError, csv.Error) as error:
        raise ValueError(f"Could not read evaluation split {split_path}: {error}") from error
    if not rows:
        raise ValueError("evaluation split is empty")

    source_roles: dict[str, set[str]] = defaultdict(set)
    hash_roles: dict[str, set[str]] = defaultdict(set)
    group_roles: dict[str, set[str]] = defaultdict(set)
    metadata_by_output: dict[str, dict] = {}
    for item in metadata_items:
        key = _absolute_key(item["relative_path"], project_root)
        if key in metadata_by_output:
            raise ValueError(f"duplicate processed path in metadata: {item['relative_path']}")
        metadata_by_output[key] = item

    enrollment_rows = []
    query_rows = []
    for row in rows:
        identity = row.get("identity", "").strip()
        role = row.get("proposed_role", "").strip()
        source_text = row.get("original_local_path", "").strip()
        output_text = row.get("processed_local_path", "").strip()
        group = row.get("near_duplicate_group", "").strip()
        if role not in {"enrollment", "known_test", "unused_reserve"}:
            raise ValueError(f"unexpected split role: {role!r}")
        if not identity or not source_text or not output_text or not group:
            raise ValueError("evaluation split has a row with missing identity/path/group fields")
        group_roles[group].add(role)
        source = Path(source_text)
        output_key = _absolute_key(output_text, project_root)
        source_key = _absolute_key(source, project_root)
        if not source.is_file():
            raise ValueError(f"split source photo is missing: {source}")
        if output_key not in metadata_by_output:
            raise ValueError(f"split crop has no matching stored embedding: {output_text}")
        item = metadata_by_output[output_key]
        if item["identity"] != identity:
            raise ValueError(f"split identity does not match embedding metadata: {output_text}")
        if Path(source).parent.name.casefold() != identity.casefold():
            raise ValueError(f"source folder label does not match split identity: {source}")
        if role in {"enrollment", "known_test"}:
            source_roles[source_key].add(role)
            hash_roles[_sha256_file(source)].add(role)
            enrollment_row = {**row, "_source_key": source_key, "_output_key": output_key}
            if role == "enrollment":
                enrollment_rows.append(enrollment_row)
            else:
                query_rows.append(enrollment_row)

    if not enrollment_rows or not query_rows:
        raise ValueError("split needs both enrollment and known_test images")
    if any(len(group_role_set) > 1 for group_role_set in group_roles.values()):
        raise ValueError("a near-duplicate group crosses enrollment/test/reserve roles")
    if any(len(source_role_set) > 1 for source_role_set in source_roles.values()):
        raise ValueError("an original photo appears in both enrollment and test")
    if any(len(hash_role_set) > 1 for hash_role_set in hash_roles.values()):
        raise ValueError("byte-identical original photos cross enrollment and test")

    enrollment_paths = {
        metadata_by_output[row["_output_key"]]["relative_path"] for row in enrollment_rows
    }
    per_identity_enrollment = dict(Counter(row["identity"] for row in enrollment_rows))
    per_identity_queries = dict(Counter(row["identity"] for row in query_rows))
    if set(per_identity_enrollment) != set(per_identity_queries):
        raise ValueError("every evaluated identity must have enrollment and test photos")

    evaluations = []
    for row in sorted(query_rows, key=lambda entry: (entry["identity"].casefold(), entry["original_local_path"].casefold())):
        item = metadata_by_output[row["_output_key"]]
        query_index = item["embedding_index"]
        result = match_query_embedding(
            embedding_matrix[query_index],
            embedding_matrix,
            metadata_items,
            top_k=top_k,
            include_paths=enrollment_paths,
            exclude_relative_path=item["relative_path"],
        )
        ranked_identities = [candidate["identity"] for candidate in result["ranked_candidates"]]
        rank = ranked_identities.index(row["identity"]) + 1
        evaluations.append(
            {
                "actual_identity": row["identity"],
                "query_filename": Path(row["original_local_path"]).name,
                "predicted_candidate": result["best_candidate"],
                "actual_identity_rank": rank,
                "correct_top1": rank == 1,
                "ranked_candidates": result["ranked_candidates"],
            }
        )

    correct = sum(record["correct_top1"] for record in evaluations)
    return {
        "query_count": len(evaluations),
        "correct_top1": correct,
        "top1_rate": correct / len(evaluations),
        "per_identity_enrollment": per_identity_enrollment,
        "per_identity_queries": per_identity_queries,
        "near_duplicate_group_leakage": False,
        "exact_photo_leakage": False,
        "unused_reserve_count": sum(row.get("proposed_role") == "unused_reserve" for row in rows),
        "examples": evaluations,
    }


def embed_query_image(image_path: Path, *, model=None, device=None) -> np.ndarray:
    """Embed an already-cropped 160x160 face using the existing Day 6 pipeline."""
    import cv2
    import torch

    # Importing these helpers reuses the exact model loader and RGB/[-1, 1] transform from Day 6.
    try:
        from extract_embeddings import get_weights_path, load_model, make_tensor
    except ModuleNotFoundError as error:
        if error.name != "extract_embeddings":
            raise
        from src.extract_embeddings import get_weights_path, load_model, make_tensor

    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read query image: {image_path}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        weights_path = get_weights_path(torch)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"FaceNet weights are missing at {weights_path}; no automatic download is attempted."
            )
        model = load_model(torch, weights_path, device)
    tensor = make_tensor(torch, image).to(device)
    with torch.no_grad():
        vector = model(tensor).squeeze(0).detach().cpu().numpy()
    return _as_vector(vector, dimensions=EMBEDDING_DIMENSIONS)


def _print_evaluation(report: dict) -> None:
    print(f"Held-out known-person queries: {report['query_count']}")
    print(f"Top-1 candidate matches: {report['correct_top1']}/{report['query_count']} ({report['top1_rate']:.1%})")
    print(f"Enrollment photos by identity: {report['per_identity_enrollment']}")
    print(f"Test photos by identity: {report['per_identity_queries']}")
    print(f"Unused reserve rows: {report['unused_reserve_count']}")
    print("Leakage checks: no shared original paths, exact hashes, or near-duplicate groups across roles.")
    example = report["examples"][0]
    print(
        f"\nExample query: {example['query_filename']} (actual label: {example['actual_identity']})"
    )
    for rank, candidate in enumerate(example["ranked_candidates"], start=1):
        print(
            f"  {rank}. {candidate['identity']}: aggregate={candidate['aggregate_similarity']:.4f} "
            f"({candidate['aggregation']})"
        )
        scores = ", ".join(
            f"{Path(score['relative_path']).name}={score['similarity']:.4f}"
            for score in candidate["individual_scores"]
        )
        print(f"     individual reference scores: {scores}")
    print("These are candidate scores only; no threshold or Unknown Person decision is applied.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare FaceNet embeddings locally; the best identity is only a candidate."
    )
    parser.add_argument(
        "--query-image",
        type=Path,
        help="an already-cropped 160x160 query face; omit to run the held-out split evaluation",
    )
    parser.add_argument("--embeddings", type=Path, help="NPZ file; default is the newest local bundle")
    parser.add_argument("--metadata", type=Path, help="matching JSON file; default is the newest local bundle")
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH, help="held-out split CSV")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="references averaged per identity")
    args = parser.parse_args(argv)

    try:
        if bool(args.embeddings) != bool(args.metadata):
            raise ValueError("provide both --embeddings and --metadata, or neither")
        if args.embeddings and args.metadata:
            npz_path, metadata_path = args.embeddings, args.metadata
        else:
            npz_path, metadata_path = discover_latest_bundle()
        matrix, items = load_embedding_bundle(npz_path, metadata_path)
        print(f"Loaded {len(items)} embeddings from {npz_path.name}; model: InceptionResnetV1/VGGFace2")
        print(f"Embedding counts by identity: {dict(Counter(item['identity'] for item in items))}")

        if args.query_image:
            query_image = args.query_image.resolve()
            query = embed_query_image(query_image)
            try:
                exclude_path = query_image.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                exclude_path = query_image.as_posix()
            result = match_query_embedding(
                query,
                matrix,
                items,
                top_k=args.top_k,
                exclude_relative_path=exclude_path,
            )
            print("Candidate ranking (highest first):")
            for rank, candidate in enumerate(result["ranked_candidates"], start=1):
                print(
                    f"  {rank}. {candidate['identity']}: {candidate['aggregate_similarity']:.4f} "
                    f"({candidate['aggregation']})"
                )
            print(f"Best candidate: {result['best_candidate']} (not a final identity decision)")
            print("Query-image mode is a candidate demo. For evaluation, run without --query-image.")
        else:
            report = evaluate_known_split(matrix, items, args.split, top_k=args.top_k)
            _print_evaluation(report)
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
