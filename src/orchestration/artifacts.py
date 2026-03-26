"""
Artifact export helpers for the Docker-first inference contract.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional


def dataset_slug(dataset_name: str) -> str:
    """Create a filesystem-safe dataset identifier."""
    return dataset_name.replace("/", "_").replace("\\", "_")


def export_inference_artifacts(
    dataset_name: str,
    checkpoint_bytes: bytes,
    tokenizer_path: str,
    meta_path: Optional[str] = None,
    manifest: Optional[Dict[str, Any]] = None,
    trained_models_dir: str = "models/trained_models",
    current_dir: str = "models/current",
) -> str:
    """
    Export a trained checkpoint and its tokenizer to stable local locations.

    Two locations are maintained:
    - models/trained_models/<dataset_slug>/ for run history
    - models/current/ for Docker inference
    """
    slug = dataset_slug(dataset_name)
    dataset_dir = Path(trained_models_dir) / slug
    current_path = Path(current_dir)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    current_path.mkdir(parents=True, exist_ok=True)

    dataset_model_path = dataset_dir / "best_model.pt"
    dataset_model_path.write_bytes(checkpoint_bytes)

    dataset_tokenizer_path = dataset_dir / "tokenizer.json"
    shutil.copy2(tokenizer_path, dataset_tokenizer_path)

    current_model_path = current_path / "best_model.pt"
    shutil.copy2(dataset_model_path, current_model_path)

    current_tokenizer_path = current_path / "tokenizer.json"
    shutil.copy2(dataset_tokenizer_path, current_tokenizer_path)

    if meta_path and Path(meta_path).exists():
        dataset_meta_path = dataset_dir / "meta.pkl"
        shutil.copy2(meta_path, dataset_meta_path)
        shutil.copy2(dataset_meta_path, current_path / "meta.pkl")

    manifest_payload = {
        "dataset": dataset_name,
        "dataset_slug": slug,
        "model_path": str(dataset_model_path),
        "tokenizer_path": str(dataset_tokenizer_path),
        "current_model_path": str(current_model_path),
        "current_tokenizer_path": str(current_tokenizer_path),
    }
    if manifest:
        manifest_payload.update(manifest)

    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, indent=2),
        encoding="utf-8",
    )
    (current_path / "manifest.json").write_text(
        json.dumps(manifest_payload, indent=2),
        encoding="utf-8",
    )

    return str(dataset_model_path)

