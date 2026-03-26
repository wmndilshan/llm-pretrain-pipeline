"""
Pipeline Module

Data preprocessing and Modal GPU training:
- Local preprocessing (free)
- Modal GPU training with volume commits
- Checkpoint management
"""

from .preprocessing import (
    DataPreprocessor,
)

from .checkpoint import (
    CheckpointManager,
    CheckpointMetadata,
)

# Modal training (optional import)
try:
    from .modal_training import (
        ModalTrainingOrchestrator,
        train_on_modal_gpu,
        download_model_from_volume,
        list_checkpoints,
    )
except ImportError:
    # Modal not installed
    ModalTrainingOrchestrator = None
    train_on_modal_gpu = None
    download_model_from_volume = None
    list_checkpoints = None

__all__ = [
    # Preprocessing
    "DataPreprocessor",

    # Checkpoints
    "CheckpointManager",
    "CheckpointMetadata",

    # Modal training
    "ModalTrainingOrchestrator",
    "train_on_modal_gpu",
    "download_model_from_volume",
    "list_checkpoints",
]
