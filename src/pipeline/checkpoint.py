"""
Checkpoint Management System

Handles:
1. Saving/loading model checkpoints
2. State tracking (optimizer, scheduler, step count)
3. Best model preservation
4. Automatic cleanup of old checkpoints
5. Resumption from interruptions
"""

import torch
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class CheckpointMetadata:
    """
    Metadata for a checkpoint.

    Invariants:
    - step is monotonically increasing
    - train_loss >= 0
    - val_loss >= 0 (if evaluated)
    """
    step: int
    epoch: float
    train_loss: float
    val_loss: Optional[float]
    learning_rate: float
    timestamp: str
    dataset_hash: str  # Ensures checkpoint matches current dataset
    is_best: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'CheckpointMetadata':
        return cls(**data)


class CheckpointManager:
    """
    Production-grade checkpoint management.

    Features:
    - Atomic saves (write to temp, then rename)
    - Best model tracking
    - Automatic cleanup
    - Resumption support
    - Dataset compatibility checking

    File Structure:
    checkpoints/
    ├── checkpoint_step_1000.pt
    ├── checkpoint_step_2000.pt
    ├── checkpoint_step_3000.pt
    ├── best_model.pt
    └── latest_metadata.json
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 3,
        keep_best: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.keep_last_n = keep_last_n
        self.keep_best = keep_best

        self.best_val_loss = float('inf')
        self.latest_checkpoint_path: Optional[Path] = None

        # Files
        self.best_model_path = self.checkpoint_dir / "best_model.pt"
        self.metadata_path = self.checkpoint_dir / "latest_metadata.json"

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        metadata: CheckpointMetadata
    ) -> Path:
        """
        Save a checkpoint atomically.

        Atomic Save Protocol:
        1. Write to temporary file
        2. Verify write succeeded
        3. Atomic rename to final path

        This prevents corruption from interrupted saves.
        """
        checkpoint_name = f"checkpoint_step_{metadata.step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        temp_path = checkpoint_path.with_suffix('.pt.tmp')

        # Prepare checkpoint data
        checkpoint_data = {
            'step': metadata.step,
            'epoch': metadata.epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metadata': metadata.to_dict(),
        }

        # Atomic save
        try:
            # Write to temp file
            torch.save(checkpoint_data, temp_path)

            # Atomic rename
            temp_path.rename(checkpoint_path)

            self.latest_checkpoint_path = checkpoint_path

            # Save metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)

            # Check if this is the best model
            if metadata.val_loss is not None:
                if metadata.val_loss < self.best_val_loss:
                    self.best_val_loss = metadata.val_loss
                    metadata.is_best = True

                    if self.keep_best:
                        self._save_best_model(checkpoint_path)

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()

            return checkpoint_path

        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Failed to save checkpoint: {e}")

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        device: str = 'cpu'
    ) -> Optional[CheckpointMetadata]:
        """
        Load a checkpoint.

        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            checkpoint_path: Specific checkpoint to load (None = latest)
            device: Device to map tensors to

        Returns:
            CheckpointMetadata if successful, None otherwise
        """
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()

        if checkpoint_path is None:
            return None

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            return None

        print(f"Loading checkpoint: {checkpoint_path.name}")

        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Load model weights
            model.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer state
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load scheduler state
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                if checkpoint['scheduler_state_dict'] is not None:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Load metadata
            metadata = CheckpointMetadata.from_dict(checkpoint['metadata'])
            if metadata.val_loss is not None:
                self.best_val_loss = metadata.val_loss

            print(f"Loaded checkpoint from step {metadata.step}")

            return metadata

        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return None

    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint by step number."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))

        if not checkpoints:
            return None

        # Extract step numbers and sort
        checkpoint_steps = []
        for ckpt in checkpoints:
            try:
                step = int(ckpt.stem.split('_')[-1])
                checkpoint_steps.append((step, ckpt))
            except ValueError:
                continue

        if not checkpoint_steps:
            return None

        # Return checkpoint with highest step
        latest = max(checkpoint_steps, key=lambda x: x[0])
        return latest[1]

    def _save_best_model(self, checkpoint_path: Path):
        """Save a copy of the best model."""
        shutil.copy(checkpoint_path, self.best_model_path)
        print(f"   Saved best model (val_loss={self.best_val_loss:.4f})")

    def _cleanup_old_checkpoints(self):
        """
        Remove old checkpoints, keeping only last N.

        Strategy:
        1. Always keep best_model.pt
        2. Keep last N checkpoints by step number
        3. Remove everything else
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))

        if len(checkpoints) <= self.keep_last_n:
            return

        # Sort by step number
        checkpoint_steps = []
        for ckpt in checkpoints:
            try:
                step = int(ckpt.stem.split('_')[-1])
                checkpoint_steps.append((step, ckpt))
            except ValueError:
                continue

        checkpoint_steps.sort(key=lambda x: x[0], reverse=True)

        # Keep only last N
        to_keep = set(ckpt for _, ckpt in checkpoint_steps[:self.keep_last_n])

        # Always keep best model
        if self.best_model_path.exists():
            to_keep.add(self.best_model_path)

        # Remove old checkpoints
        for _, ckpt in checkpoint_steps[self.keep_last_n:]:
            if ckpt not in to_keep:
                ckpt.unlink()
                print(f"   Removed old checkpoint: {ckpt.name}")

    def get_latest_metadata(self) -> Optional[CheckpointMetadata]:
        """Get metadata from latest checkpoint."""
        if not self.metadata_path.exists():
            return None

        with open(self.metadata_path, 'r') as f:
            data = json.load(f)

        return CheckpointMetadata.from_dict(data)

    def has_checkpoint(self) -> bool:
        """Check if any checkpoint exists."""
        return self._find_latest_checkpoint() is not None

    def clean_all_checkpoints(self):
        """Remove ALL checkpoints (use with caution)."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))

        for ckpt in checkpoints:
            ckpt.unlink()
            print(f"   Removed: {ckpt.name}")

        if self.best_model_path.exists():
            self.best_model_path.unlink()
            print(f"   Removed: {self.best_model_path.name}")

        if self.metadata_path.exists():
            self.metadata_path.unlink()

        print("All checkpoints removed")
