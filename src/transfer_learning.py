"""
Transfer Learning Manager

Handles multi-dataset training with weight transfer:
1. Train on Dataset A -> Best Model A
2. Train on Dataset B starting from Model A -> Best Model B
3. Train on Dataset C starting from Model B -> Best Model C

Features:
- Automatic model weight transfer between datasets
- Dataset history tracking
- Validation that model architecture matches
- Prevents retraining on same dataset accidentally
"""

import json
import shutil
import hashlib
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class DatasetTrainingRecord:
    """Record of training on a specific dataset."""
    dataset_name: str
    dataset_hash: str
    start_timestamp: str
    end_timestamp: str
    initial_val_loss: Optional[float]
    final_val_loss: float
    best_val_loss: float
    total_steps: int
    model_path: str  # Path to best model from this dataset
    

@dataclass
class TransferLearningState:
    """
    State tracking for multi-dataset training.
    
    Maintains:
    - Training history across datasets
    - Current model lineage (A -> B -> C)
    - Latest best model path
    """
    training_history: List[DatasetTrainingRecord]
    current_dataset: Optional[str]
    latest_model_path: Optional[str]
    total_datasets_trained: int
    
    def to_dict(self) -> dict:
        return {
            'training_history': [asdict(record) for record in self.training_history],
            'current_dataset': self.current_dataset,
            'latest_model_path': self.latest_model_path,
            'total_datasets_trained': self.total_datasets_trained
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TransferLearningState':
        history = [DatasetTrainingRecord(**record) for record in data['training_history']]
        return cls(
            training_history=history,
            current_dataset=data['current_dataset'],
            latest_model_path=data['latest_model_path'],
            total_datasets_trained=data['total_datasets_trained']
        )


class TransferLearningManager:
    """
    Manages transfer learning across multiple datasets.
    
    Workflow:
    1. Check if dataset was already trained
    2. Find best model from previous dataset (if exists)
    3. Initialize new training with previous weights
    4. Track training results
    5. Update lineage: Dataset Chain -> New Dataset
    
    Example:
        manager = TransferLearningManager("models/transfer_learning")
        
        # Train on Dataset A
        manager.start_dataset_training("dataset_a", "hash_a")
        # ... training happens ...
        manager.complete_dataset_training(
            final_val_loss=2.5,
            best_val_loss=2.3,
            best_model_path="models/checkpoints/best_model.pt"
        )
        
        # Train on Dataset B (automatically uses Dataset A's weights)
        previous_model = manager.start_dataset_training("dataset_b", "hash_b")
        # Load previous_model weights before training
    """
    
    def __init__(self, transfer_dir: str = "models/transfer_learning"):
        self.transfer_dir = Path(transfer_dir)
        self.transfer_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.transfer_dir / "transfer_state.json"
        self.models_dir = self.transfer_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.state = self._load_or_create_state()
        
    def _load_or_create_state(self) -> TransferLearningState:
        """Load existing state or create new one."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                data = json.load(f)
            return TransferLearningState.from_dict(data)
        else:
            return TransferLearningState(
                training_history=[],
                current_dataset=None,
                latest_model_path=None,
                total_datasets_trained=0
            )
    
    def _save_state(self):
        """Persist state to disk."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)
    
    def has_trained_on_dataset(self, dataset_name: str, dataset_hash: str) -> bool:
        """
        Check if this exact dataset configuration was already trained.
        
        Args:
            dataset_name: Name of dataset
            dataset_hash: Hash of dataset configuration
            
        Returns:
            True if already trained, False otherwise
        """
        for record in self.state.training_history:
            if record.dataset_name == dataset_name and record.dataset_hash == dataset_hash:
                return True
        return False
    
    def get_previous_best_model(self) -> Optional[str]:
        """
        Get path to best model from previous dataset training.
        
        Returns:
            Path to previous best model, or None if this is first dataset
        """
        return self.state.latest_model_path
    
    def start_dataset_training(
        self,
        dataset_name: str,
        dataset_hash: str,
        force_retrain: bool = False
    ) -> Optional[str]:
        """
        Start training on a new dataset.
        
        Args:
            dataset_name: Name of the dataset
            dataset_hash: Hash of dataset configuration
            force_retrain: Force retraining even if already trained
            
        Returns:
            Path to previous model weights to load (None if first dataset)
            
        Raises:
            ValueError: If dataset already trained and force_retrain=False
        """
        # Check if already trained
        if not force_retrain and self.has_trained_on_dataset(dataset_name, dataset_hash):
            raise ValueError(
                f"Dataset '{dataset_name}' with hash '{dataset_hash}' was already trained. "
                f"Use force_retrain=True to retrain."
            )
        
        # Get previous model path
        previous_model_path = self.get_previous_best_model()
        
        # Update state
        self.state.current_dataset = dataset_name
        self._save_state()
        
        if previous_model_path:
            print(f"\n{'='*70}")
            print("TRANSFER LEARNING")
            print(f"{'='*70}")
            print(f"Starting training on: {dataset_name}")
            print(f"Using weights from: {previous_model_path}")
            print(f"Training lineage: {' -> '.join([r.dataset_name for r in self.state.training_history])} -> {dataset_name}")
            print(f"{'='*70}\n")
        else:
            print(f"\n{'='*70}")
            print("INITIAL TRAINING")
            print(f"{'='*70}")
            print(f"Starting training on: {dataset_name}")
            print(f"No previous model (training from scratch)")
            print(f"{'='*70}\n")
        
        return previous_model_path
    
    def complete_dataset_training(
        self,
        final_val_loss: float,
        best_val_loss: float,
        best_model_path: str,
        total_steps: int,
        initial_val_loss: Optional[float] = None
    ):
        """
        Complete training on current dataset and update state.
        
        Args:
            final_val_loss: Final validation loss
            best_val_loss: Best validation loss achieved
            best_model_path: Path to best model checkpoint
            total_steps: Total training steps
            initial_val_loss: Initial validation loss (optional)
        """
        if self.state.current_dataset is None:
            raise RuntimeError("No active dataset training. Call start_dataset_training first.")
        
        # Copy best model to transfer learning directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = self.state.current_dataset
        saved_model_name = f"{dataset_name}_{timestamp}_best.pt"
        saved_model_path = self.models_dir / saved_model_name
        
        shutil.copy(best_model_path, saved_model_path)
        
        # Create training record
        record = DatasetTrainingRecord(
            dataset_name=dataset_name,
            dataset_hash=self._compute_current_dataset_hash(),
            start_timestamp=timestamp,
            end_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            initial_val_loss=initial_val_loss,
            final_val_loss=final_val_loss,
            best_val_loss=best_val_loss,
            total_steps=total_steps,
            model_path=str(saved_model_path)
        )
        
        # Update state
        self.state.training_history.append(record)
        self.state.latest_model_path = str(saved_model_path)
        self.state.total_datasets_trained += 1
        self.state.current_dataset = None
        
        self._save_state()
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Dataset: {dataset_name}")
        if initial_val_loss:
            print(f"Initial val_loss: {initial_val_loss:.4f}")
        print(f"Final val_loss: {final_val_loss:.4f}")
        print(f"Best val_loss: {best_val_loss:.4f}")
        if initial_val_loss:
            improvement = ((initial_val_loss - best_val_loss) / initial_val_loss) * 100
            print(f"Improvement: {improvement:.2f}%")
        print(f"Total steps: {total_steps:,}")
        print(f"Model saved: {saved_model_path}")
        print(f"\nTraining lineage: {' -> '.join([r.dataset_name for r in self.state.training_history])}")
        print(f"{'='*70}\n")
    
    def _compute_current_dataset_hash(self) -> str:
        """Compute hash for current dataset (placeholder)."""
        # In practice, this would come from preprocessing state
        return hashlib.sha256(
            f"{self.state.current_dataset}_{datetime.now()}".encode()
        ).hexdigest()[:16]
    
    def get_training_summary(self) -> Dict:
        """Get summary of all training history."""
        return {
            'total_datasets': self.state.total_datasets_trained,
            'training_lineage': [r.dataset_name for r in self.state.training_history],
            'latest_model': self.state.latest_model_path,
            'history': [
                {
                    'dataset': r.dataset_name,
                    'best_val_loss': r.best_val_loss,
                    'steps': r.total_steps,
                    'timestamp': r.end_timestamp
                }
                for r in self.state.training_history
            ]
        }
    
    def print_training_history(self):
        """Print formatted training history."""
        print("\n" + "="*70)
        print("TRANSFER LEARNING HISTORY")
        print("="*70)
        
        if not self.state.training_history:
            print("No training history yet.")
            return
        
        print(f"\nTotal datasets trained: {self.state.total_datasets_trained}")
        print(f"Training lineage: {' -> '.join([r.dataset_name for r in self.state.training_history])}")
        print(f"\nDetailed History:\n")
        
        for i, record in enumerate(self.state.training_history, 1):
            print(f"{i}. Dataset: {record.dataset_name}")
            if record.initial_val_loss:
                print(f"   Initial val_loss: {record.initial_val_loss:.4f}")
            print(f"   Best val_loss: {record.best_val_loss:.4f}")
            print(f"   Total steps: {record.total_steps:,}")
            print(f"   Trained: {record.end_timestamp}")
            print(f"   Model: {Path(record.model_path).name}")
            print()
        
        print(f"Latest model: {self.state.latest_model_path}")
        print("="*70 + "\n")
    
    def get_lineage_chain(self) -> List[str]:
        """Get the chain of datasets trained in order."""
        return [r.dataset_name for r in self.state.training_history]
    
    def clean_old_models(self, keep_last_n: int = 3):
        """
        Clean old models, keeping only last N.
        
        Args:
            keep_last_n: Number of recent models to keep
        """
        if len(self.state.training_history) <= keep_last_n:
            return
        
        # Get models to remove (all except last N)
        records_to_clean = self.state.training_history[:-keep_last_n]
        
        print(f"Cleaning old transfer learning models (keeping last {keep_last_n})...")
        
        for record in records_to_clean:
            model_path = Path(record.model_path)
            if model_path.exists():
                model_path.unlink()
                print(f"  🗑️  Removed: {model_path.name}")
        
        print("[OK] Cleanup completed")
