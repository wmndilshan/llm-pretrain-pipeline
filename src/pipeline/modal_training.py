"""
Modal GPU Training with Volume Commit Pattern

Enhanced training pipeline with:
- Proper volume commit after each checkpoint
- Integration with training results tracking
- Pre-training validation
- Budget tracking
- Progressive training support
- Auto-resume from last checkpoint
"""

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

# Load environment
try:
    from dotenv import load_dotenv
    import os
    _proj = Path(__file__).resolve().parent.parent.parent
    load_dotenv(_proj / ".env")
    load_dotenv(Path.cwd() / ".env")
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
except ImportError:
    pass

import modal
import numpy as np

from ..utils.config import load_yaml_config
from .checkpoint import CheckpointManager

# Modal app definition
app = modal.App("llm-training-v2")

# Image with all dependencies
MODAL_IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "regex>=2023.6.3",
        "tokenizers>=0.13.0",
        "transformers>=4.30.0",
        "datasets>=2.14.0",
        "python-dotenv>=1.0.0",
    )
    .add_local_dir(
        Path(__file__).parent.parent,
        remote_path="/root/app/src"
    )
)

# Persistent volumes
models_volume = modal.Volume.from_name("llm-models", create_if_missing=True)
data_volume = modal.Volume.from_name("llm-data", create_if_missing=True)


@app.function(
    image=MODAL_IMAGE,
    volumes={"/models": models_volume},
)
def download_model_from_volume(remote_path: str) -> bytes:
    """Download model file from Modal volume."""
    path = Path("/models") / remote_path.lstrip("/")
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return path.read_bytes()


@app.function(
    image=MODAL_IMAGE,
    volumes={"/models": models_volume},
)
def list_checkpoints() -> list:
    """List available checkpoints in volume."""
    checkpoint_dir = Path("/models/checkpoints")
    if not checkpoint_dir.exists():
        return []

    checkpoints = []
    for f in checkpoint_dir.glob("*.pt"):
        stat = f.stat()
        checkpoints.append({
            "name": f.name,
            "path": str(f),
            "size_mb": stat.st_size / (1024 * 1024),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })

    return sorted(checkpoints, key=lambda x: x["modified"], reverse=True)


def _run_training_job(
    config: Dict[str, Any],
    data_paths: Dict[str, str],
    previous_checkpoint: Optional[str] = None,
    phase_name: Optional[str] = None,
    checkpoint_every: int = 1000,
) -> Dict[str, Any]:
    """
    Train LLM on Modal GPU with proper volume commits.

    Features:
    - Commits volume after each checkpoint save
    - Returns structured training result
    - Supports resume from checkpoint
    - Phase-aware for progressive training

    Args:
        config: Training configuration
        data_paths: Paths to training data in volume
        previous_checkpoint: Path to load weights from
        phase_name: Phase name for progressive training
        checkpoint_every: Steps between checkpoints

    Returns:
        Dictionary with training results
    """
    import torch
    import time

    # Setup paths
    config = json.loads(json.dumps(config))
    config.setdefault("checkpoint", {})
    config["checkpoint"]["save_dir"] = "/models/checkpoints"
    config["checkpoint"]["save_every"] = checkpoint_every
    config.setdefault("logging", {})
    config["logging"]["log_dir"] = "/tmp/logs"
    requested_gpu = config.get("hardware", {}).get("modal_gpu", "A10G")

    # Make project importable
    sys.path.insert(0, "/root/app")

    result = {
        "status": "running",
        "phase": phase_name or "default",
        "started_at": datetime.now().isoformat(),
        "gpu": None,
        "gpu_tier": requested_gpu,
        "steps_completed": 0,
        "final_loss": 0.0,
        "best_loss": float('inf'),
        "final_val_loss": None,
        "checkpoint_path": "",
        "error": None,
    }

    try:
        # GPU info
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[INFO] GPU: {result['gpu']} ({gpu_memory_gb:.1f} GB)")

            # Optimize batch size
            config = _optimize_batch_size_for_gpu(config, gpu_memory_gb)
            print(f"[INFO] Batch size: {config['training']['batch_size']}")
        else:
            print("[WARN] No GPU available, using CPU")

        # Import trainer
        from src.core.trainer import Trainer

        print(f"[STEP] Training phase: {phase_name or 'default'}")
        trainer = Trainer(config)
        trainer.setup(data_dir="/data/processed")

        checkpoint_dir = Path("/models/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = checkpoint_dir / "best_model.pt"

        def build_checkpoint_payload(step: int, train_loss: float, val_loss: Optional[float]) -> Dict[str, Any]:
            return {
                "step": step,
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "scheduler_state_dict": trainer.scheduler.state_dict() if trainer.scheduler else None,
                "config": config,
                "phase": phase_name,
                "metadata": {
                    "step": step,
                    "epoch": trainer.current_epoch,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss) if val_loss is not None else None,
                    "learning_rate": float(trainer.optimizer.param_groups[0]["lr"]),
                    "phase": phase_name,
                    "dataset_name": config.get("dataset", {}).get("name"),
                    "dataset_hash": config.get("dataset", {}).get("processed_hash"),
                    "tokenizer_hash": config.get("dataset", {}).get("tokenizer_hash"),
                    "previous_checkpoint": previous_checkpoint,
                    "timestamp": datetime.now().isoformat(),
                },
            }

        def save_checkpoint(path: Path, step: int, train_loss: float, val_loss: Optional[float]) -> None:
            torch.save(build_checkpoint_payload(step, train_loss, val_loss), path)

        # Load previous checkpoint if specified
        if previous_checkpoint:
            checkpoint_path = Path(previous_checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found in Modal volume: {checkpoint_path}")

            print(f"[INFO] Loading checkpoint: {previous_checkpoint}")
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
            compatible, reason = CheckpointManager._state_dict_compatible(
                trainer.model,
                checkpoint.get("model_state_dict", {})
            )
            if not compatible:
                raise RuntimeError(
                    f"Resume checkpoint is incompatible with the current model architecture: {reason}"
                )
            trainer.model.load_state_dict(checkpoint["model_state_dict"])

            optimizer_state = checkpoint.get("optimizer_state_dict")
            if optimizer_state:
                trainer.optimizer.load_state_dict(optimizer_state)

            scheduler_state = checkpoint.get("scheduler_state_dict")
            if scheduler_state and trainer.scheduler:
                trainer.scheduler.load_state_dict(scheduler_state)

            metadata = checkpoint.get("metadata", {})
            trainer.current_step = int(checkpoint.get("step", metadata.get("step", 0)))
            if metadata.get("val_loss") is not None:
                trainer.best_val_loss = float(metadata["val_loss"])
            print(f"[OK] Checkpoint loaded at step {trainer.current_step}")
        else:
            # Resume from last checkpoint in the active Modal volume only when
            # we are not explicitly transferring from a previous run.
            trainer.resume_from_checkpoint()

        # Training with validation-based early stopping
        print("[STEP] Starting training with validation monitoring...")
        start_time = time.time()
        best_val_loss = trainer.best_val_loss if trainer.best_val_loss != float('inf') else float('inf')
        best_train_loss = float('inf')
        last_commit_step = 0
        early_stop = False
        last_eval_loss = None
        steps_completed = trainer.current_step
        avg_train_loss = 0.0
        stop_after_checkpoint = False

        # Early stopping settings
        patience = config.get("training", {}).get("patience", 5)  # Stop after N eval cycles without improvement
        patience_counter = 0
        eval_interval = config.get("training", {}).get("eval_interval", 500)
        min_improvement = 0.001  # Minimum improvement to reset patience

        # Running average for smoother loss tracking
        running_train_loss = 0.0
        loss_window = []
        window_size = 100

        # Overfitting detection
        overfitting_detected = False
        overfitting_count = 0

        # Custom training loop with volume commits
        max_steps = config["training"]["max_steps"]

        for step in range(trainer.current_step + 1, max_steps + 1):
            train_loss = trainer.train_step()
            steps_completed = step

            # Track running average
            loss_window.append(train_loss)
            if len(loss_window) > window_size:
                loss_window.pop(0)
            avg_train_loss = sum(loss_window) / len(loss_window)

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss

            # Validation and early stopping check
            if step % eval_interval == 0:
                # Run validation
                val_loss = trainer.evaluate()
                last_eval_loss = val_loss
                val_ppl = np.exp(min(val_loss, 20))
                train_ppl = np.exp(min(avg_train_loss, 20))

                # Check for improvement
                improvement = best_val_loss - val_loss
                if improvement > min_improvement:
                    best_val_loss = val_loss
                    trainer.best_val_loss = val_loss
                    patience_counter = 0
                    result["best_loss"] = val_loss
                    save_checkpoint(best_model_path, step, avg_train_loss, val_loss)
                    if step % checkpoint_every != 0:
                        models_volume.commit()
                        last_commit_step = step
                    print(f"[EVAL] Step {step}: val_loss={val_loss:.4f} (ppl={val_ppl:.2f}) "
                          f"train_loss={avg_train_loss:.4f} (ppl={train_ppl:.2f}) - NEW BEST!")
                else:
                    patience_counter += 1
                    print(f"[EVAL] Step {step}: val_loss={val_loss:.4f} (ppl={val_ppl:.2f}) "
                          f"train_loss={avg_train_loss:.4f} (ppl={train_ppl:.2f}) - patience {patience_counter}/{patience}")

                # Check for overfitting (train loss << val loss)
                if avg_train_loss < 0.01 and val_loss > avg_train_loss * 10:
                    print(f"[WARN] Overfitting detected: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")
                    patience_counter += 1  # Penalize overfitting

                # Early stopping
                if patience_counter >= patience:
                    print(f"\n[STOP] Early stopping triggered after {patience} eval cycles without improvement")
                    print(f"[STOP] Best validation loss: {best_val_loss:.4f}")
                    early_stop = True

            # Checkpoint and commit
            if step % checkpoint_every == 0 or step == max_steps or early_stop:
                checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
                save_checkpoint(checkpoint_path, step, avg_train_loss, last_eval_loss)

                if not best_model_path.exists():
                    save_checkpoint(best_model_path, step, avg_train_loss, last_eval_loss)

                # CRITICAL: Commit volume after checkpoint
                models_volume.commit()
                print(f"[CHECKPOINT] Step {step}: loss={avg_train_loss:.4f}, volume committed")
                last_commit_step = step
                if early_stop:
                    stop_after_checkpoint = True

            # Progress logging with perplexity
            if step % 100 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed
                eta = (max_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
                perplexity = np.exp(min(avg_train_loss, 20))  # Cap to avoid overflow
                print(f"[PROGRESS] Step {step}/{max_steps}: loss={avg_train_loss:.4f}, "
                      f"ppl={perplexity:.2f}, speed={steps_per_sec:.1f} steps/s, ETA={eta/60:.1f}min")

            # Early stopping detection (overfitting)
            if avg_train_loss < 1e-5 and step > 500:
                overfitting_count += 1
                if not overfitting_detected:
                    overfitting_detected = True
                    print(f"\n[WARN] Overfitting detected at step {step}!")
                    print(f"[WARN] Loss: {avg_train_loss:.8f} - Model is memorizing data.")
                    print(f"[WARN] Consider using more training data (current config: max_samples in config.yaml)")
                    result["warning"] = "Overfitting detected - loss dropped to near zero"

                # Stop after 100 consecutive overfitting steps
                if overfitting_count >= 100:
                    print(f"\n[STOP] Early stopping triggered - overfitting for 100 consecutive steps")
                    print(f"[STOP] Saving checkpoint and stopping training...")
                    early_stop = True
            else:
                overfitting_count = 0  # Reset counter if loss goes back up

            if stop_after_checkpoint:
                break

        # Final commit
        if not best_model_path.exists():
            save_checkpoint(best_model_path, steps_completed, avg_train_loss, last_eval_loss)
        if last_commit_step != steps_completed:
            models_volume.commit()

        # Training complete
        duration = time.time() - start_time
        result["status"] = "completed"
        result["completed_at"] = datetime.now().isoformat()
        result["steps_completed"] = steps_completed
        result["final_loss"] = avg_train_loss
        result["final_val_loss"] = last_eval_loss
        result["best_loss"] = best_val_loss if best_val_loss != float('inf') else avg_train_loss
        result["duration_hours"] = duration / 3600
        result["checkpoint_path"] = "/models/checkpoints/best_model.pt"

        print(f"[OK] Training complete: {steps_completed} steps in {duration/3600:.2f} hours")
        print(f"[OK] Final loss: {avg_train_loss:.4f}, Best loss: {result['best_loss']:.4f}")

    except Exception as e:
        import traceback
        result["status"] = "failed"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        result["completed_at"] = datetime.now().isoformat()
        print(f"[FAIL] Training failed: {e}")

        # Still commit to preserve any checkpoints
        try:
            models_volume.commit()
        except:
            pass

    return result


@app.function(
    image=MODAL_IMAGE,
    gpu="A10G",
    timeout=24 * 60 * 60,
    volumes={
        "/models": models_volume,
        "/data": data_volume,
    },
)
def train_on_modal_gpu(
    config: Dict[str, Any],
    data_paths: Dict[str, str],
    previous_checkpoint: Optional[str] = None,
    phase_name: Optional[str] = None,
    checkpoint_every: int = 1000,
) -> Dict[str, Any]:
    """Train on Modal using the A10G tier."""
    return _run_training_job(
        config=config,
        data_paths=data_paths,
        previous_checkpoint=previous_checkpoint,
        phase_name=phase_name,
        checkpoint_every=checkpoint_every,
    )


@app.function(
    image=MODAL_IMAGE,
    gpu="T4",
    timeout=24 * 60 * 60,
    volumes={
        "/models": models_volume,
        "/data": data_volume,
    },
)
def train_on_modal_t4(
    config: Dict[str, Any],
    data_paths: Dict[str, str],
    previous_checkpoint: Optional[str] = None,
    phase_name: Optional[str] = None,
    checkpoint_every: int = 1000,
) -> Dict[str, Any]:
    """Train on Modal using the T4 tier."""
    return _run_training_job(
        config=config,
        data_paths=data_paths,
        previous_checkpoint=previous_checkpoint,
        phase_name=phase_name,
        checkpoint_every=checkpoint_every,
    )


@app.function(
    image=MODAL_IMAGE,
    gpu="A100-40GB",
    timeout=24 * 60 * 60,
    volumes={
        "/models": models_volume,
        "/data": data_volume,
    },
)
def train_on_modal_a100_40gb(
    config: Dict[str, Any],
    data_paths: Dict[str, str],
    previous_checkpoint: Optional[str] = None,
    phase_name: Optional[str] = None,
    checkpoint_every: int = 1000,
) -> Dict[str, Any]:
    """Train on Modal using the A100 40GB tier."""
    return _run_training_job(
        config=config,
        data_paths=data_paths,
        previous_checkpoint=previous_checkpoint,
        phase_name=phase_name,
        checkpoint_every=checkpoint_every,
    )


@app.function(
    image=MODAL_IMAGE,
    gpu="A100-80GB",
    timeout=24 * 60 * 60,
    volumes={
        "/models": models_volume,
        "/data": data_volume,
    },
)
def train_on_modal_a100_80gb(
    config: Dict[str, Any],
    data_paths: Dict[str, str],
    previous_checkpoint: Optional[str] = None,
    phase_name: Optional[str] = None,
    checkpoint_every: int = 1000,
) -> Dict[str, Any]:
    """Train on Modal using the A100 80GB tier."""
    return _run_training_job(
        config=config,
        data_paths=data_paths,
        previous_checkpoint=previous_checkpoint,
        phase_name=phase_name,
        checkpoint_every=checkpoint_every,
    )


@app.function(
    image=MODAL_IMAGE,
    gpu="H100",
    timeout=24 * 60 * 60,
    volumes={
        "/models": models_volume,
        "/data": data_volume,
    },
)
def train_on_modal_h100(
    config: Dict[str, Any],
    data_paths: Dict[str, str],
    previous_checkpoint: Optional[str] = None,
    phase_name: Optional[str] = None,
    checkpoint_every: int = 1000,
) -> Dict[str, Any]:
    """Train on Modal using the H100 tier."""
    return _run_training_job(
        config=config,
        data_paths=data_paths,
        previous_checkpoint=previous_checkpoint,
        phase_name=phase_name,
        checkpoint_every=checkpoint_every,
    )


TRAINING_FUNCTIONS = {
    "T4": train_on_modal_t4,
    "A10G": train_on_modal_gpu,
    "A100-40GB": train_on_modal_a100_40gb,
    "A100-80GB": train_on_modal_a100_80gb,
    "H100": train_on_modal_h100,
}


def _optimize_batch_size_for_gpu(config: Dict[str, Any], gpu_memory_gb: float) -> Dict[str, Any]:
    """Optimize batch size based on GPU memory."""
    d_model = config["model"].get("d_model", 768)
    num_layers = config["model"].get("n_layers", config["model"].get("num_layers", 12))
    d_ff = config["model"].get("d_ff", d_model * 4)
    vocab_size = config["model"].get("vocab_size", 32000)
    seq_len = config["model"].get("max_seq_len", config["model"].get("max_seq_length", 1024))

    # Estimate memory requirements
    embedding_params = vocab_size * d_model * 2
    attention_params = num_layers * 4 * d_model * d_model
    ffn_params = num_layers * 2 * d_model * d_ff
    total_params = embedding_params + attention_params + ffn_params

    model_size_gb = (total_params * 4) / (1024**3)  # float32
    optimizer_size_gb = model_size_gb * 2  # Adam states
    activation_per_sample_gb = (num_layers * seq_len * d_model * 4 * 4) / (1024**3)

    available_gb = gpu_memory_gb * 0.8 - model_size_gb - optimizer_size_gb

    if available_gb > 0:
        optimal = int(available_gb / activation_per_sample_gb)
        optimal = max(1, min(optimal, 256))
        optimal = 2 ** int(np.log2(max(1, optimal)))
        config["training"]["batch_size"] = optimal

    return config


class ModalTrainingOrchestrator:
    """
    Orchestrates training with proper validation and result tracking.

    Features:
    - Pre-training validation
    - Budget checking
    - Local preprocessing
    - Modal GPU training
    - Volume commit pattern
    - Training result tracking
    """

    def __init__(
        self,
        config_path: str,
        budget_tracker=None,
        skip_validation: bool = False
    ):
        self.config_path = Path(config_path)
        self.config = load_yaml_config(config_path)

        self.budget_tracker = budget_tracker
        self.skip_validation = skip_validation
        self.training_history = []
        self.history_file = Path("training_history.json")
        self._load_history()

    def _load_history(self):
        """Load training history."""
        if self.history_file.exists():
            with open(self.history_file, "r") as f:
                self.training_history = json.load(f)

    def _save_history(self):
        """Save training history."""
        with open(self.history_file, "w") as f:
            json.dump(self.training_history, f, indent=2)

    def validate(self) -> Tuple[bool, Any]:
        """Run pre-training validation."""
        if self.skip_validation:
            return True, None

        try:
            from ..orchestration.validation import PreTrainingValidator

            validator = PreTrainingValidator(
                config=self.config,
                budget_tracker=self.budget_tracker,
                verbose=True
            )
            report = validator.validate_all()
            return report.can_proceed, report

        except ImportError:
            print("[WARN] Validation module not available, skipping")
            return True, None

    def preprocess_locally(self, dataset_name: str) -> Dict[str, str]:
        """Preprocess dataset locally (no GPU cost)."""
        try:
            from ..pipeline.preprocessing import DataPreprocessor
        except ImportError:
            from src.preprocessing import DataPreprocessor

        self.config["dataset"]["name"] = dataset_name

        root = self.config_path.resolve().parent.parent
        cache_dir = (root / self.config["dataset"]["cache_dir"].lstrip("./")).resolve()
        processed_dir = (root / self.config["dataset"]["processed_dir"].lstrip("./")).resolve()

        cache_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        preprocessor = DataPreprocessor(
            dataset_name=dataset_name,
            cache_dir=str(cache_dir),
            processed_dir=str(processed_dir),
            split_ratios=self.config["dataset"]["split_ratios"],
            max_seq_length=self.config["dataset"].get("max_seq_length", 1024),
            vocab_size=self.config["model"]["vocab_size"],
            max_samples=self.config.get("dataset", {}).get("max_samples"),
            tokenizer_backend=self.config["dataset"].get("tokenizer_backend", "hf"),
        )

        train_path, val_path, test_path = preprocessor.run(force_reprocess=False)
        parent = Path(train_path).parent
        preprocessing_state_path = parent / "preprocessing_state.json"
        if preprocessing_state_path.exists():
            with open(preprocessing_state_path, "r") as f:
                preprocessing_state = json.load(f)
            self.config["dataset"]["processed_hash"] = preprocessing_state.get("dataset_hash")
            self.config["dataset"]["tokenizer_hash"] = preprocessing_state.get("tokenizer_hash")

        return {
            "train": train_path,
            "val": val_path,
            "test": test_path,
            "meta": str(parent / "meta.pkl"),
            "tokenizer": str(parent / "tokenizer.json"),
            "preprocessing_state": str(preprocessing_state_path),
        }

    def upload_to_modal(self, data_paths: Dict[str, str]):
        """Upload preprocessed data to Modal volume."""
        print("[STEP] Uploading data to Modal volume...")

        with data_volume.batch_upload(force=True) as batch:
            for name, local_path in data_paths.items():
                if not Path(local_path).exists():
                    continue
                fname = Path(local_path).name
                remote = f"processed/{fname}"
                batch.put_file(local_path, remote)
                print(f"  Uploaded: {fname}")

        print("[OK] Upload complete")

    def upload_resume_checkpoint(self, checkpoint_path: str) -> str:
        """Upload a local checkpoint into the Modal models volume for resume/transfer."""
        if checkpoint_path.startswith("/models/"):
            return checkpoint_path

        local_checkpoint = Path(checkpoint_path)
        if not local_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {local_checkpoint}")

        remote_relpath = f"imports/{local_checkpoint.name}"
        print(f"[STEP] Uploading resume checkpoint: {local_checkpoint}")

        with models_volume.batch_upload(force=True) as batch:
            batch.put_file(str(local_checkpoint), remote_relpath)

        print(f"[OK] Resume checkpoint available at /models/{remote_relpath}")
        return f"/models/{remote_relpath}"

    def train_on_dataset(
        self,
        dataset_name: str,
        skip_validation: bool = False,
        previous_checkpoint: Optional[str] = None,
        phase_name: Optional[str] = None,
    ) -> str:
        """
        Train on a dataset with full pipeline.

        Args:
            dataset_name: HuggingFace dataset name
            skip_validation: Skip pre-training validation
            previous_checkpoint: Path to previous checkpoint
            phase_name: Phase name for progressive training

        Returns:
            Path to best model checkpoint
        """
        from ..orchestration.training_results import create_training_result, TrainingStatus

        print(f"\n{'='*60}")
        print(f"TRAINING: {dataset_name}")
        print(f"{'='*60}\n")

        # Create training result object
        result = create_training_result(
            config=self.config,
            run_name=f"{dataset_name.split('/')[-1]}_{datetime.now().strftime('%Y%m%d')}",
            phase=phase_name or "",
        )

        # Validation
        if not skip_validation:
            can_proceed, report = self.validate()
            if not can_proceed:
                print("[FAIL] Validation failed. Cannot proceed.")
                result.fail(Exception("Pre-training validation failed"))
                return ""

            if report and report.estimated_cost:
                result.cost.estimated_cost = report.estimated_cost

        # Cost estimation
        if self.budget_tracker:
            estimate = self.budget_tracker.estimate_cost(
                max_steps=self.config["training"]["max_steps"],
                gpu_type=self.config.get("hardware", {}).get("modal_gpu", "A10G"),
                model_params_millions=self.budget_tracker.estimate_model_params_millions_from_config(self.config),
            )
            print(f"[INFO] Estimated cost: ${estimate.estimated_cost:.2f}")
            result.cost.estimated_cost = estimate.estimated_cost
            result.cost.gpu_type = estimate.gpu_type

            if not estimate.within_safety_margin:
                print(f"[FAIL] Insufficient budget. Need ${estimate.estimated_cost:.2f}, "
                      f"have ${estimate.remaining_budget:.2f}")
                result.fail(Exception("Insufficient budget"))
                return ""

        # Local preprocessing
        print("\n[PHASE 1] Local preprocessing (no GPU cost)")
        try:
            data_paths = self.preprocess_locally(dataset_name)
        except Exception as e:
            print(f"[FAIL] Preprocessing failed: {e}")
            result.fail(e)
            return ""

        remote_previous_checkpoint = previous_checkpoint
        if previous_checkpoint:
            remote_previous_checkpoint = self.upload_resume_checkpoint(previous_checkpoint)

        # Upload to Modal
        print("\n[PHASE 2] Upload to Modal volume")
        self.upload_to_modal(data_paths)

        # GPU training
        print("\n[PHASE 3] Modal GPU training")
        result.start()
        requested_gpu = self.config.get("hardware", {}).get("modal_gpu", "A10G")
        train_function = TRAINING_FUNCTIONS.get(requested_gpu, train_on_modal_gpu)
        print(f"[INFO] Requested Modal GPU tier: {requested_gpu}")

        # Run Modal functions within app context
        with app.run():
            with modal.enable_output():
                train_result = train_function.remote(
                    config=self.config,
                    data_paths=data_paths,
                    previous_checkpoint=remote_previous_checkpoint,
                    phase_name=phase_name,
                )

            # Process result (still inside app context for download)
            if train_result["status"] == "completed":
                # Download model (inside app context)
                local_path = self._download_best_model(dataset_name, data_paths, train_result)
                result.complete(local_path)

                # Record in budget tracker
                if self.budget_tracker:
                    gpu_tier = train_result.get("gpu_tier", self.config.get("hardware", {}).get("modal_gpu", "A10G"))
                    actual_hours = train_result.get("duration_hours", 0)
                    actual_cost = 0.0
                    if gpu_tier in self.budget_tracker.GPU_SPECS:
                        actual_cost = round(
                            actual_hours * self.budget_tracker.GPU_SPECS[gpu_tier]["cost_per_hour"],
                            2,
                        )
                    result.cost.actual_cost = actual_cost
                    self.budget_tracker.record_spending(
                        dataset=dataset_name,
                        gpu_type=gpu_tier,
                        duration_hours=actual_hours,
                        cost_usd=actual_cost,
                        steps_trained=train_result["steps_completed"],
                        phase=phase_name
                    )

                # Update history
                self.training_history.append({
                    "dataset": dataset_name,
                    "best_model_path": local_path,
                    "phase": phase_name,
                    "timestamp": datetime.now().isoformat(),
                    "steps": train_result["steps_completed"],
                    "loss": train_result["final_loss"],
                    "val_loss": train_result.get("final_val_loss"),
                    "gpu_tier": train_result.get("gpu_tier"),
                })
                self._save_history()

                print(f"\n[OK] Training complete!")
                print(f"[OK] Best model: {local_path}")
                return local_path

            else:
                result.fail(Exception(train_result.get("error", "Unknown error")))
                print(f"\n[FAIL] Training failed: {train_result.get('error')}")
                return ""

    def _download_best_model(self, dataset_name: str, data_paths: Dict[str, str], train_result: Dict[str, Any]) -> str:
        """Download best model from Modal volume."""
        from ..orchestration.artifacts import export_inference_artifacts

        print("[STEP] Downloading best model...")

        remote_path = "checkpoints/best_model.pt"
        model_bytes = download_model_from_volume.remote(remote_path)
        manifest = {
            "phase": train_result.get("phase"),
            "steps_completed": train_result.get("steps_completed"),
            "best_loss": train_result.get("best_loss"),
            "final_loss": train_result.get("final_loss"),
            "final_val_loss": train_result.get("final_val_loss"),
            "gpu": train_result.get("gpu"),
            "gpu_tier": train_result.get("gpu_tier"),
            "exported_at": datetime.now().isoformat(),
        }
        local_path = export_inference_artifacts(
            dataset_name=dataset_name,
            checkpoint_bytes=model_bytes,
            tokenizer_path=data_paths["tokenizer"],
            meta_path=data_paths.get("meta"),
            manifest=manifest,
        )

        print(f"[OK] Saved to {local_path}")
        print("[OK] Updated Docker inference bundle at models/current")
        return str(local_path)


@app.local_entrypoint()
def main(
    dataset: str = "roneneldan/TinyStories",
    config: str = "configs/config.yaml",
    steps: int = 10000,
    skip_validation: bool = False,
    phase: str = "",
):
    """
    Train LLM on Modal GPU.

    Usage:
        modal run src/pipeline/modal_training.py
        modal run src/pipeline/modal_training.py --dataset roneneldan/TinyStories --steps 10000
    """
    root = Path(__file__).parent.parent.parent
    config_path = root / config if not Path(config).is_absolute() else Path(config)

    if not config_path.exists():
        print(f"[FAIL] Config not found: {config_path}")
        return

    # Initialize budget tracker
    try:
        from src.orchestration.budget_tracker import BudgetTracker
        budget_tracker = BudgetTracker()
    except ImportError:
        budget_tracker = None

    orchestrator = ModalTrainingOrchestrator(
        str(config_path),
        budget_tracker=budget_tracker,
        skip_validation=skip_validation
    )
    orchestrator.config["training"]["max_steps"] = steps
    orchestrator.train_on_dataset(dataset, phase_name=phase or None)


__all__ = [
    "app",
    "ModalTrainingOrchestrator",
    "train_on_modal_gpu",
    "download_model_from_volume",
    "list_checkpoints",
]
