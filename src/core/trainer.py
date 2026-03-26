"""
Production Training Pipeline

Implements:
1. Training loop with gradient accumulation
2. Mixed precision training (AMP)
3. Learning rate scheduling
4. Automatic checkpointing
5. Evaluation loops
6. TensorBoard logging
7. Resumption from interruptions
8. Model architecture selection (base/enhanced)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path

# Optional tensorboard import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False
from typing import Optional, Dict
import time
from tqdm import tqdm

from .models import get_model, ModelConfig, ModelVersionManager
from .dataset import create_dataloader, InfiniteDataLoader
from ..pipeline.checkpoint import CheckpointManager, CheckpointMetadata


class CosineWarmupScheduler:
    """
    Learning rate scheduler with linear warmup and cosine decay.

    Schedule:
    - Warmup: lr(t) = lr_max x (t / T_warmup) for t <= T_warmup
    - Cosine: lr(t) = lr_min + 0.5(lr_max - lr_min)(1 + cos(pi(t-T_warmup)/(T_max-T_warmup)))
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 1e-5
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0

    def step(self):
        """Update learning rate."""
        self.current_step += 1
        lr = self._get_lr()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self) -> float:
        """Calculate current learning rate."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay

    def state_dict(self) -> dict:
        return {'current_step': self.current_step}

    def load_state_dict(self, state_dict: dict):
        self.current_step = state_dict['current_step']


class Trainer:
    """
    Production-grade training pipeline.

    Features:
    - Model architecture selection (base/enhanced)
    - Automatic mixed precision (AMP)
    - Gradient clipping
    - Checkpoint management
    - Model versioning
    - Resume from interruption
    - TensorBoard logging
    - Validation tracking
    """

    def __init__(self, config: dict, model_config: Optional[ModelConfig] = None):
        self.config = config
        self.model_config = model_config

        # Setup device
        self.device = self._setup_device()
        print(f"Using device: {self.device}")

        # Paths
        self.log_dir = Path(config['logging']['log_dir'])
        self.checkpoint_dir = Path(config['checkpoint']['save_dir'])

        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components (will be set in setup)
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[CosineWarmupScheduler] = None
        self.scaler: Optional[GradScaler] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.version_manager: Optional[ModelVersionManager] = None
        self.train_loader: Optional[InfiniteDataLoader] = None
        self.val_loader = None
        self.writer: Optional[SummaryWriter] = None

        # Training state
        self.current_step = 0
        self.current_epoch = 0.0
        self.best_val_loss = float('inf')
        self.initial_val_loss: Optional[float] = None

    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        device_name = self.config['hardware']['device']

        if device_name == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif device_name == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def setup(self, data_dir: str, dataset_name: str = None, dataset_hash: str = None):
        """
        Initialize all components.

        Args:
            data_dir: Directory containing processed data files
            dataset_name: Name of dataset (for transfer learning)
            dataset_hash: Hash of dataset config (for transfer learning)
        """
        data_dir = Path(data_dir)

        print("\n" + "=" * 70)
        print("TRAINING SETUP")
        print("=" * 70)

        # Create model config if not provided
        if self.model_config is None:
            enhanced_cfg = self.config['model'].get('enhanced', {})
            self.model_config = ModelConfig(
                architecture=self.config['model'].get('architecture', 'base'),
                model_name=self.config['model'].get('model_name', ''),
                parameter_count=self.config['model'].get('parameter_count', 0),
                architecture_family=self.config['model'].get('architecture_family', 'decoder-only-transformer'),
                vocab_size=self.config['model']['vocab_size'],
                d_model=self.config['model']['d_model'],
                num_heads=self.config['model']['num_heads'],
                num_layers=self.config['model']['num_layers'],
                d_ff=self.config['model']['d_ff'],
                max_seq_length=self.config['model']['max_seq_length'],
                dropout=self.config['model']['dropout'],
                use_rotary_embeddings=self.config['model'].get(
                    'use_rotary_embeddings',
                    enhanced_cfg.get('use_rotary_embeddings', True),
                ),
                use_flash_attention=self.config['model'].get(
                    'use_flash_attention',
                    enhanced_cfg.get('use_flash_attention', True),
                ),
                use_grouped_query_attention=self.config['model'].get(
                    'use_grouped_query_attention',
                    enhanced_cfg.get('use_grouped_query_attention', True),
                ),
                gqa_num_kv_heads=self.config['model'].get(
                    'gqa_num_kv_heads',
                    enhanced_cfg.get('gqa_num_kv_heads', 2),
                ),
                use_rms_norm=self.config['model'].get(
                    'use_rms_norm',
                    enhanced_cfg.get('use_rms_norm', True),
                ),
                use_swiglu=self.config['model'].get(
                    'use_swiglu',
                    enhanced_cfg.get('use_swiglu', True),
                ),
                gradient_checkpointing=self.config['model'].get(
                    'gradient_checkpointing',
                    enhanced_cfg.get('gradient_checkpointing', False),
                ),
                compile_model=self.config['model'].get('compile_model', False),
                version_models=self.config['model'].get('version_models', True),
            )

        # Create model
        print(f"\n[1/5] Creating {self.model_config.architecture} model...")
        self.model = get_model(self.model_config.architecture, self.model_config)
        self.model = self.model.to(self.device)

        num_params = self.model.count_parameters()
        print(f"   Model created: {num_params:,} parameters")
        print(f"   Architecture: {self.model_config.architecture}")

        # Create version manager
        if self.model_config.version_models:
            self.version_manager = ModelVersionManager(str(self.checkpoint_dir))

        # Create optimizer
        print("\n[2/5] Creating optimizer...")
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            betas=(self.config['training']['beta1'], self.config['training']['beta2']),
            weight_decay=self.config['training']['weight_decay']
        )
        print("   AdamW optimizer created")

        # Create scheduler
        print("\n[3/5] Creating scheduler...")
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=self.config['training']['warmup_steps'],
            max_steps=self.config['training']['max_steps']
        )
        print("   Cosine warmup scheduler created")

        # Create data loaders
        print("\n[4/5] Creating data loaders...")
        meta_path = str(data_dir / "meta.pkl")

        dataloader_config = self.config.get('data_loading', {})
        train_dataloader = create_dataloader(
            data_path=str(data_dir / "train.bin"),
            meta_path=meta_path,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory'],
            prefetch_factor=dataloader_config.get('prefetch_factor'),
            persistent_workers=dataloader_config.get('persistent_workers', False),
        )
        self.train_loader = InfiniteDataLoader(train_dataloader)

        val_dataloader = create_dataloader(
            data_path=str(data_dir / "val.bin"),
            meta_path=meta_path,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory'],
            prefetch_factor=dataloader_config.get('prefetch_factor'),
            persistent_workers=dataloader_config.get('persistent_workers', False),
        )
        self.val_loader = val_dataloader

        print("   Data loaders created")

        # Setup mixed precision
        if self.config['hardware']['mixed_precision'] and self.device.type == 'cuda':
            self.scaler = GradScaler()
            print("   Mixed precision enabled")
        else:
            self.scaler = None

        # Create checkpoint manager
        print("\n[5/5] Creating checkpoint manager...")
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            keep_last_n=self.config['checkpoint']['keep_last_n'],
            keep_best=self.config['checkpoint']['keep_best']
        )
        print("   Checkpoint manager created")

        # Setup TensorBoard
        if self.config['logging']['tensorboard'] and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            print("   TensorBoard logging enabled")
        elif self.config['logging']['tensorboard'] and not TENSORBOARD_AVAILABLE:
            print("   TensorBoard requested but not installed")

        print("\n" + "=" * 70)
        print("SETUP COMPLETED")
        print("=" * 70)

        return True

    def resume_from_checkpoint(self) -> bool:
        """
        Attempt to resume from latest checkpoint.

        Returns:
            True if resumed, False otherwise
        """
        if not self.config['checkpoint']['resume_from_latest']:
            return False

        if not self.checkpoint_manager.has_checkpoint():
            print("No checkpoint found. Starting from scratch.")
            return False

        print("\nResuming from checkpoint...")

        metadata = self.checkpoint_manager.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=str(self.device)
        )

        if metadata is None:
            print("Failed to load checkpoint. Starting from scratch.")
            return False

        # Restore training state
        self.current_step = metadata.step
        self.current_epoch = metadata.epoch
        if metadata.val_loss is not None:
            self.best_val_loss = metadata.val_loss

        print(f"Resumed from step {self.current_step}")
        return True

    def train_step(self) -> float:
        """
        Perform a single training step.

        Returns:
            Loss value for this step
        """
        self.model.train()
        grad_clip = self.config['training']['grad_clip']

        # Get batch
        batch = next(self.train_loader)
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward pass
        if self.scaler is not None:
            with autocast():
                _, loss, _ = self.model(input_ids, targets=labels)
        else:
            _, loss, _ = self.model(input_ids, targets=labels)

        # Backward pass
        self.optimizer.zero_grad()

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()

        # Update scheduler
        self.scheduler.step()

        # Update step counter
        self.current_step += 1

        return loss.item()

    def train(self):
        """Main training loop."""
        max_steps = self.config['training']['max_steps']
        eval_interval = self.config['training']['eval_interval']
        save_interval = self.config['training']['save_interval']
        log_interval = self.config['training']['log_interval']
        grad_clip = self.config['training']['grad_clip']

        print("\n" + "=" * 70)
        print("TRAINING")
        print("=" * 70)
        print(f"Max steps: {max_steps:,}")
        print(f"Starting from step: {self.current_step:,}")
        print(f"Architecture: {self.model_config.architecture}")
        print("=" * 70 + "\n")

        self.model.train()

        # Training metrics
        running_loss = 0.0
        step_start_time = time.time()

        pbar = tqdm(
            range(self.current_step, max_steps),
            desc="Training",
            initial=self.current_step,
            total=max_steps
        )

        for step in pbar:
            # Get batch
            batch = next(self.train_loader)
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            if self.scaler is not None:
                with autocast():
                    _, loss, _ = self.model(input_ids, targets=labels)
            else:
                _, loss, _ = self.model(input_ids, targets=labels)

            # Backward pass
            self.optimizer.zero_grad()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()

            # Update scheduler
            self.scheduler.step()

            # Update metrics
            running_loss += loss.item()
            self.current_step = step + 1

            # Logging
            if (step + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                lr = self.optimizer.param_groups[0]['lr']
                elapsed = time.time() - step_start_time
                tokens_per_sec = (log_interval * self.config['training']['batch_size'] *
                                self.config['model']['max_seq_length']) / elapsed

                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{lr:.2e}',
                    'tok/s': f'{tokens_per_sec:.0f}'
                })

                if self.writer:
                    self.writer.add_scalar('train/loss', avg_loss, step + 1)
                    self.writer.add_scalar('train/lr', lr, step + 1)
                    self.writer.add_scalar('train/tokens_per_sec', tokens_per_sec, step + 1)

                running_loss = 0.0
                step_start_time = time.time()

            # Evaluation
            if (step + 1) % eval_interval == 0:
                val_loss = self.evaluate()

                # Track initial val loss
                if self.initial_val_loss is None:
                    self.initial_val_loss = val_loss

                print(f"\nStep {step + 1}: val_loss = {val_loss:.4f}")

                if self.writer:
                    self.writer.add_scalar('val/loss', val_loss, step + 1)

                self.model.train()

            # Checkpointing
            if (step + 1) % save_interval == 0:
                self._save_checkpoint(step + 1, loss.item(), None)

        # Final evaluation and checkpoint
        print("\n" + "=" * 70)
        print("FINAL EVALUATION")
        print("=" * 70)

        val_loss = self.evaluate()
        print(f"Final val_loss = {val_loss:.4f}")

        self._save_checkpoint(max_steps, 0.0, val_loss)

        # Save versioned model if enabled
        if self.version_manager:
            print("\nSaving versioned model...")
            path, version = self.version_manager.save_versioned_model(
                model=self.model,
                optimizer=self.optimizer,
                config=self.model_config,
                metrics={'val_loss': val_loss, 'steps': max_steps}
            )
            print(f"   Saved as version {version}: {path}")

        # Cleanup
        self._cleanup_after_training()

        print("\n" + "=" * 70)
        print("TRAINING COMPLETED")
        print("=" * 70)

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate model on validation set.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            if self.scaler is not None:
                with autocast():
                    _, loss, _ = self.model(input_ids, targets=labels)
            else:
                _, loss, _ = self.model(input_ids, targets=labels)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def _save_checkpoint(self, step: int, train_loss: float, val_loss: Optional[float]):
        """Save checkpoint."""
        import hashlib

        # Create metadata
        metadata = CheckpointMetadata(
            step=step,
            epoch=self.current_epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=self.optimizer.param_groups[0]['lr'],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            dataset_hash=hashlib.sha256(str(self.config['dataset']).encode()).hexdigest()[:16]
        )

        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metadata=metadata
        )

    def _cleanup_after_training(self):
        """Cleanup after successful training completion."""
        print("\nPerforming post-training cleanup...")

        # Keep only best model
        if self.config['checkpoint']['keep_best'] and self.checkpoint_manager.best_model_path.exists():
            print("   Keeping best model, removing intermediate checkpoints...")

            final_model_path = self.checkpoint_dir / "final_model.pt"
            import shutil
            shutil.copy(
                self.checkpoint_manager.best_model_path,
                final_model_path
            )
            print(f"   Final model saved: {final_model_path}")

            for checkpoint_path in self.checkpoint_dir.glob("checkpoint_step_*.pt"):
                checkpoint_path.unlink()

            if self.checkpoint_manager.metadata_path.exists():
                self.checkpoint_manager.metadata_path.unlink()

        # Close TensorBoard writer
        if self.writer:
            self.writer.close()

        print("   Cleanup completed")
