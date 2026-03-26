"""
Structured Training Results

Comprehensive result objects for tracking training outcomes:
- TrainingMetrics: Loss, perplexity, learning rate history
- TrainingResult: Complete training run result
- TrainingSession: Multi-run session tracking
- Serialization to JSON for persistence
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum


class TrainingStatus(Enum):
    """Status of a training run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    # Loss tracking
    final_loss: float = 0.0
    best_loss: float = float('inf')
    initial_loss: float = 0.0
    loss_improvement: float = 0.0  # Percentage improvement

    # Perplexity
    final_perplexity: float = 0.0
    best_perplexity: float = float('inf')

    # Learning rate
    final_lr: float = 0.0
    peak_lr: float = 0.0

    # Gradient stats
    avg_grad_norm: float = 0.0
    max_grad_norm: float = 0.0
    gradient_overflow_count: int = 0

    # Training progress
    steps_completed: int = 0
    epochs_completed: float = 0.0
    tokens_processed: int = 0

    # Performance
    avg_step_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0

    # Checkpoints
    checkpoints_saved: int = 0
    best_checkpoint_step: int = 0

    # History (for plotting)
    loss_history: List[float] = field(default_factory=list)
    lr_history: List[float] = field(default_factory=list)
    perplexity_history: List[float] = field(default_factory=list)
    step_times: List[float] = field(default_factory=list)

    def update_from_step(
        self,
        step: int,
        loss: float,
        lr: float,
        step_time_ms: float,
        grad_norm: Optional[float] = None,
        batch_size: int = 32,
        seq_len: int = 1024
    ):
        """Update metrics from a training step."""
        import math

        # Loss
        if step == 1:
            self.initial_loss = loss
        self.final_loss = loss
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_checkpoint_step = step

        # Perplexity
        perplexity = math.exp(min(loss, 10))  # Cap to prevent overflow
        self.final_perplexity = perplexity
        if perplexity < self.best_perplexity:
            self.best_perplexity = perplexity

        # Learning rate
        self.final_lr = lr
        if lr > self.peak_lr:
            self.peak_lr = lr

        # Gradient
        if grad_norm is not None:
            self.max_grad_norm = max(self.max_grad_norm, grad_norm)
            # Running average
            n = self.steps_completed
            self.avg_grad_norm = (self.avg_grad_norm * n + grad_norm) / (n + 1)

        # Progress
        self.steps_completed = step
        self.tokens_processed += batch_size * seq_len

        # Performance
        n = len(self.step_times)
        self.avg_step_time_ms = (self.avg_step_time_ms * n + step_time_ms) / (n + 1)
        self.tokens_per_second = (batch_size * seq_len) / (step_time_ms / 1000) if step_time_ms > 0 else 0

        # History (sample every N steps to save memory)
        if step % 100 == 0 or step == 1:
            self.loss_history.append(loss)
            self.lr_history.append(lr)
            self.perplexity_history.append(perplexity)
            self.step_times.append(step_time_ms)

    def finalize(self):
        """Finalize metrics after training."""
        if self.initial_loss > 0:
            self.loss_improvement = (1 - self.final_loss / self.initial_loss) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding large history arrays for summary)."""
        return {
            "final_loss": round(self.final_loss, 4),
            "best_loss": round(self.best_loss, 4),
            "loss_improvement_pct": round(self.loss_improvement, 2),
            "final_perplexity": round(self.final_perplexity, 2),
            "best_perplexity": round(self.best_perplexity, 2),
            "steps_completed": self.steps_completed,
            "tokens_processed": self.tokens_processed,
            "tokens_per_second": round(self.tokens_per_second, 1),
            "avg_step_time_ms": round(self.avg_step_time_ms, 2),
            "checkpoints_saved": self.checkpoints_saved,
        }


@dataclass
class ResourceUsage:
    """Resource usage during training."""
    gpu_type: str = ""
    gpu_memory_gb: float = 0.0
    peak_gpu_memory_gb: float = 0.0
    gpu_utilization_pct: float = 0.0
    cpu_memory_gb: float = 0.0
    disk_usage_gb: float = 0.0


@dataclass
class CostInfo:
    """Cost information for training run."""
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    cost_per_step: float = 0.0
    cost_per_token: float = 0.0
    gpu_hours: float = 0.0
    gpu_type: str = ""


@dataclass
class TrainingConfig:
    """Snapshot of training configuration."""
    model_architecture: str = ""
    model_params_millions: float = 0.0
    vocab_size: int = 0
    d_model: int = 0
    n_layers: int = 0
    n_heads: int = 0
    max_seq_len: int = 0
    batch_size: int = 0
    gradient_accumulation: int = 1
    learning_rate: float = 0.0
    weight_decay: float = 0.0
    warmup_steps: int = 0
    max_steps: int = 0
    dataset: str = ""
    mixed_precision: bool = True


@dataclass
class TrainingResult:
    """
    Complete result of a training run.

    Contains all information about a single training session:
    - Status and timing
    - Configuration snapshot
    - Training metrics
    - Resource usage
    - Cost information
    - Checkpoint paths
    - Error details if failed
    """

    # Identification
    run_id: str = ""
    run_name: str = ""
    phase: str = ""  # foundation, expansion, specialization

    # Status
    status: TrainingStatus = TrainingStatus.PENDING
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    # Timing
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0
    duration_hours: float = 0.0

    # Configuration
    config: TrainingConfig = field(default_factory=TrainingConfig)

    # Metrics
    metrics: TrainingMetrics = field(default_factory=TrainingMetrics)

    # Resources
    resources: ResourceUsage = field(default_factory=ResourceUsage)

    # Cost
    cost: CostInfo = field(default_factory=CostInfo)

    # Outputs
    checkpoint_path: str = ""
    best_checkpoint_path: str = ""
    tokenizer_path: str = ""
    log_path: str = ""

    # Metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def start(self):
        """Mark training as started."""
        self.status = TrainingStatus.RUNNING
        self.started_at = datetime.now().isoformat()

    def complete(self, checkpoint_path: str = ""):
        """Mark training as completed."""
        self.status = TrainingStatus.COMPLETED
        self.completed_at = datetime.now().isoformat()
        self.checkpoint_path = checkpoint_path
        self._calculate_duration()
        self.metrics.finalize()

    def fail(self, error: Exception):
        """Mark training as failed."""
        import traceback
        self.status = TrainingStatus.FAILED
        self.completed_at = datetime.now().isoformat()
        self.error_message = str(error)
        self.error_traceback = traceback.format_exc()
        self._calculate_duration()

    def cancel(self):
        """Mark training as cancelled."""
        self.status = TrainingStatus.CANCELLED
        self.completed_at = datetime.now().isoformat()
        self._calculate_duration()

    def _calculate_duration(self):
        """Calculate duration from timestamps."""
        if self.started_at and self.completed_at:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.completed_at)
            self.duration_seconds = (end - start).total_seconds()
            self.duration_hours = self.duration_seconds / 3600

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "phase": self.phase,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_hours": round(self.duration_hours, 2),
            "config": asdict(self.config),
            "metrics": self.metrics.to_dict(),
            "resources": asdict(self.resources),
            "cost": asdict(self.cost),
            "checkpoint_path": self.checkpoint_path,
            "best_checkpoint_path": self.best_checkpoint_path,
            "error_message": self.error_message,
            "tags": self.tags,
            "notes": self.notes,
        }

    def to_json(self, path: Optional[str] = None) -> str:
        """Serialize to JSON."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2)

        if path:
            with open(path, 'w') as f:
                f.write(json_str)

        return json_str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingResult':
        """Create from dictionary."""
        result = cls()
        result.run_id = data.get("run_id", "")
        result.run_name = data.get("run_name", "")
        result.phase = data.get("phase", "")
        result.status = TrainingStatus(data.get("status", "pending"))
        result.started_at = data.get("started_at", "")
        result.completed_at = data.get("completed_at", "")
        result.duration_hours = data.get("duration_hours", 0.0)
        result.checkpoint_path = data.get("checkpoint_path", "")
        result.error_message = data.get("error_message")
        result.tags = data.get("tags", [])
        result.notes = data.get("notes", "")

        if "config" in data:
            result.config = TrainingConfig(**data["config"])

        return result

    @classmethod
    def from_json(cls, path: str) -> 'TrainingResult':
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Training Run: {self.run_name or self.run_id}",
            f"Status: {self.status.value.upper()}",
            f"Duration: {self.duration_hours:.2f} hours",
            "",
            "Metrics:",
            f"  Final Loss: {self.metrics.final_loss:.4f}",
            f"  Best Loss: {self.metrics.best_loss:.4f}",
            f"  Loss Improvement: {self.metrics.loss_improvement:.1f}%",
            f"  Steps Completed: {self.metrics.steps_completed:,}",
            f"  Tokens Processed: {self.metrics.tokens_processed:,}",
            f"  Throughput: {self.metrics.tokens_per_second:.0f} tokens/sec",
            "",
            "Cost:",
            f"  Estimated: ${self.cost.estimated_cost:.2f}",
            f"  Actual: ${self.cost.actual_cost:.2f}",
            f"  GPU Hours: {self.cost.gpu_hours:.2f}",
        ]

        if self.error_message:
            lines.extend(["", f"Error: {self.error_message}"])

        if self.checkpoint_path:
            lines.extend(["", f"Checkpoint: {self.checkpoint_path}"])

        return "\n".join(lines)


@dataclass
class TrainingSession:
    """
    Multi-run training session.

    Tracks multiple training runs (e.g., progressive training phases).
    """

    session_id: str = ""
    session_name: str = ""
    started_at: str = ""
    completed_at: str = ""

    # Runs
    runs: List[TrainingResult] = field(default_factory=list)

    # Aggregates
    total_steps: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_duration_hours: float = 0.0

    # Status
    status: str = "pending"  # pending, running, completed, partial, failed

    def add_run(self, result: TrainingResult):
        """Add a training run to the session."""
        self.runs.append(result)
        self._update_aggregates()

    def _update_aggregates(self):
        """Update aggregate statistics."""
        self.total_steps = sum(r.metrics.steps_completed for r in self.runs)
        self.total_tokens = sum(r.metrics.tokens_processed for r in self.runs)
        self.total_cost = sum(r.cost.actual_cost for r in self.runs)
        self.total_duration_hours = sum(r.duration_hours for r in self.runs)

        # Determine status
        statuses = [r.status for r in self.runs]
        if all(s == TrainingStatus.COMPLETED for s in statuses):
            self.status = "completed"
        elif any(s == TrainingStatus.FAILED for s in statuses):
            if any(s == TrainingStatus.COMPLETED for s in statuses):
                self.status = "partial"
            else:
                self.status = "failed"
        elif any(s == TrainingStatus.RUNNING for s in statuses):
            self.status = "running"
        else:
            self.status = "pending"

    def start(self):
        """Start the session."""
        self.started_at = datetime.now().isoformat()
        self.status = "running"

    def complete(self):
        """Mark session as complete."""
        self.completed_at = datetime.now().isoformat()
        self._update_aggregates()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "total_steps": self.total_steps,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 2),
            "total_duration_hours": round(self.total_duration_hours, 2),
            "runs": [r.to_dict() for r in self.runs],
        }

    def to_json(self, path: str):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'TrainingSession':
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        session = cls()
        session.session_id = data.get("session_id", "")
        session.session_name = data.get("session_name", "")
        session.started_at = data.get("started_at", "")
        session.completed_at = data.get("completed_at", "")
        session.status = data.get("status", "pending")
        session.total_steps = data.get("total_steps", 0)
        session.total_tokens = data.get("total_tokens", 0)
        session.total_cost = data.get("total_cost", 0.0)
        session.total_duration_hours = data.get("total_duration_hours", 0.0)

        for run_data in data.get("runs", []):
            session.runs.append(TrainingResult.from_dict(run_data))

        return session

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Training Session: {self.session_name or self.session_id}",
            f"Status: {self.status.upper()}",
            f"Duration: {self.total_duration_hours:.2f} hours",
            f"Total Cost: ${self.total_cost:.2f}",
            "",
            f"Runs: {len(self.runs)}",
        ]

        for i, run in enumerate(self.runs):
            status_icon = "[OK]" if run.status == TrainingStatus.COMPLETED else "[FAIL]"
            lines.append(f"  {i+1}. {status_icon} {run.run_name}: {run.metrics.final_loss:.4f} loss")

        return "\n".join(lines)


def create_training_result(
    config: Dict[str, Any],
    run_name: str = "",
    phase: str = "",
    tags: List[str] = None
) -> TrainingResult:
    """
    Create a TrainingResult from config dictionary.

    Args:
        config: Training configuration dictionary
        run_name: Human-readable run name
        phase: Training phase (foundation, expansion, specialization)
        tags: Optional tags for categorization

    Returns:
        Initialized TrainingResult
    """
    import uuid

    result = TrainingResult(
        run_id=str(uuid.uuid4())[:8],
        run_name=run_name or f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        phase=phase,
        tags=tags or [],
    )

    # Extract config
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    dataset_config = config.get('dataset', {})

    result.config = TrainingConfig(
        model_architecture=model_config.get('architecture', 'base'),
        vocab_size=model_config.get('vocab_size', 32000),
        d_model=model_config.get('d_model', 768),
        n_layers=model_config.get('num_layers', model_config.get('n_layers', 12)),
        n_heads=model_config.get('num_heads', model_config.get('n_heads', 12)),
        max_seq_len=model_config.get('max_seq_length', model_config.get('max_seq_len', 1024)),
        batch_size=training_config.get('batch_size', 32),
        gradient_accumulation=training_config.get('gradient_accumulation_steps', 1),
        learning_rate=training_config.get('learning_rate', 3e-4),
        weight_decay=training_config.get('weight_decay', 0.01),
        warmup_steps=training_config.get('warmup_steps', 1000),
        max_steps=training_config.get('max_steps', 10000),
        dataset=dataset_config.get('name', ''),
        mixed_precision=config.get('hardware', {}).get('mixed_precision', True),
    )

    # Estimate parameters
    d = result.config.d_model
    n = result.config.n_layers
    v = result.config.vocab_size
    params = v * d + n * (4 * d * d + 8 * d * d)  # Rough estimate
    result.config.model_params_millions = params / 1e6

    return result
