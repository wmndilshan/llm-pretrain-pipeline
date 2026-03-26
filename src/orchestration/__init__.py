"""
Orchestration Module

Central coordination for LLM training pipeline:
- Budget tracking with 90% safety margin
- Pre-training validation
- Progressive 3-phase training
- Training results and session tracking
- Monthly scheduling
"""

from .budget_tracker import (
    BudgetTracker,
    TrainingCost,
    CostEstimate,
)

from .validation import (
    PreTrainingValidator,
    ValidationResult,
    ValidationReport,
    ValidationLevel,
    validate_before_training,
)

from .progressive_training import (
    ProgressiveTrainer,
    TrainingPhase,
    PhaseConfig,
    PhaseResult,
    ProgressiveTrainingResult,
    DEFAULT_PHASES,
)

from .training_results import (
    TrainingResult,
    TrainingSession,
    TrainingStatus,
    TrainingMetrics,
    TrainingConfig,
    CostInfo,
    ResourceUsage,
    create_training_result,
)

from .orchestrator import (
    TrainingOrchestrator,
    OrchestratorConfig,
)

from .scheduler import (
    MonthlyScheduler,
    ScheduledTraining,
)

from .artifacts import (
    dataset_slug,
    export_inference_artifacts,
)

__all__ = [
    # Budget
    "BudgetTracker",
    "TrainingCost",
    "CostEstimate",

    # Validation
    "PreTrainingValidator",
    "ValidationResult",
    "ValidationReport",
    "ValidationLevel",
    "validate_before_training",

    # Progressive Training
    "ProgressiveTrainer",
    "TrainingPhase",
    "PhaseConfig",
    "PhaseResult",
    "ProgressiveTrainingResult",
    "DEFAULT_PHASES",

    # Results
    "TrainingResult",
    "TrainingSession",
    "TrainingStatus",
    "TrainingMetrics",
    "TrainingConfig",
    "CostInfo",
    "ResourceUsage",
    "create_training_result",

    # Orchestrator
    "TrainingOrchestrator",
    "OrchestratorConfig",

    # Scheduler
    "MonthlyScheduler",
    "ScheduledTraining",

    # Artifacts
    "dataset_slug",
    "export_inference_artifacts",
]
