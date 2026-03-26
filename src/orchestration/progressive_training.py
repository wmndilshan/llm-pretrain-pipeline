"""
Progressive Training System

Implements three-phase curriculum learning:
1. Foundation Phase: Train on simple data (TinyStories) - builds basic language understanding
2. Expansion Phase: Train on web data (OpenWebText) - adds factual knowledge
3. Specialization Phase: Train on domain data (Code, etc.) - domain expertise

Benefits:
- Cheaper: $31.90 vs $99 for single run
- Better: Curriculum learning improves quality
- Flexible: Can specialize for different domains monthly
"""

import json
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum

from ..utils.config import load_yaml_config

try:
    from .budget_tracker import BudgetTracker, CostEstimate
except ImportError:
    from budget_tracker import BudgetTracker, CostEstimate


class TrainingPhase(Enum):
    """Training phases in progressive curriculum."""
    FOUNDATION = "foundation"
    EXPANSION = "expansion"
    SPECIALIZATION = "specialization"


@dataclass
class PhaseConfig:
    """Configuration for a training phase."""
    name: str
    phase: TrainingPhase
    dataset: str
    max_steps: int
    gpu_type: str
    estimated_cost: float
    description: str
    checkpoint_name: str
    requires_checkpoint: Optional[str] = None  # Previous phase checkpoint


@dataclass
class PhaseResult:
    """Result of a completed training phase."""
    phase: str
    dataset: str
    status: str  # completed, failed, skipped
    steps_trained: int
    duration_hours: float
    cost_usd: float
    final_loss: float
    final_perplexity: float
    checkpoint_path: str
    started_at: str
    completed_at: str
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressiveTrainingResult:
    """Complete result of progressive training."""
    status: str  # completed, partial, failed
    phases_completed: List[str]
    phases_failed: List[str]
    phases_skipped: List[str]
    total_steps: int
    total_duration_hours: float
    total_cost_usd: float
    final_checkpoint: str
    final_loss: float
    phase_results: List[PhaseResult] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""


# Default progressive training configuration
DEFAULT_PHASES = [
    PhaseConfig(
        name="Foundation",
        phase=TrainingPhase.FOUNDATION,
        dataset="roneneldan/TinyStories",
        max_steps=15000,
        gpu_type="A10G",
        estimated_cost=4.40,
        description="Build basic language understanding with simple narratives",
        checkpoint_name="foundation_checkpoint.pt",
        requires_checkpoint=None,
    ),
    PhaseConfig(
        name="Expansion",
        phase=TrainingPhase.EXPANSION,
        dataset="Skylion007/openwebtext",
        max_steps=35000,
        gpu_type="A100-40GB",
        estimated_cost=17.50,
        description="Add factual knowledge from web text",
        checkpoint_name="expanded_checkpoint.pt",
        requires_checkpoint="foundation_checkpoint.pt",
    ),
    PhaseConfig(
        name="Specialization",
        phase=TrainingPhase.SPECIALIZATION,
        dataset="codeparrot/github-code",
        max_steps=20000,
        gpu_type="A100-40GB",
        estimated_cost=10.00,
        description="Add domain-specific knowledge (code)",
        checkpoint_name="production_checkpoint.pt",
        requires_checkpoint="expanded_checkpoint.pt",
    ),
]


class ProgressiveTrainer:
    """
    Manages progressive training across multiple phases.

    Each phase builds on the previous:
    - Phase 1 (Foundation): Basic language from simple stories
    - Phase 2 (Expansion): World knowledge from web text
    - Phase 3 (Specialization): Domain expertise (customizable)

    Total budget: ~$31.90 for complete training
    """

    def __init__(
        self,
        config_path: str,
        budget_tracker: Optional[BudgetTracker] = None,
        phases: Optional[List[PhaseConfig]] = None,
        checkpoint_dir: str = "models/progressive_checkpoints",
        state_file: str = "progressive_training_state.json"
    ):
        self.config_path = Path(config_path)
        self.budget_tracker = budget_tracker or BudgetTracker()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.state_file = Path(state_file)

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load base config
        self.base_config = self._load_config()
        self.phases = phases or self._load_phases_from_config() or DEFAULT_PHASES

        # Load or create state
        self.state = self._load_state()

    def _load_config(self) -> Dict[str, Any]:
        """Load base training configuration."""
        return load_yaml_config(self.config_path)

    def _load_phases_from_config(self) -> List[PhaseConfig]:
        """Load progressive phases from the merged runtime config when present."""
        phases = []
        phase_dicts = self.base_config.get("progressive_training", {}).get("phases", [])
        for phase_data in phase_dicts:
            phase_value = phase_data.get("phase", "foundation")
            try:
                phase_enum = TrainingPhase(phase_value)
            except ValueError:
                continue

            phases.append(
                PhaseConfig(
                    name=phase_data.get("name", phase_value.title()),
                    phase=phase_enum,
                    dataset=phase_data.get("dataset", self.base_config.get("dataset", {}).get("name", "")),
                    max_steps=phase_data.get("max_steps", self.base_config.get("training", {}).get("max_steps", 10000)),
                    gpu_type=phase_data.get("gpu_type", self.base_config.get("hardware", {}).get("modal_gpu", "A10G")),
                    estimated_cost=phase_data.get("estimated_cost", 0.0),
                    description=phase_data.get("description", ""),
                    checkpoint_name=phase_data.get("checkpoint_name", f"{phase_value}.pt"),
                    requires_checkpoint=phase_data.get("requires_checkpoint"),
                )
            )
        return phases

    def _load_state(self) -> Dict[str, Any]:
        """Load progressive training state."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "completed_phases": [],
            "current_phase": None,
            "phase_results": [],
            "started_at": None,
            "last_checkpoint": None,
        }

    def _save_state(self):
        """Save progressive training state."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def estimate_total_cost(self) -> Dict[str, Any]:
        """Estimate total cost for all phases."""
        total_cost = 0.0
        total_steps = 0
        total_hours = 0.0
        phase_estimates = []

        for phase in self.phases:
            estimate = self.budget_tracker.estimate_cost(
                max_steps=phase.max_steps,
                gpu_type=phase.gpu_type,
                model_params_millions=self.budget_tracker.estimate_model_params_millions_from_config(self.base_config),
            )
            total_cost += estimate.estimated_cost
            total_steps += phase.max_steps
            total_hours += estimate.estimated_hours

            phase_estimates.append({
                "phase": phase.name,
                "dataset": phase.dataset,
                "steps": phase.max_steps,
                "gpu": phase.gpu_type,
                "cost": estimate.estimated_cost,
                "hours": estimate.estimated_hours,
            })

        return {
            "total_cost": round(total_cost, 2),
            "total_steps": total_steps,
            "total_hours": round(total_hours, 2),
            "phases": phase_estimates,
            "within_budget": self.budget_tracker.can_afford(total_cost),
            "budget_remaining": round(
                self.budget_tracker.safety_margin_budget -
                self.budget_tracker.get_current_month_spending(), 2
            ),
        }

    def get_next_phase(self) -> Optional[PhaseConfig]:
        """Get the next phase to run."""
        completed = set(self.state.get("completed_phases", []))

        for phase in self.phases:
            if phase.phase.value not in completed:
                return phase

        return None  # All phases completed

    def can_run_phase(self, phase: PhaseConfig) -> tuple[bool, str]:
        """
        Check if a phase can be run.

        Returns:
            (can_run, reason)
        """
        # Check budget
        estimate = self.budget_tracker.estimate_cost(
            max_steps=phase.max_steps,
            gpu_type=phase.gpu_type,
            model_params_millions=self.budget_tracker.estimate_model_params_millions_from_config(self.base_config),
        )

        if not estimate.within_safety_margin:
            return False, f"Insufficient budget. Need ${estimate.estimated_cost:.2f}, have ${estimate.remaining_budget:.2f} (safety margin)"

        # Check prerequisites
        if phase.requires_checkpoint:
            checkpoint_path = self.checkpoint_dir / phase.requires_checkpoint
            if not checkpoint_path.exists():
                return False, f"Required checkpoint not found: {phase.requires_checkpoint}"

        return True, "Ready to run"

    def run_phase(
        self,
        phase: PhaseConfig,
        orchestrator,  # ModalTrainingOrchestrator or similar
        skip_validation: bool = False
    ) -> PhaseResult:
        """
        Run a single training phase.

        Args:
            phase: Phase configuration
            orchestrator: Training orchestrator with train_on_dataset method
            skip_validation: Skip pre-training validation

        Returns:
            PhaseResult with training outcome
        """
        from ..utils.logging import log_step, log_info, log_ok, log_fail, log_warn

        log_step(f"PROGRESSIVE TRAINING: {phase.name} Phase")
        log_info(f"Dataset: {phase.dataset}")
        log_info(f"Steps: {phase.max_steps:,}")
        log_info(f"GPU: {phase.gpu_type}")
        log_info(f"Estimated cost: ${phase.estimated_cost:.2f}")

        # Check if phase can run
        can_run, reason = self.can_run_phase(phase)
        if not can_run:
            log_fail(f"Cannot run phase: {reason}")
            return PhaseResult(
                phase=phase.phase.value,
                dataset=phase.dataset,
                status="skipped",
                steps_trained=0,
                duration_hours=0,
                cost_usd=0,
                final_loss=0,
                final_perplexity=0,
                checkpoint_path="",
                started_at=datetime.now().isoformat(),
                completed_at=datetime.now().isoformat(),
                error_message=reason,
            )

        # Update state
        self.state["current_phase"] = phase.phase.value
        if not self.state["started_at"]:
            self.state["started_at"] = datetime.now().isoformat()
        self._save_state()

        started_at = datetime.now()

        try:
            if hasattr(orchestrator, "config"):
                orchestrator.config = copy.deepcopy(orchestrator.config)
                orchestrator.config.setdefault("dataset", {})["name"] = phase.dataset
                orchestrator.config.setdefault("training", {})["max_steps"] = phase.max_steps
                orchestrator.config.setdefault("hardware", {})["modal_gpu"] = phase.gpu_type

            # Set previous checkpoint if required
            previous_checkpoint = None
            if phase.requires_checkpoint:
                previous_checkpoint = str(self.checkpoint_dir / phase.requires_checkpoint)
                log_info(f"Loading from checkpoint: {previous_checkpoint}")

            # Run training
            log_info("Starting training...")
            result_path = orchestrator.train_on_dataset(
                phase.dataset,
                skip_validation=skip_validation,
                previous_checkpoint=previous_checkpoint,
                phase_name=phase.phase.value
            )

            completed_at = datetime.now()
            duration = (completed_at - started_at).total_seconds() / 3600

            # Copy checkpoint to progressive checkpoint directory
            import shutil
            checkpoint_dest = self.checkpoint_dir / phase.checkpoint_name
            if Path(result_path).exists():
                shutil.copy(result_path, checkpoint_dest)

            # Get actual cost from estimate
            estimate = self.budget_tracker.estimate_cost(
                max_steps=phase.max_steps,
                gpu_type=phase.gpu_type,
                model_params_millions=self.budget_tracker.estimate_model_params_millions_from_config(self.base_config),
            )

            # Record spending
            self.budget_tracker.record_spending(
                dataset=phase.dataset,
                gpu_type=phase.gpu_type,
                duration_hours=duration,
                cost_usd=estimate.estimated_cost,
                steps_trained=phase.max_steps,
                phase=phase.phase.value
            )

            # Update state
            self.state["completed_phases"].append(phase.phase.value)
            self.state["last_checkpoint"] = str(checkpoint_dest)
            self.state["current_phase"] = None
            self._save_state()

            log_ok(f"Phase completed: {phase.name}")

            return PhaseResult(
                phase=phase.phase.value,
                dataset=phase.dataset,
                status="completed",
                steps_trained=phase.max_steps,
                duration_hours=round(duration, 2),
                cost_usd=estimate.estimated_cost,
                final_loss=0,  # Would get from training
                final_perplexity=0,
                checkpoint_path=str(checkpoint_dest),
                started_at=started_at.isoformat(),
                completed_at=completed_at.isoformat(),
            )

        except Exception as e:
            log_fail(f"Phase failed: {e}")
            return PhaseResult(
                phase=phase.phase.value,
                dataset=phase.dataset,
                status="failed",
                steps_trained=0,
                duration_hours=0,
                cost_usd=0,
                final_loss=0,
                final_perplexity=0,
                checkpoint_path="",
                started_at=started_at.isoformat(),
                completed_at=datetime.now().isoformat(),
                error_message=str(e),
            )

    def run_all_phases(
        self,
        orchestrator,
        skip_validation: bool = False,
        stop_on_failure: bool = True
    ) -> ProgressiveTrainingResult:
        """
        Run all progressive training phases.

        Args:
            orchestrator: Training orchestrator
            skip_validation: Skip pre-training validation
            stop_on_failure: Stop if any phase fails

        Returns:
            ProgressiveTrainingResult with complete outcome
        """
        from ..utils.logging import log_step, log_info, log_ok, log_fail

        log_step("PROGRESSIVE TRAINING - FULL RUN")

        # Estimate total cost
        estimate = self.estimate_total_cost()
        log_info(f"Total estimated cost: ${estimate['total_cost']:.2f}")
        log_info(f"Total steps: {estimate['total_steps']:,}")
        log_info(f"Total hours: {estimate['total_hours']:.1f}")

        if not estimate['within_budget']:
            log_fail(f"Cannot afford full training. Need ${estimate['total_cost']:.2f}, have ${estimate['budget_remaining']:.2f}")

        started_at = datetime.now()
        phase_results = []
        completed = []
        failed = []
        skipped = []

        for phase in self.phases:
            # Skip already completed phases
            if phase.phase.value in self.state.get("completed_phases", []):
                log_info(f"Skipping {phase.name} (already completed)")
                skipped.append(phase.phase.value)
                continue

            # Run phase
            result = self.run_phase(phase, orchestrator, skip_validation)
            phase_results.append(result)

            if result.status == "completed":
                completed.append(phase.phase.value)
            elif result.status == "failed":
                failed.append(phase.phase.value)
                if stop_on_failure:
                    log_fail(f"Stopping due to failure in {phase.name}")
                    break
            else:
                skipped.append(phase.phase.value)

        completed_at = datetime.now()

        # Calculate totals
        total_steps = sum(r.steps_trained for r in phase_results)
        total_hours = sum(r.duration_hours for r in phase_results)
        total_cost = sum(r.cost_usd for r in phase_results)

        # Determine final status
        if len(failed) > 0:
            status = "partial" if len(completed) > 0 else "failed"
        elif len(completed) == len(self.phases):
            status = "completed"
        else:
            status = "partial"

        # Get final checkpoint
        final_checkpoint = ""
        for result in reversed(phase_results):
            if result.checkpoint_path:
                final_checkpoint = result.checkpoint_path
                break

        return ProgressiveTrainingResult(
            status=status,
            phases_completed=completed,
            phases_failed=failed,
            phases_skipped=skipped,
            total_steps=total_steps,
            total_duration_hours=round(total_hours, 2),
            total_cost_usd=round(total_cost, 2),
            final_checkpoint=final_checkpoint,
            final_loss=phase_results[-1].final_loss if phase_results else 0,
            phase_results=phase_results,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
        )

    def reset(self):
        """Reset progressive training state (start fresh)."""
        self.state = {
            "completed_phases": [],
            "current_phase": None,
            "phase_results": [],
            "started_at": None,
            "last_checkpoint": None,
        }
        self._save_state()

    def get_status(self) -> Dict[str, Any]:
        """Get current progressive training status."""
        return {
            "completed_phases": self.state.get("completed_phases", []),
            "current_phase": self.state.get("current_phase"),
            "last_checkpoint": self.state.get("last_checkpoint"),
            "started_at": self.state.get("started_at"),
            "next_phase": self.get_next_phase().name if self.get_next_phase() else None,
            "budget_summary": self.budget_tracker.get_spending_summary(),
        }
