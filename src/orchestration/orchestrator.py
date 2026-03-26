"""
Main Training Orchestrator

Central orchestrator that ties together all training components:
- Budget tracking with 90% safety margin
- Pre-training validation
- Progressive training (3 phases)
- Modal GPU training with volume commits
- Structured training results
- Scheduling and automation

Usage:
    orchestrator = TrainingOrchestrator("configs/config.yaml")

    # Single dataset training
    result = orchestrator.train("roneneldan/TinyStories")

    # Progressive training (3 phases)
    result = orchestrator.train_progressive()

    # Custom phase
    result = orchestrator.train_phase("foundation", "roneneldan/TinyStories")
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .budget_tracker import BudgetTracker
from .validation import PreTrainingValidator, ValidationReport
from .progressive_training import ProgressiveTrainer, TrainingPhase, PhaseConfig
from .training_results import (
    TrainingResult, TrainingSession, TrainingStatus,
    create_training_result
)
from ..utils.config import load_yaml_config
from ..utils.logging import log_fail, log_info, log_ok, log_step, log_warn


@dataclass
class OrchestratorConfig:
    """Configuration for training orchestrator."""
    config_path: str
    monthly_budget: float = 30.0
    skip_validation: bool = False
    auto_confirm: bool = False  # Skip cost confirmation prompts
    checkpoint_dir: str = "models/checkpoints"
    results_dir: str = "training_results"


class TrainingOrchestrator:
    """
    Central orchestrator for LLM training pipeline.

    Integrates:
    - Budget tracking with 90% safety margin
    - Pre-training validation
    - Progressive 3-phase training
    - Modal GPU training
    - Structured result tracking
    """

    def __init__(
        self,
        config_path: str,
        monthly_budget: float = 30.0,
        skip_validation: bool = False,
        auto_confirm: bool = False
    ):
        self.config_path = Path(config_path)
        self.monthly_budget = monthly_budget
        self.skip_validation = skip_validation
        self.auto_confirm = auto_confirm

        # Load config
        self.config = load_yaml_config(config_path)

        # Initialize components
        self.budget_tracker = BudgetTracker(monthly_budget=monthly_budget)
        self.progressive_trainer = ProgressiveTrainer(
            config_path=config_path,
            budget_tracker=self.budget_tracker
        )

        # Results tracking
        self.results_dir = Path("training_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: Optional[TrainingSession] = None

    def _log(self, message: str, level: str = "INFO"):
        """Route orchestrator output through the shared professional logger."""
        if level == "OK":
            log_ok(message)
        elif level == "WARN":
            log_warn(message)
        elif level == "FAIL":
            log_fail(message)
        elif level == "STEP":
            log_step(message)
        else:
            log_info(message)

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        budget_summary = self.budget_tracker.get_spending_summary()
        progressive_status = self.progressive_trainer.get_status()

        return {
            "config_path": str(self.config_path),
            "budget": {
                "monthly": self.monthly_budget,
                "spent_this_month": budget_summary["current_month"],
                "remaining": budget_summary["remaining"],
                "remaining_safe": budget_summary["remaining_safe"],
                "utilization_pct": budget_summary["budget_utilization_pct"],
            },
            "progressive_training": {
                "completed_phases": progressive_status["completed_phases"],
                "next_phase": progressive_status["next_phase"],
                "last_checkpoint": progressive_status["last_checkpoint"],
            },
            "model": {
                "architecture": self.config.get("model", {}).get("architecture", "base"),
                "vocab_size": self.config.get("model", {}).get("vocab_size", 32000),
            }
        }

    def validate(self, verbose: bool = True) -> Tuple[bool, ValidationReport]:
        """
        Run pre-training validation.

        Returns:
            (can_proceed, validation_report)
        """
        validator = PreTrainingValidator(
            config=self.config,
            budget_tracker=self.budget_tracker,
            verbose=verbose
        )
        report = validator.validate_all()
        return report.can_proceed, report

    def estimate_cost(
        self,
        max_steps: Optional[int] = None,
        gpu_type: str = "A10G"
    ) -> Dict[str, Any]:
        """Estimate training cost."""
        steps = max_steps or self.config.get("training", {}).get("max_steps", 10000)
        estimate = self.budget_tracker.estimate_cost(
            max_steps=steps,
            gpu_type=gpu_type,
            model_params_millions=self.budget_tracker.estimate_model_params_millions_from_config(self.config),
        )

        return {
            "estimated_cost": estimate.estimated_cost,
            "estimated_hours": estimate.estimated_hours,
            "gpu_type": estimate.gpu_type,
            "steps": estimate.steps,
            "within_budget": estimate.within_safety_margin,
            "remaining_after": estimate.remaining_budget - estimate.estimated_cost,
            "warning": estimate.warning_message,
        }

    def train(
        self,
        dataset_name: Optional[str] = None,
        max_steps: Optional[int] = None,
        gpu_type: str = "A10G",
        skip_validation: bool = False,
        resume_from: Optional[str] = None,
    ) -> TrainingResult:
        """
        Train on a single dataset.

        Args:
            dataset_name: HuggingFace dataset name (default from config)
            max_steps: Maximum training steps (default from config)
            gpu_type: GPU type for Modal
            skip_validation: Skip pre-training validation
            resume_from: Path to checkpoint to resume from

        Returns:
            TrainingResult with complete training outcome
        """
        # Get dataset from config if not specified
        dataset = dataset_name or self.config.get("dataset", {}).get("name", "roneneldan/TinyStories")
        steps = max_steps or self.config.get("training", {}).get("max_steps", 10000)

        self._log(f"TRAINING: {dataset}", "STEP")
        self._log(f"Steps: {steps:,}, GPU: {gpu_type}")

        # Create result object
        result = create_training_result(
            config=self.config,
            run_name=f"{dataset.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        # Validation
        if not (skip_validation or self.skip_validation):
            self._log("Running pre-training validation...", "STEP")
            can_proceed, report = self.validate(verbose=True)

            if not can_proceed:
                self._log("Validation failed. Cannot proceed.", "FAIL")
                result.fail(Exception("Pre-training validation failed"))
                return result

            if report.estimated_cost:
                result.cost.estimated_cost = report.estimated_cost

        # Cost estimation and confirmation
        estimate = self.budget_tracker.estimate_cost(
            max_steps=steps,
            gpu_type=gpu_type,
            model_params_millions=self.budget_tracker.estimate_model_params_millions_from_config(self.config),
        )
        self._log(f"Estimated cost: ${estimate.estimated_cost:.2f}", "INFO")

        if not estimate.within_safety_margin:
            self._log(f"Insufficient budget! Need ${estimate.estimated_cost:.2f}, "
                     f"have ${estimate.remaining_budget:.2f}", "FAIL")
            result.fail(Exception("Insufficient budget"))
            return result

        if not self.auto_confirm:
            response = input(f"\nProceed with training? (${estimate.estimated_cost:.2f}) [y/N]: ")
            if response.lower() != 'y':
                self._log("Training cancelled by user", "WARN")
                result.cancel()
                return result

        # Import and run Modal training
        try:
            from ..pipeline.modal_training import ModalTrainingOrchestrator

            modal_orchestrator = ModalTrainingOrchestrator(
                str(self.config_path),
                budget_tracker=self.budget_tracker,
                skip_validation=True  # Already validated
            )
            modal_orchestrator.config["training"]["max_steps"] = steps
            modal_orchestrator.config.setdefault("hardware", {})["modal_gpu"] = gpu_type

            result.start()
            checkpoint_path = modal_orchestrator.train_on_dataset(
                dataset,
                skip_validation=True,
                previous_checkpoint=resume_from,
            )

            if checkpoint_path:
                result.complete(checkpoint_path)
                self._log(f"Training complete! Checkpoint: {checkpoint_path}", "OK")
            else:
                result.fail(Exception("Training failed - no checkpoint produced"))
                self._log("Training failed", "FAIL")

        except Exception as e:
            result.fail(e)
            self._log(f"Training error: {e}", "FAIL")

        # Save result
        result_path = self.results_dir / f"result_{result.run_id}.json"
        result.to_json(str(result_path))
        self._log(f"Result saved: {result_path}", "INFO")

        return result

    def train_progressive(
        self,
        stop_on_failure: bool = True,
        skip_validation: bool = False
    ) -> TrainingSession:
        """
        Run progressive 3-phase training.

        Phases:
        1. Foundation: TinyStories (basic language)
        2. Expansion: OpenWebText (world knowledge)
        3. Specialization: Code data (domain expertise)

        Args:
            stop_on_failure: Stop if any phase fails
            skip_validation: Skip pre-training validation

        Returns:
            TrainingSession with all phase results
        """
        self._log("PROGRESSIVE TRAINING (3 Phases)", "STEP")

        # Create session
        session = TrainingSession(
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            session_name="progressive_training",
        )
        session.start()
        self.current_session = session

        # Estimate total cost
        estimate = self.progressive_trainer.estimate_total_cost()
        self._log(f"Total estimated cost: ${estimate['total_cost']:.2f}", "INFO")
        self._log(f"Total steps: {estimate['total_steps']:,}", "INFO")
        self._log(f"Estimated hours: {estimate['total_hours']:.1f}", "INFO")

        if not estimate['within_budget']:
            self._log(f"Cannot afford full training. Budget: ${estimate['budget_remaining']:.2f}", "FAIL")

        # Print phase breakdown
        print("\nPhase breakdown:")
        for phase in estimate['phases']:
            print(f"  {phase['phase']}: {phase['dataset']}")
            print(f"    Steps: {phase['steps']:,}, GPU: {phase['gpu']}, Cost: ${phase['cost']:.2f}")

        if not self.auto_confirm:
            response = input(f"\nProceed with progressive training? [y/N]: ")
            if response.lower() != 'y':
                self._log("Training cancelled by user", "WARN")
                session.status = "cancelled"
                return session

        # Run phases
        for i, phase in enumerate(self.progressive_trainer.phases):
            self._log(f"\n{'='*60}", "STEP")
            self._log(f"PHASE {i+1}/3: {phase.name}", "STEP")
            self._log(f"Dataset: {phase.dataset}", "INFO")
            self._log(f"Steps: {phase.max_steps:,}, GPU: {phase.gpu_type}", "INFO")
            self._log(f"='*60", "STEP")

            # Check if already completed
            if phase.phase.value in self.progressive_trainer.state.get("completed_phases", []):
                self._log(f"Phase already completed, skipping", "INFO")
                continue

            # Create result for this phase
            result = create_training_result(
                config=self.config,
                run_name=f"{phase.name}_{datetime.now().strftime('%Y%m%d')}",
                phase=phase.phase.value,
            )

            # Run phase
            try:
                from ..pipeline.modal_training import ModalTrainingOrchestrator

                modal_orchestrator = ModalTrainingOrchestrator(
                    str(self.config_path),
                    budget_tracker=self.budget_tracker,
                    skip_validation=skip_validation or self.skip_validation
                )

                # Update config for this phase
                modal_orchestrator.config["dataset"]["name"] = phase.dataset
                modal_orchestrator.config["training"]["max_steps"] = phase.max_steps
                modal_orchestrator.config.setdefault("hardware", {})["modal_gpu"] = phase.gpu_type

                # Get previous checkpoint
                previous_checkpoint = None
                if phase.requires_checkpoint:
                    previous_checkpoint = str(
                        self.progressive_trainer.checkpoint_dir / phase.requires_checkpoint
                    )

                result.start()
                checkpoint_path = modal_orchestrator.train_on_dataset(
                    phase.dataset,
                    skip_validation=True,
                    previous_checkpoint=previous_checkpoint,
                    phase_name=phase.phase.value,
                )

                if checkpoint_path:
                    result.complete(checkpoint_path)
                    self._log(f"Phase {phase.name} complete!", "OK")

                    # Copy to progressive checkpoint dir
                    import shutil
                    dest = self.progressive_trainer.checkpoint_dir / phase.checkpoint_name
                    shutil.copy(checkpoint_path, dest)

                    # Update state
                    self.progressive_trainer.state["completed_phases"].append(phase.phase.value)
                    self.progressive_trainer.state["last_checkpoint"] = str(dest)
                    self.progressive_trainer._save_state()
                else:
                    result.fail(Exception("No checkpoint produced"))
                    self._log(f"Phase {phase.name} failed", "FAIL")

                    if stop_on_failure:
                        break

            except Exception as e:
                result.fail(e)
                self._log(f"Phase {phase.name} error: {e}", "FAIL")

                if stop_on_failure:
                    break

            # Add to session
            session.add_run(result)

        # Complete session
        session.complete()

        # Save session
        session_path = self.results_dir / f"session_{session.session_id}.json"
        session.to_json(str(session_path))
        self._log(f"\nSession saved: {session_path}", "INFO")

        # Print summary
        print("\n" + "="*60)
        print(session.summary())
        print("="*60)

        return session

    def train_phase(
        self,
        phase_name: str,
        dataset_name: Optional[str] = None,
    ) -> TrainingResult:
        """
        Train a specific phase.

        Args:
            phase_name: Phase name (foundation, expansion, specialization)
            dataset_name: Override dataset for this phase

        Returns:
            TrainingResult
        """
        # Find phase config
        phase_map = {
            "foundation": TrainingPhase.FOUNDATION,
            "expansion": TrainingPhase.EXPANSION,
            "specialization": TrainingPhase.SPECIALIZATION,
        }

        if phase_name.lower() not in phase_map:
            raise ValueError(f"Unknown phase: {phase_name}. Must be one of: {list(phase_map.keys())}")

        target_phase = phase_map[phase_name.lower()]

        for phase in self.progressive_trainer.phases:
            if phase.phase == target_phase:
                dataset = dataset_name or phase.dataset
                return self.train(
                    dataset_name=dataset,
                    max_steps=phase.max_steps,
                    gpu_type=phase.gpu_type,
                )

        raise ValueError(f"Phase not found: {phase_name}")

    def resume(self) -> TrainingSession:
        """Resume progressive training from last checkpoint."""
        self._log("Resuming progressive training...", "STEP")

        status = self.progressive_trainer.get_status()
        completed = status["completed_phases"]

        if not completed:
            self._log("No phases completed. Starting from beginning.", "INFO")
        else:
            self._log(f"Completed phases: {completed}", "INFO")
            self._log(f"Next phase: {status['next_phase']}", "INFO")

        return self.train_progressive(skip_validation=True)

    def reset(self, confirm: bool = False):
        """Reset progressive training state."""
        if not confirm:
            response = input("This will reset all progressive training progress. Continue? [y/N]: ")
            if response.lower() != 'y':
                self._log("Reset cancelled", "INFO")
                return

        self.progressive_trainer.reset()
        self._log("Progressive training state reset", "OK")


def main():
    """CLI for training orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM Training Orchestrator")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--budget", type=float, default=30.0, help="Monthly budget in USD")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Status command
    subparsers.add_parser("status", help="Show orchestrator status")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train on single dataset")
    train_parser.add_argument("--dataset", help="Dataset name")
    train_parser.add_argument("--steps", type=int, help="Max training steps")
    train_parser.add_argument("--gpu", default="A10G", help="GPU type")
    train_parser.add_argument("--skip-validation", action="store_true")
    train_parser.add_argument("--yes", action="store_true", help="Auto-confirm")

    # Progressive command
    prog_parser = subparsers.add_parser("progressive", help="Run progressive training")
    prog_parser.add_argument("--skip-validation", action="store_true")
    prog_parser.add_argument("--yes", action="store_true", help="Auto-confirm")

    # Resume command
    subparsers.add_parser("resume", help="Resume progressive training")

    # Validate command
    subparsers.add_parser("validate", help="Run validation only")

    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset progressive training")
    reset_parser.add_argument("--yes", action="store_true", help="Skip confirmation")

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = TrainingOrchestrator(
        config_path=args.config,
        monthly_budget=args.budget,
        auto_confirm=getattr(args, 'yes', False)
    )

    if args.command == "status":
        import json
        print(json.dumps(orchestrator.get_status(), indent=2))

    elif args.command == "train":
        orchestrator.train(
            dataset_name=args.dataset,
            max_steps=args.steps,
            gpu_type=args.gpu,
            skip_validation=args.skip_validation,
        )

    elif args.command == "progressive":
        orchestrator.train_progressive(
            skip_validation=args.skip_validation
        )

    elif args.command == "resume":
        orchestrator.resume()

    elif args.command == "validate":
        orchestrator.validate(verbose=True)

    elif args.command == "reset":
        orchestrator.reset(confirm=args.yes)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
