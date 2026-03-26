"""
Monthly Training Scheduler

Handles automated scheduling of training runs.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from .budget_tracker import BudgetTracker


@dataclass
class ScheduledTraining:
    """Scheduled training run configuration."""
    dataset: str
    config_path: str
    max_steps: int
    scheduled_date: str
    status: str = "pending"  # pending, running, completed, failed, skipped
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result_path: Optional[str] = None
    error_message: Optional[str] = None


class MonthlyScheduler:
    """
    Schedule and manage monthly training runs.

    Features:
    - Schedule training runs for specific dates
    - Track training history
    - Integrate with budget tracker
    - Support multiple datasets per month
    """

    def __init__(
        self,
        schedule_file: str = "training_schedule.json",
        budget_tracker: Optional[BudgetTracker] = None
    ):
        self.schedule_file = Path(schedule_file)
        self.budget_tracker = budget_tracker or BudgetTracker()
        self.scheduled_runs: List[ScheduledTraining] = []
        self._load_schedule()

    def _load_schedule(self):
        """Load schedule from file."""
        if self.schedule_file.exists():
            with open(self.schedule_file, 'r') as f:
                data = json.load(f)
                self.scheduled_runs = [
                    ScheduledTraining(**run) for run in data.get('runs', [])
                ]

    def _save_schedule(self):
        """Save schedule to file."""
        data = {
            'runs': [asdict(run) for run in self.scheduled_runs]
        }
        with open(self.schedule_file, 'w') as f:
            json.dump(data, f, indent=2)

    def schedule_training(
        self,
        dataset: str,
        config_path: str,
        max_steps: int,
        scheduled_date: Optional[str] = None
    ) -> ScheduledTraining:
        """
        Schedule a new training run.

        Args:
            dataset: Dataset name/path
            config_path: Path to training config
            max_steps: Maximum training steps
            scheduled_date: Date to run (ISO format), defaults to now

        Returns:
            ScheduledTraining object
        """
        if scheduled_date is None:
            scheduled_date = datetime.now().isoformat()

        run = ScheduledTraining(
            dataset=dataset,
            config_path=config_path,
            max_steps=max_steps,
            scheduled_date=scheduled_date
        )

        self.scheduled_runs.append(run)
        self._save_schedule()

        return run

    def get_pending_runs(self) -> List[ScheduledTraining]:
        """Get all pending training runs."""
        now = datetime.now()
        return [
            run for run in self.scheduled_runs
            if run.status == "pending" and
            datetime.fromisoformat(run.scheduled_date) <= now
        ]

    def mark_started(self, dataset: str):
        """Mark a training run as started."""
        for run in self.scheduled_runs:
            if run.dataset == dataset and run.status == "pending":
                run.status = "running"
                run.started_at = datetime.now().isoformat()
                self._save_schedule()
                return

    def mark_completed(self, dataset: str, result_path: str):
        """Mark a training run as completed."""
        for run in self.scheduled_runs:
            if run.dataset == dataset and run.status == "running":
                run.status = "completed"
                run.completed_at = datetime.now().isoformat()
                run.result_path = result_path
                self._save_schedule()
                return

    def mark_failed(self, dataset: str, error_message: str):
        """Mark a training run as failed."""
        for run in self.scheduled_runs:
            if run.dataset == dataset and run.status == "running":
                run.status = "failed"
                run.completed_at = datetime.now().isoformat()
                run.error_message = error_message
                self._save_schedule()
                return

    def get_history(self, limit: int = 10) -> List[ScheduledTraining]:
        """Get recent training history."""
        completed = [
            run for run in self.scheduled_runs
            if run.status in ("completed", "failed")
        ]
        # Sort by completion time, most recent first
        completed.sort(
            key=lambda x: x.completed_at or "",
            reverse=True
        )
        return completed[:limit]

    def get_summary(self) -> Dict[str, any]:
        """Get schedule summary."""
        pending = len([r for r in self.scheduled_runs if r.status == "pending"])
        running = len([r for r in self.scheduled_runs if r.status == "running"])
        completed = len([r for r in self.scheduled_runs if r.status == "completed"])
        failed = len([r for r in self.scheduled_runs if r.status == "failed"])

        budget_summary = self.budget_tracker.get_spending_summary()

        return {
            "pending_runs": pending,
            "running_runs": running,
            "completed_runs": completed,
            "failed_runs": failed,
            "total_runs": len(self.scheduled_runs),
            "budget": budget_summary
        }

    def run_pending(self, orchestrator) -> List[str]:
        """
        Execute all pending training runs.

        Args:
            orchestrator: ModalTrainingOrchestrator instance

        Returns:
            List of completed model paths
        """
        results = []
        pending = self.get_pending_runs()

        for run in pending:
            # Check budget
            estimate = self.budget_tracker.estimate_cost(run.max_steps)
            if not estimate.within_safety_margin:
                print(f"Skipping {run.dataset}: Over budget")
                run.status = "skipped"
                run.error_message = "Over budget"
                self._save_schedule()
                continue

            try:
                self.mark_started(run.dataset)
                print(f"Starting training: {run.dataset}")

                # Run training
                model_path = orchestrator.train_on_dataset(
                    run.dataset,
                    skip_validation=True
                )

                self.mark_completed(run.dataset, model_path)
                results.append(model_path)

                # Record cost (estimate)
                self.budget_tracker.record_spending(
                    dataset=run.dataset,
                    gpu_type=estimate.gpu_type,
                    duration_hours=estimate.estimated_hours,
                    cost_usd=estimate.estimated_cost,
                    steps_trained=run.max_steps
                )

            except Exception as e:
                self.mark_failed(run.dataset, str(e))
                print(f"Training failed for {run.dataset}: {e}")

        return results
