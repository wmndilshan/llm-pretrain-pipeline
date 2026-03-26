"""
Budget Tracking for Cloud Training

Features:
- Monthly budget limits with 90% safety margin
- Detailed cost estimation by GPU type
- Historical spending tracking
- Automatic budget warnings
- Spending persistence in JSON
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict, field


@dataclass
class TrainingCost:
    """Record of a single training run cost."""
    timestamp: str
    dataset: str
    gpu_type: str
    duration_hours: float
    cost_usd: float
    steps_trained: int
    phase: Optional[str] = None  # foundation, expansion, specialization


@dataclass
class CostEstimate:
    """Detailed cost estimation result."""
    estimated_cost: float
    estimated_hours: float
    gpu_type: str
    steps: int
    current_month_spending: float
    remaining_budget: float
    safety_margin_budget: float  # 90% of monthly budget
    within_budget: bool
    within_safety_margin: bool
    budget_utilization_pct: float
    warning_message: Optional[str] = None


class BudgetTracker:
    """
    Track and manage training budget with 90% safety margin.

    Features:
    - Monthly budget limits
    - 90% safety margin (stops at $27 of $30)
    - Detailed cost estimation before training
    - Historical spending tracking
    - Budget warnings and blocks

    GPU Pricing (Modal rates):
    - T4: $0.50/hr, ~100 steps/min
    - A10G: $1.10/hr, ~250 steps/min
    - A100-40GB: $3.00/hr, ~500 steps/min
    - A100-80GB: $4.00/hr, ~600 steps/min
    - H100: $5.00/hr, ~800 steps/min
    """

    # GPU pricing and performance specs
    GPU_SPECS = {
        "T4": {
            "cost_per_hour": 0.50,
            "steps_per_minute": 100,
            "memory_gb": 16,
            "recommended_batch_size": 16,
        },
        "A10G": {
            "cost_per_hour": 1.10,
            "steps_per_minute": 250,
            "memory_gb": 24,
            "recommended_batch_size": 32,
        },
        "A100-40GB": {
            "cost_per_hour": 3.00,
            "steps_per_minute": 500,
            "memory_gb": 40,
            "recommended_batch_size": 64,
        },
        "A100-80GB": {
            "cost_per_hour": 4.00,
            "steps_per_minute": 600,
            "memory_gb": 80,
            "recommended_batch_size": 128,
        },
        "H100": {
            "cost_per_hour": 5.00,
            "steps_per_minute": 800,
            "memory_gb": 80,
            "recommended_batch_size": 128,
        },
    }

    # Safety margin - stop at 90% of budget
    SAFETY_MARGIN = 0.90

    def __init__(
        self,
        budget_file: str = "budget_tracking.json",
        monthly_budget: float = 30.0
    ):
        self.budget_file = Path(budget_file)
        self.monthly_budget = monthly_budget
        self.safety_margin_budget = monthly_budget * self.SAFETY_MARGIN
        self.spending_history: List[TrainingCost] = []
        self._load_history()

    def _load_history(self):
        """Load spending history from file."""
        if self.budget_file.exists():
            try:
                with open(self.budget_file, 'r') as f:
                    data = json.load(f)
                    self.spending_history = [
                        TrainingCost(**record) for record in data.get('history', [])
                    ]
                    self.monthly_budget = data.get('monthly_budget', self.monthly_budget)
                    self.safety_margin_budget = self.monthly_budget * self.SAFETY_MARGIN
            except (json.JSONDecodeError, KeyError):
                self.spending_history = []

    def _save_history(self):
        """Save spending history to file."""
        data = {
            'monthly_budget': self.monthly_budget,
            'safety_margin': self.SAFETY_MARGIN,
            'history': [asdict(record) for record in self.spending_history]
        }
        with open(self.budget_file, 'w') as f:
            json.dump(data, f, indent=2)

    def estimate_cost(
        self,
        max_steps: int,
        gpu_type: str = "A10G",
        model_params_millions: float = 200,
        include_overhead: bool = True
    ) -> CostEstimate:
        """
        Estimate training cost with detailed breakdown.

        Args:
            max_steps: Number of training steps
            gpu_type: Type of GPU to use
            model_params_millions: Model size in millions of parameters
            include_overhead: Include 10% overhead for startup/teardown

        Returns:
            CostEstimate with detailed breakdown
        """
        spec = self.GPU_SPECS.get(gpu_type, self.GPU_SPECS["A10G"])

        # Adjust steps/min based on model size (larger models = slower)
        size_factor = max(0.5, min(2.0, 200 / model_params_millions))
        adjusted_steps_per_min = spec["steps_per_minute"] * size_factor

        # Calculate duration
        steps_per_hour = adjusted_steps_per_min * 60
        hours = max_steps / steps_per_hour

        # Add overhead for container startup, checkpointing, etc.
        if include_overhead:
            hours *= 1.10  # 10% overhead

        # Calculate cost
        cost = hours * spec["cost_per_hour"]

        # Check budget
        current_spending = self.get_current_month_spending()
        remaining = self.monthly_budget - current_spending
        safety_remaining = self.safety_margin_budget - current_spending

        within_budget = cost <= remaining
        within_safety = cost <= safety_remaining

        # Generate warning message if needed
        warning = None
        utilization = (current_spending + cost) / self.monthly_budget * 100

        if not within_safety:
            warning = f"Cost ${cost:.2f} exceeds safety margin. Only ${safety_remaining:.2f} available (90% of budget)."
        elif utilization > 80:
            warning = f"Warning: This will use {utilization:.1f}% of monthly budget."

        return CostEstimate(
            estimated_cost=round(cost, 2),
            estimated_hours=round(hours, 2),
            gpu_type=gpu_type,
            steps=max_steps,
            current_month_spending=round(current_spending, 2),
            remaining_budget=round(remaining, 2),
            safety_margin_budget=round(self.safety_margin_budget, 2),
            within_budget=within_budget,
            within_safety_margin=within_safety,
            budget_utilization_pct=round(utilization, 1),
            warning_message=warning
        )

    @staticmethod
    def estimate_model_params_millions_from_config(config: Dict[str, Any]) -> float:
        """Estimate model size from the active config for more realistic cost checks."""
        model_config = config.get("model", {})
        d_model = model_config.get("d_model", 768)
        num_layers = model_config.get("num_layers", model_config.get("n_layers", 12))
        d_ff = model_config.get("d_ff", d_model * 4)
        vocab_size = model_config.get("vocab_size", 32000)

        embedding_params = vocab_size * d_model
        attention_params = num_layers * 4 * d_model * d_model
        ffn_params = num_layers * 3 * d_model * d_ff
        total_params = embedding_params + attention_params + ffn_params

        return max(1.0, round(total_params / 1_000_000, 1))

    def can_afford(self, cost: float) -> bool:
        """
        Check if budget allows for a training run.
        Uses 90% safety margin.

        Args:
            cost: Estimated cost in USD

        Returns:
            True if within safety margin
        """
        remaining = self.safety_margin_budget - self.get_current_month_spending()
        return cost <= remaining

    def get_current_month_spending(self) -> float:
        """Get total spending for current month."""
        now = datetime.now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        total = 0.0
        for record in self.spending_history:
            record_time = datetime.fromisoformat(record.timestamp)
            if record_time >= month_start:
                total += record.cost_usd

        return total

    def record_spending(
        self,
        dataset: str,
        gpu_type: str,
        duration_hours: float,
        cost_usd: float,
        steps_trained: int,
        phase: Optional[str] = None
    ) -> TrainingCost:
        """
        Record a completed training run.

        Args:
            dataset: Dataset name
            gpu_type: GPU used
            duration_hours: Training duration
            cost_usd: Actual cost
            steps_trained: Steps completed
            phase: Training phase (foundation, expansion, specialization)

        Returns:
            TrainingCost record
        """
        record = TrainingCost(
            timestamp=datetime.now().isoformat(),
            dataset=dataset,
            gpu_type=gpu_type,
            duration_hours=duration_hours,
            cost_usd=cost_usd,
            steps_trained=steps_trained,
            phase=phase
        )
        self.spending_history.append(record)
        self._save_history()
        return record

    def get_spending_summary(self) -> Dict[str, Any]:
        """Get comprehensive spending summary."""
        current_month = self.get_current_month_spending()

        # Last 30 days
        thirty_days_ago = datetime.now() - timedelta(days=30)
        last_30_days = sum(
            r.cost_usd for r in self.spending_history
            if datetime.fromisoformat(r.timestamp) >= thirty_days_ago
        )

        # All time
        all_time = sum(r.cost_usd for r in self.spending_history)

        # By phase
        by_phase = {}
        for r in self.spending_history:
            if r.phase:
                by_phase[r.phase] = by_phase.get(r.phase, 0) + r.cost_usd

        return {
            "current_month": round(current_month, 2),
            "monthly_budget": self.monthly_budget,
            "safety_margin_budget": round(self.safety_margin_budget, 2),
            "remaining": round(self.monthly_budget - current_month, 2),
            "remaining_safe": round(self.safety_margin_budget - current_month, 2),
            "last_30_days": round(last_30_days, 2),
            "all_time": round(all_time, 2),
            "total_runs": len(self.spending_history),
            "by_phase": by_phase,
            "budget_utilization_pct": round(current_month / self.monthly_budget * 100, 1)
        }

    def get_recommended_gpu(
        self,
        max_steps: int,
        model_params_millions: float = 200,
        max_cost: Optional[float] = None
    ) -> str:
        """
        Recommend best GPU for the training job.

        Args:
            max_steps: Number of training steps
            model_params_millions: Model size
            max_cost: Maximum allowed cost (default: remaining safe budget)

        Returns:
            Recommended GPU type
        """
        if max_cost is None:
            max_cost = self.safety_margin_budget - self.get_current_month_spending()

        # Try GPUs from cheapest to most expensive
        gpu_order = ["T4", "A10G", "A100-40GB", "A100-80GB", "H100"]

        for gpu in gpu_order:
            estimate = self.estimate_cost(max_steps, gpu, model_params_millions)
            if estimate.within_safety_margin:
                return gpu

        # If nothing fits, return cheapest
        return "T4"

    def set_monthly_budget(self, budget: float):
        """Update monthly budget."""
        self.monthly_budget = budget
        self.safety_margin_budget = budget * self.SAFETY_MARGIN
        self._save_history()

    def reset_monthly_spending(self):
        """Reset spending for a new month (keeps history)."""
        # History is preserved, but current month calculation will return 0
        # since we filter by current month
        pass
