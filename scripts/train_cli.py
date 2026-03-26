#!/usr/bin/env python3
"""
Interactive Training CLI

Professional interactive CLI for configuring and launching training runs.

Usage:
    python scripts/train_cli.py

Requirements:
    pip install questionary
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import questionary
    from questionary import Style
except ImportError:
    print("Installing questionary for interactive CLI...")
    os.system(f"{sys.executable} -m pip install questionary")
    import questionary
    from questionary import Style

from src.utils.config import load_env, load_yaml_config
from src.orchestration.budget_tracker import BudgetTracker


custom_style = Style([
    ('qmark', 'fg:#0b7285 bold'),
    ('question', 'bold'),
    ('answer', 'fg:#0b7285 bold'),
    ('pointer', 'fg:#0b7285 bold'),
    ('highlighted', 'fg:#0b7285 bold'),
    ('selected', 'fg:#0b7285'),
    ('separator', 'fg:#6c6c6c'),
    ('instruction', 'fg:#6c6c6c'),
    ('text', ''),
])


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    """Print a clean startup banner."""
    banner = """
\033[96m
  =========================================================
    LLM Training Pipeline
    Modal GPU | Budget Tracking | Auto-scaling
  =========================================================
\033[0m"""
    print(banner)


def get_budget_status(budget_tracker: BudgetTracker) -> str:
    """Get formatted budget status."""
    summary = budget_tracker.get_spending_summary()
    used_pct = (summary['current_month'] / summary['monthly_budget']) * 100

    if used_pct < 50:
        color = '\033[92m'
    elif used_pct < 80:
        color = '\033[93m'
    else:
        color = '\033[91m'

    return (
        f"Budget: {color}${summary['remaining_safe']:.2f}\033[0m available "
        f"(${summary['current_month']:.2f}/${summary['monthly_budget']:.2f} used)"
    )


def format_gpu_choice(gpu: dict, estimate, within_budget: bool) -> str:
    """Format GPU choice for display."""
    status = "\033[92m[OK]\033[0m" if within_budget else "\033[91m[OVER]\033[0m"
    cost_color = "\033[92m" if within_budget else "\033[91m"
    recommended = " \033[93m[Recommended]\033[0m" if gpu['name'] == 'A10G' else ""

    return (
        f"{status} {gpu['name']:<12} "
        f"\033[2m{gpu['vram']:<8}\033[0m "
        f"\033[94m{gpu['speed']:<6}\033[0m "
        f"{cost_color}${estimate.estimated_cost:.2f}\033[0m "
        f"\033[2m({estimate.estimated_hours:.1f}h)\033[0m"
        f"{recommended}"
    )


def main():
    """Main interactive CLI."""
    load_env()
    clear_screen()
    print_banner()

    config_path = "configs/config.yaml"
    if not Path(config_path).exists():
        print("\033[91m[FAIL] configs/config.yaml not found\033[0m")
        return

    config = load_yaml_config(config_path)

    monthly_budget = config.get('budget', {}).get('monthly_limit', 30.0)
    budget_tracker = BudgetTracker(monthly_budget=monthly_budget)

    print(f"  {get_budget_status(budget_tracker)}\n")

    model_choices = [
        questionary.Choice(
            title="enhanced  \033[2m(Modern: RoPE, GQA, Flash Attention)\033[0m \033[93m[Recommended]\033[0m",
            value="enhanced"
        ),
        questionary.Choice(
            title="base      \033[2m(GPT-2 style transformer)\033[0m",
            value="base"
        ),
    ]

    model = questionary.select(
        "Select model architecture:",
        choices=model_choices,
        style=custom_style,
        instruction="(Use arrow keys)"
    ).ask()

    if not model:
        print("\n\033[2mCancelled.\033[0m")
        return

    steps_choices = [
        questionary.Choice(title="1,000    \033[2m(Quick test, ~7 min)\033[0m", value=1000),
        questionary.Choice(title="5,000    \033[2m(Short training, ~30 min)\033[0m", value=5000),
        questionary.Choice(title="10,000   \033[2m(Medium training, ~1 hour)\033[0m \033[93m[Recommended]\033[0m", value=10000),
        questionary.Choice(title="50,000   \033[2m(Full training, ~5 hours)\033[0m", value=50000),
        questionary.Choice(title="Custom   \033[2m(Enter your own)\033[0m", value="custom"),
    ]

    steps = questionary.select(
        "Select training steps:",
        choices=steps_choices,
        style=custom_style,
        instruction="(Use arrow keys)"
    ).ask()

    if not steps:
        print("\n\033[2mCancelled.\033[0m")
        return

    if steps == "custom":
        steps = questionary.text(
            "Enter number of steps:",
            style=custom_style,
            validate=lambda x: x.isdigit() and int(x) > 0
        ).ask()
        if not steps:
            print("\n\033[2mCancelled.\033[0m")
            return
        steps = int(steps)

    gpus = [
        {"name": "T4", "vram": "16GB", "speed": "1x"},
        {"name": "A10G", "vram": "24GB", "speed": "2.5x"},
        {"name": "A100-40GB", "vram": "40GB", "speed": "5x"},
        {"name": "A100-80GB", "vram": "80GB", "speed": "6x"},
        {"name": "H100", "vram": "80GB", "speed": "10x"},
    ]

    gpu_choices = []
    for gpu in gpus:
        estimate = budget_tracker.estimate_cost(max_steps=steps, gpu_type=gpu["name"])
        within_budget = estimate.within_safety_margin
        gpu_choices.append(
            questionary.Choice(
                title=format_gpu_choice(gpu, estimate, within_budget),
                value=(gpu["name"], estimate)
            )
        )

    print("\n  \033[2m[OK] = Within budget   [OVER] = Over budget\033[0m\n")

    gpu_result = questionary.select(
        "Select GPU:",
        choices=gpu_choices,
        style=custom_style,
        instruction="(Use arrow keys)"
    ).ask()

    if not gpu_result:
        print("\n\033[2mCancelled.\033[0m")
        return

    gpu_name, estimate = gpu_result

    if not estimate.within_safety_margin:
        proceed = questionary.confirm(
            f"\033[91mThis exceeds your safe budget. "
            f"(Need ${estimate.estimated_cost:.2f}, have ${estimate.remaining_budget:.2f})\033[0m\n"
            "Continue anyway?",
            default=False,
            style=custom_style
        ).ask()

        if not proceed:
            print("\n\033[2mCancelled.\033[0m")
            return

    print("\n" + "-" * 50)
    print("\033[1m  Configuration Summary\033[0m")
    print("-" * 50)
    print(f"  Model:      \033[97m{model}\033[0m")
    print(f"  GPU:        \033[97m{gpu_name}\033[0m")
    print(f"  Steps:      \033[97m{steps:,}\033[0m")
    print(f"  Est. Cost:  \033[92m${estimate.estimated_cost:.2f}\033[0m")
    print(f"  Est. Time:  \033[94m{estimate.estimated_hours:.1f} hours\033[0m")
    print("-" * 50 + "\n")

    confirm = questionary.confirm(
        "Start training?",
        default=True,
        style=custom_style
    ).ask()

    if not confirm:
        print("\n\033[2mCancelled.\033[0m")
        return

    print("\n\033[92m\033[1mStarting training...\033[0m\n")

    config['model']['architecture'] = model
    config['training']['max_steps'] = steps
    config.setdefault('hardware', {})['modal_gpu'] = gpu_name

    from src.pipeline.modal_training import ModalTrainingOrchestrator

    orchestrator = ModalTrainingOrchestrator(
        config_path=config_path,
        budget_tracker=budget_tracker,
        skip_validation=False
    )
    orchestrator.config = config

    dataset_name = config['dataset']['name']
    result = orchestrator.train_on_dataset(
        dataset_name=dataset_name,
        skip_validation=False,
    )

    if result:
        print("\n\033[92m\033[1m[OK] Training complete.\033[0m")
        print(f"  Model saved: {result}")
    else:
        print("\n\033[93mTraining finished without producing a model.\033[0m")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n\033[2mInterrupted.\033[0m")
