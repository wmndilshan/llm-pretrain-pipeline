#!/usr/bin/env python3
"""
Interactive Training CLI

Professional interactive CLI for configuring and launching training runs.

Usage:
    python scripts/train_interactive.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_env, load_yaml_config
from src.orchestration.budget_tracker import BudgetTracker


class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RED = '\033[91m'
    WHITE = '\033[97m'


def colorize(text: str, color: str) -> str:
    """Apply color to text."""
    return f"{color}{text}{Colors.RESET}"


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print a clean startup header."""
    header = """
    ===============================================================
      LLM Training Pipeline
      Modal GPU | Cost-aware configuration | Guided launch
    ===============================================================
    """
    print(colorize(header, Colors.CYAN))


def print_budget_status(budget_tracker: BudgetTracker):
    """Print current budget status."""
    summary = budget_tracker.get_spending_summary()

    print(colorize("\n  Budget Status", Colors.BOLD + Colors.WHITE))
    print(colorize("  " + "-" * 40, Colors.DIM))

    used_pct = (summary['current_month'] / summary['monthly_budget']) * 100
    bar_width = 30
    filled = int(bar_width * used_pct / 100)
    bar = "#" * filled + "-" * (bar_width - filled)

    color = Colors.GREEN if used_pct < 50 else (Colors.YELLOW if used_pct < 80 else Colors.RED)

    monthly_budget_text = colorize(f"${summary['monthly_budget']:.2f}", Colors.WHITE)
    current_month_text = colorize(f"${summary['current_month']:.2f}", color)
    remaining_safe_text = colorize(f"${summary['remaining_safe']:.2f}", Colors.GREEN)

    print(f"  Monthly Budget: {monthly_budget_text}")
    print(f"  Used:           {current_month_text} ({used_pct:.1f}%)")
    print(f"  Available:      {remaining_safe_text} (90% safety)")
    print(f"  [{colorize(bar, color)}]")


def select_gpu(budget_tracker: BudgetTracker, max_steps: int) -> tuple:
    """Interactive GPU selection with cost estimates."""
    gpus = [
        {"name": "T4", "vram": "16GB", "speed": "1x", "hourly": 0.50},
        {"name": "A10G", "vram": "24GB", "speed": "2.5x", "hourly": 1.10},
        {"name": "A100-40GB", "vram": "40GB", "speed": "5x", "hourly": 3.00},
        {"name": "A100-80GB", "vram": "80GB", "speed": "6x", "hourly": 4.00},
        {"name": "H100", "vram": "80GB", "speed": "10x", "hourly": 5.00},
    ]

    print(colorize("\n  Select GPU", Colors.BOLD + Colors.WHITE))
    print(colorize("  " + "-" * 60, Colors.DIM))
    print()

    for i, gpu in enumerate(gpus):
        estimate = budget_tracker.estimate_cost(max_steps=max_steps, gpu_type=gpu["name"])
        if estimate.within_safety_margin:
            status = colorize("[OK]", Colors.GREEN)
            cost_color = Colors.GREEN
        else:
            status = colorize("[OVER]", Colors.RED)
            cost_color = Colors.RED

        recommended = colorize(" [Recommended]", Colors.YELLOW) if gpu["name"] == "A10G" else ""

        print(
            f"  {status} {colorize(f'[{i+1}]', Colors.CYAN)} {colorize(gpu['name'], Colors.WHITE):<12} "
            f"{colorize(gpu['vram'], Colors.DIM):<8} "
            f"{colorize(gpu['speed'], Colors.BLUE):<6} "
            f"{colorize(f'${estimate.estimated_cost:.2f}', cost_color):<10} "
            f"{colorize(f'{estimate.estimated_hours:.1f}h', Colors.DIM)}"
            f"{recommended}"
        )

    print()
    print(colorize("  [OK] = Within budget   [OVER] = Over budget", Colors.DIM))
    print()

    while True:
        try:
            choice = input(colorize("  Select GPU [1-5] (default: 2): ", Colors.CYAN))
            if choice == "":
                choice = 2
            else:
                choice = int(choice)

            if 1 <= choice <= 5:
                selected = gpus[choice - 1]
                estimate = budget_tracker.estimate_cost(max_steps=max_steps, gpu_type=selected["name"])
                return selected["name"], estimate

            print(colorize("  Invalid choice. Please enter 1-5.", Colors.RED))
        except ValueError:
            print(colorize("  Invalid input. Please enter a number.", Colors.RED))


def select_model():
    """Interactive model architecture selection."""
    models = [
        {"name": "base", "desc": "GPT-2 style transformer", "params": "~15M"},
        {"name": "enhanced", "desc": "Modern (RoPE, GQA, Flash Attention)", "params": "~15M"},
    ]

    print(colorize("\n  Select Model Architecture", Colors.BOLD + Colors.WHITE))
    print(colorize("  " + "-" * 50, Colors.DIM))
    print()

    for i, model in enumerate(models):
        recommended = colorize(" [Recommended]", Colors.YELLOW) if model["name"] == "enhanced" else ""
        print(
            f"  {colorize(f'[{i+1}]', Colors.CYAN)} {colorize(model['name'], Colors.WHITE):<12} "
            f"{colorize(model['desc'], Colors.DIM):<40} "
            f"{colorize(model['params'], Colors.BLUE)}"
            f"{recommended}"
        )

    print()

    while True:
        try:
            choice = input(colorize("  Select model [1-2] (default: 2): ", Colors.CYAN))
            if choice == "":
                choice = 2
            else:
                choice = int(choice)

            if 1 <= choice <= 2:
                return models[choice - 1]["name"]

            print(colorize("  Invalid choice. Please enter 1-2.", Colors.RED))
        except ValueError:
            print(colorize("  Invalid input. Please enter a number.", Colors.RED))


def select_steps():
    """Interactive training steps selection."""
    presets = [
        {"steps": 1000, "desc": "Quick test", "time": "~7 min"},
        {"steps": 5000, "desc": "Short training", "time": "~30 min"},
        {"steps": 10000, "desc": "Medium training", "time": "~1 hour"},
        {"steps": 50000, "desc": "Full training", "time": "~5 hours"},
        {"steps": 0, "desc": "Custom", "time": "varies"},
    ]

    print(colorize("\n  Select Training Steps", Colors.BOLD + Colors.WHITE))
    print(colorize("  " + "-" * 50, Colors.DIM))
    print()

    for i, preset in enumerate(presets):
        if preset["steps"] > 0:
            step_text = colorize(f"{preset['steps']:,}", Colors.WHITE)
            print(
                f"  {colorize(f'[{i+1}]', Colors.CYAN)} {step_text:<12} "
                f"{colorize(preset['desc'], Colors.DIM):<20} "
                f"{colorize(preset['time'], Colors.BLUE)}"
            )
        else:
            print(
                f"  {colorize(f'[{i+1}]', Colors.CYAN)} {colorize('Custom', Colors.WHITE):<12} "
                f"{colorize(preset['desc'], Colors.DIM)}"
            )

    print()

    while True:
        try:
            choice = input(colorize("  Select preset [1-5] (default: 3): ", Colors.CYAN))
            if choice == "":
                choice = 3
            else:
                choice = int(choice)

            if 1 <= choice <= 5:
                if choice == 5:
                    custom = input(colorize("  Enter number of steps: ", Colors.CYAN))
                    return int(custom)
                return presets[choice - 1]["steps"]

            print(colorize("  Invalid choice. Please enter 1-5.", Colors.RED))
        except ValueError:
            print(colorize("  Invalid input. Please enter a number.", Colors.RED))


def confirm_training(config: dict):
    """Show training configuration and confirm."""
    print(colorize("\n  Training Configuration", Colors.BOLD + Colors.WHITE))
    print(colorize("  " + "=" * 50, Colors.CYAN))
    print()
    steps_text = colorize(f"{config['steps']:,}", Colors.WHITE)
    cost_text = colorize(f"${config['cost']:.2f}", Colors.GREEN)
    time_text = colorize(f"{config['time']:.1f} hours", Colors.BLUE)
    print(f"  {colorize('Model:', Colors.DIM):<20} {colorize(config['model'], Colors.WHITE)}")
    print(f"  {colorize('GPU:', Colors.DIM):<20} {colorize(config['gpu'], Colors.WHITE)}")
    print(f"  {colorize('Steps:', Colors.DIM):<20} {steps_text}")
    print(f"  {colorize('Est. Cost:', Colors.DIM):<20} {cost_text}")
    print(f"  {colorize('Est. Time:', Colors.DIM):<20} {time_text}")
    print()
    print(colorize("  " + "=" * 50, Colors.CYAN))
    print()

    confirm = input(colorize("  Start training? [Y/n]: ", Colors.YELLOW))
    return confirm.lower() in ('', 'y', 'yes')


def run_training(model: str, gpu: str, steps: int, config_path: str):
    """Launch the training."""
    from src.pipeline.modal_training import ModalTrainingOrchestrator
    from src.orchestration.budget_tracker import BudgetTracker

    print(colorize("\n  Starting training...", Colors.GREEN + Colors.BOLD))
    print()

    config = load_yaml_config(config_path)
    config['model']['architecture'] = model
    config['training']['max_steps'] = steps
    config.setdefault('hardware', {})['modal_gpu'] = gpu

    budget_tracker = BudgetTracker(monthly_budget=config.get('budget', {}).get('monthly_limit', 30.0))

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

    return result


def main():
    """Main interactive CLI."""
    load_env()

    clear_screen()
    print_header()

    config_path = "configs/config.yaml"
    config = load_yaml_config(config_path)

    monthly_budget = config.get('budget', {}).get('monthly_limit', 30.0)
    budget_tracker = BudgetTracker(monthly_budget=monthly_budget)

    print_budget_status(budget_tracker)

    model = select_model()
    steps = select_steps()
    gpu, estimate = select_gpu(budget_tracker, steps)

    training_config = {
        'model': model,
        'gpu': gpu,
        'steps': steps,
        'cost': estimate.estimated_cost,
        'time': estimate.estimated_hours,
    }

    if not estimate.within_safety_margin:
        print(colorize("\n  WARNING: This exceeds your safe budget.", Colors.RED + Colors.BOLD))
        print(colorize(f"  Estimated cost: ${estimate.estimated_cost:.2f}", Colors.RED))
        print(colorize(f"  Available: ${estimate.remaining_budget:.2f}", Colors.RED))

        proceed = input(colorize("\n  Continue anyway? [y/N]: ", Colors.YELLOW))
        if proceed.lower() != 'y':
            print(colorize("\n  Training cancelled.", Colors.DIM))
            return

    if confirm_training(training_config):
        result = run_training(model, gpu, steps, config_path)

        if result:
            print(colorize("\n  [OK] Training complete.", Colors.GREEN + Colors.BOLD))
            print(colorize(f"  Model saved: {result}", Colors.GREEN))
        else:
            print(colorize("\n  Training finished without producing a model.", Colors.YELLOW))
    else:
        print(colorize("\n  Training cancelled.", Colors.DIM))


if __name__ == "__main__":
    main()
