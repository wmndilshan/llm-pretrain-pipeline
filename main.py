"""
Main Pipeline Orchestrator

Coordinates the entire training pipeline:
1. Data preprocessing (local - free)
2. Upload to Modal volume
3. Training on Modal GPU (paid)
4. Model versioning and checkpointing
5. Budget tracking with 90% safety margin

Usage:
    python main.py                           # Full pipeline (preprocess + Modal train)
    python main.py --preprocess-only         # Only preprocess data locally
    python main.py --local                   # Train locally (CPU/GPU) instead of Modal
    python main.py --estimate-cost           # Estimate training cost without running
"""

import yaml
import argparse
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import load_env, load_yaml_config
from src.utils.logging import log_step, log_info, log_ok, log_warn, log_fail
from src.core.models import ModelConfig, get_model, list_models, ModelVersionManager


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    return load_yaml_config(config_path)


def get_model_config(config: dict) -> ModelConfig:
    """Create ModelConfig from config dictionary."""
    model_cfg = config.get('model', {})
    enhanced_cfg = model_cfg.get('enhanced', {})

    return ModelConfig(
        architecture=model_cfg.get('architecture', 'base'),
        model_name=model_cfg.get('model_name', ''),
        parameter_count=model_cfg.get('parameter_count', 0),
        architecture_family=model_cfg.get('architecture_family', 'decoder-only-transformer'),
        vocab_size=model_cfg.get('vocab_size', 10000),
        d_model=model_cfg.get('d_model', 384),
        num_heads=model_cfg.get('num_heads', 6),
        num_layers=model_cfg.get('num_layers', 6),
        d_ff=model_cfg.get('d_ff', 1536),
        max_seq_length=model_cfg.get('max_seq_length', 256),
        dropout=model_cfg.get('dropout', 0.1),
        version_models=model_cfg.get('version_models', True),
        # Enhanced model options
        use_rotary_embeddings=model_cfg.get('use_rotary_embeddings', enhanced_cfg.get('use_rotary_embeddings', True)),
        use_flash_attention=model_cfg.get('use_flash_attention', enhanced_cfg.get('use_flash_attention', True)),
        use_grouped_query_attention=model_cfg.get('use_grouped_query_attention', enhanced_cfg.get('use_grouped_query_attention', True)),
        gqa_num_kv_heads=model_cfg.get('gqa_num_kv_heads', enhanced_cfg.get('gqa_num_kv_heads', 2)),
        use_rms_norm=model_cfg.get('use_rms_norm', enhanced_cfg.get('use_rms_norm', True)),
        use_swiglu=model_cfg.get('use_swiglu', enhanced_cfg.get('use_swiglu', True)),
        gradient_checkpointing=model_cfg.get('gradient_checkpointing', enhanced_cfg.get('gradient_checkpointing', False)),
        compile_model=model_cfg.get('compile_model', False),
    )


def run_preprocessing(config: dict, force_reprocess: bool = False):
    """Run data preprocessing pipeline locally (free)."""
    from src.pipeline.preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor(
        dataset_name=config['dataset']['name'],
        cache_dir=config['dataset']['cache_dir'],
        processed_dir=config['dataset']['processed_dir'],
        split_ratios=config['dataset']['split_ratios'],
        max_seq_length=config['dataset']['max_seq_length'],
        vocab_size=config['model']['vocab_size'],
        max_samples=config['dataset'].get('max_samples'),
        tokenizer_backend=config['dataset'].get('tokenizer_backend', 'hf'),
    )

    return preprocessor.run(force_reprocess=force_reprocess)


def run_modal_training(config: dict, config_path: str, skip_validation: bool = False, skip_confirm: bool = False):
    """
    Run training on Modal GPU (paid).

    Features:
    - Budget tracking with 90% safety margin
    - Auto batch size optimization for GPU
    - Checkpoint saving with volume commits
    - Cost estimation before training
    """
    from src.orchestration.budget_tracker import BudgetTracker
    from src.pipeline.modal_training import ModalTrainingOrchestrator

    log_step("MODAL GPU TRAINING")

    # Initialize budget tracker
    monthly_budget = config.get('budget', {}).get('monthly_limit', 30.0)
    budget_tracker = BudgetTracker(monthly_budget=monthly_budget)

    # Show budget status
    summary = budget_tracker.get_spending_summary()
    log_info(f"Monthly budget: ${monthly_budget:.2f}")
    log_info(f"Spent this month: ${summary['current_month']:.2f}")
    log_info(f"Remaining (90% safety): ${summary['remaining_safe']:.2f}")

    # Estimate cost
    max_steps = config['training']['max_steps']
    gpu_type = config.get('hardware', {}).get('modal_gpu', 'A10G')
    model_params_millions = budget_tracker.estimate_model_params_millions_from_config(config)
    estimate = budget_tracker.estimate_cost(
        max_steps=max_steps,
        gpu_type=gpu_type,
        model_params_millions=model_params_millions,
    )

    log_info(f"Estimated cost: ${estimate.estimated_cost:.2f}")
    log_info(f"Estimated time: {estimate.estimated_hours:.2f} hours")
    log_info(f"GPU: {gpu_type}")
    log_info(f"Steps: {max_steps:,}")

    if not estimate.within_safety_margin:
        log_fail(f"Insufficient budget! Need ${estimate.estimated_cost:.2f}, have ${estimate.remaining_budget:.2f}")
        log_info("Reduce max_steps or wait for budget reset.")
        return None

    if estimate.warning_message:
        log_warn(estimate.warning_message)

    # Confirm with user
    if not skip_confirm:
        response = input(f"\nProceed with Modal training? (${estimate.estimated_cost:.2f}) [y/N]: ")
        if response.lower() != 'y':
            log_info("Training cancelled.")
            return None

    # Run Modal training
    orchestrator = ModalTrainingOrchestrator(
        config_path=config_path,
        budget_tracker=budget_tracker,
        skip_validation=skip_validation
    )

    dataset_name = config['dataset']['name']
    result = orchestrator.train_on_dataset(
        dataset_name=dataset_name,
        skip_validation=skip_validation,
    )

    return result


def run_local_training(config: dict, data_dir: str):
    """Run training locally on CPU/GPU (for testing)."""
    from src.core.trainer import Trainer

    log_step("LOCAL TRAINING (CPU/GPU)")
    log_warn("Training locally - this is slow! Use Modal for production.")

    model_config = get_model_config(config)
    log_info(f"Model architecture: {model_config.architecture}")

    trainer = Trainer(config, model_config=model_config)
    setup_success = trainer.setup(data_dir)

    if not setup_success:
        return None

    resumed = trainer.resume_from_checkpoint()
    if resumed:
        log_warn("Resuming from checkpoint")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            return None

    trainer.train()
    return trainer.checkpoint_manager.best_checkpoint_path


def estimate_cost(config: dict):
    """Estimate training cost without running."""
    from src.orchestration.budget_tracker import BudgetTracker

    log_step("COST ESTIMATION")

    monthly_budget = config.get('budget', {}).get('monthly_limit', 30.0)
    budget_tracker = BudgetTracker(monthly_budget=monthly_budget)

    max_steps = config['training']['max_steps']

    print("\nCost estimates by GPU type:\n")
    print(f"{'GPU':<15} {'Cost':<10} {'Time':<10} {'Within Budget'}")
    print("-" * 50)

    for gpu in ['T4', 'A10G', 'A100-40GB', 'A100-80GB', 'H100']:
        estimate = budget_tracker.estimate_cost(
            max_steps=max_steps,
            gpu_type=gpu,
            model_params_millions=budget_tracker.estimate_model_params_millions_from_config(config),
        )
        within = "Yes" if estimate.within_safety_margin else "No"
        print(f"{gpu:<15} ${estimate.estimated_cost:<9.2f} {estimate.estimated_hours:<9.2f}h {within}")

    print()
    summary = budget_tracker.get_spending_summary()
    print(f"Monthly budget: ${monthly_budget:.2f}")
    print(f"Spent this month: ${summary['current_month']:.2f}")
    print(f"Remaining (90% safety): ${summary['remaining_safe']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM Training Pipeline with Modal GPU Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                           # Full pipeline (preprocess + Modal train)
    python main.py --preprocess-only         # Only preprocess data locally (free)
    python main.py --local                   # Train locally instead of Modal
    python main.py --estimate-cost           # Show cost estimates
    python main.py --model enhanced          # Use enhanced model architecture
    python main.py --steps 10000             # Override max training steps
    python main.py --gpu A100-40GB           # Use specific GPU type

Modal Training (recommended):
    - Preprocesses data locally (free)
    - Uploads to Modal volume
    - Trains on cloud GPU (A10G default)
    - Auto batch size optimization
    - Budget tracking with 90% safety margin
        """
    )

    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, choices=list_models(),
                        help='Model architecture to use')
    parser.add_argument('--list-models', action='store_true',
                        help='List available model architectures')
    parser.add_argument('--preprocess-only', action='store_true',
                        help='Only preprocess data (no training)')
    parser.add_argument('--local', action='store_true',
                        help='Train locally instead of Modal (slow)')
    parser.add_argument('--estimate-cost', action='store_true',
                        help='Show cost estimates without training')
    parser.add_argument('--force-reprocess', action='store_true',
                        help='Force reprocessing of data')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip pre-training validation')
    parser.add_argument('--steps', type=int,
                        help='Override max training steps')
    parser.add_argument('--gpu', type=str, choices=['T4', 'A10G', 'A100-40GB', 'A100-80GB', 'H100'],
                        help='GPU type for Modal training')
    parser.add_argument('--clean', action='store_true',
                        help='Clean processed data and checkpoints')
    parser.add_argument('--list-versions', action='store_true',
                        help='List saved model versions')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Skip confirmation prompts')

    args = parser.parse_args()

    # Load environment
    load_env()

    # List models
    if args.list_models:
        print("\nAvailable model architectures:")
        for arch in list_models():
            if arch == "base":
                desc = "GPT-2 style transformer"
            elif arch == "enhanced":
                desc = "Modern (RoPE, GQA, Flash Attention)"
            else:
                desc = "Merged production profile architecture"
            print(f"  - {arch}: {desc}")
        return

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        log_fail(f"Config not found: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))

    # Apply overrides
    if args.model:
        config['model']['architecture'] = args.model
    if args.steps:
        config['training']['max_steps'] = args.steps
    if args.gpu:
        config.setdefault('hardware', {})['modal_gpu'] = args.gpu

    # List versions
    if args.list_versions:
        model_dir = config.get('checkpoint', {}).get('save_dir', 'models/checkpoints')
        version_mgr = ModelVersionManager(model_dir)
        versions = version_mgr.list_versions()
        if versions:
            print("\nSaved model versions:")
            for v in versions:
                metrics = v.get('metrics', {})
                val_loss = metrics.get('val_loss', 'N/A')
                print(f"  v{v['version']}: {v['architecture']} (val_loss: {val_loss})")
        else:
            print("\nNo saved model versions found.")
        return

    # Clean
    if args.clean:
        from src.pipeline.preprocessing import DataPreprocessor
        from src.pipeline.checkpoint import CheckpointManager

        log_step("Cleaning data and checkpoints")

        preprocessor = DataPreprocessor(
            dataset_name=config['dataset']['name'],
            cache_dir=config['dataset']['cache_dir'],
            processed_dir=config['dataset']['processed_dir'],
            split_ratios=config['dataset']['split_ratios'],
            max_seq_length=config['dataset']['max_seq_length'],
            vocab_size=config['model']['vocab_size'],
            tokenizer_backend=config['dataset'].get('tokenizer_backend', 'hf'),
        )
        preprocessor.clean(keep_tokenizer=False)

        ckpt_manager = CheckpointManager(config['checkpoint']['save_dir'])
        ckpt_manager.clean_all_checkpoints()

        log_ok("Cleanup completed")
        return

    # Estimate cost
    if args.estimate_cost:
        estimate_cost(config)
        return

    # Run pipeline
    log_step("LLM TRAINING PIPELINE")
    log_info(f"Config: {config_path}")
    log_info(f"Dataset: {config['dataset']['name']}")
    log_info(f"Model: {config['model'].get('architecture', 'base')}")
    log_info(f"Max steps: {config['training']['max_steps']:,}")

    # Stage 1: Preprocessing (local, free)
    log_step("STAGE 1: PREPROCESSING (LOCAL)")
    train_path, val_path, test_path = run_preprocessing(config, args.force_reprocess)
    data_dir = Path(train_path).parent

    if args.preprocess_only:
        log_ok("Preprocessing complete. Skipping training.")
        return

    # Stage 2: Training
    if args.local:
        # Local training (CPU/GPU)
        result = run_local_training(config, str(data_dir))
    else:
        # Modal GPU training (recommended)
        result = run_modal_training(config, str(config_path), args.skip_validation, args.yes)

    if result:
        log_step("PIPELINE COMPLETED SUCCESSFULLY")
        log_info(f"Model saved: {result}")
    else:
        log_warn("Pipeline finished without producing a model")


if __name__ == "__main__":
    main()
