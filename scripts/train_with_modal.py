#!/usr/bin/env python3
"""
Train with Modal GPU

Command-line interface for Modal-based training.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.modal_training import ModalTrainingOrchestrator
from src.utils.config import load_yaml_config
from src.utils.logging import log_fail, log_info, log_step


def main():
    parser = argparse.ArgumentParser(
        description="Train LLM on Modal GPU infrastructure"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        help='List of datasets to train on sequentially'
    )
    parser.add_argument(
        '--single-dataset',
        type=str,
        help='Train on a single dataset'
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        log_fail(f"Config not found: {config_path}")
        sys.exit(1)
    
    # Create orchestrator
    orchestrator = ModalTrainingOrchestrator(str(config_path))
    
    # Determine datasets
    if args.datasets:
        datasets = args.datasets
    elif args.single_dataset:
        datasets = [args.single_dataset]
    else:
        # Use dataset from config
        config = load_yaml_config(config_path)
        datasets = [config['dataset']['name']]
    
    log_step("MODAL GPU TRAINING")
    log_info(f"Configuration: {config_path}")
    log_info(f"Datasets: {datasets}")
    
    # Train
    if len(datasets) == 1:
        orchestrator.train_on_dataset(datasets[0])
    else:
        previous_checkpoint = None
        for dataset in datasets:
            previous_checkpoint = orchestrator.train_on_dataset(
                dataset,
                previous_checkpoint=previous_checkpoint,
            )
            if not previous_checkpoint:
                sys.exit(1)


if __name__ == "__main__":
    main()
