#!/usr/bin/env python3
"""
Transfer Learning Manager Script

View and manage transfer learning across multiple datasets.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transfer_learning import TransferLearningManager


def main():
    parser = argparse.ArgumentParser(description="Transfer Learning Manager")
    parser.add_argument(
        '--transfer-dir',
        type=str,
        default='models/transfer_learning',
        help='Transfer learning directory'
    )
    parser.add_argument(
        '--show-history',
        action='store_true',
        help='Show training history'
    )
    parser.add_argument(
        '--clean-old',
        type=int,
        metavar='N',
        help='Keep only last N models'
    )

    args = parser.parse_args()

    # Create manager
    manager = TransferLearningManager(args.transfer_dir)

    if args.show_history:
        # Show training history
        manager.print_training_history()

        # Show summary
        summary = manager.get_training_summary()

        if summary['total_datasets'] > 0:
            print("\nQuick Summary:")
            print(f"  Total datasets: {summary['total_datasets']}")
            print(f"  Lineage: {' -> '.join(summary['training_lineage'])}")
            print(f"  Latest model: {summary['latest_model']}")

            print("\nValidation Loss Progression:")
            for entry in summary['history']:
                print(f"  {entry['dataset']}: {entry['best_val_loss']:.4f}")

    elif args.clean_old:
        # Clean old models
        manager.clean_old_models(keep_last_n=args.clean_old)

    else:
        # Default: show history
        manager.print_training_history()


if __name__ == "__main__":
    main()
