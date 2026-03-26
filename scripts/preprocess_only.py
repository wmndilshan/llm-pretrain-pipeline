"""
Phase 1 only: download dataset, tokenize, split, save .bin files locally.
No Modal cost. Run this first; then run Modal train (it will use existing data/processed/).

Usage:
  python scripts/preprocess_only.py --config configs/initial_training.yaml
  python scripts/preprocess_only.py --config configs/initial_training.yaml --dataset roneneldan/TinyStories
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_yaml_config


def main():
    p = argparse.ArgumentParser(description="Local preprocessing only (download, tokenize, split, save .bin).")
    p.add_argument("--config", default="configs/initial_training.yaml", help="Config YAML")
    p.add_argument("--dataset", help="Override dataset name from config")
    p.add_argument("--force", action="store_true", help="Force reprocess even if already done")
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    config_path = root / args.config if not Path(args.config).is_absolute() else Path(args.config)
    if not config_path.exists():
        print(f"[FAIL] Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_yaml_config(config_path)

    dataset_name = args.dataset or config["dataset"]["name"]
    config["dataset"]["name"] = dataset_name

    from src.preprocessing import DataPreprocessor
    from src import log_util as log

    log.step("PHASE 1 - Local preprocessing (download, tokenize, split, save)")
    log.info("Dataset: %s" % dataset_name)
    log.info("Output: %s" % config["dataset"]["processed_dir"])

    preprocessor = DataPreprocessor(
        dataset_name=dataset_name,
        cache_dir=config["dataset"]["cache_dir"],
        processed_dir=config["dataset"]["processed_dir"],
        split_ratios=config["dataset"]["split_ratios"],
        max_seq_length=config["dataset"]["max_seq_length"],
        vocab_size=config["model"]["vocab_size"],
        max_samples=config.get("dataset", {}).get("max_samples"),
        tokenizer_backend=config["dataset"].get("tokenizer_backend", "hf"),
    )
    preprocessor.run(force_reprocess=args.force)

    log.ok("Preprocessing done. Run Modal training next with: python scripts/train_with_modal.py --config configs/config.yaml")


if __name__ == "__main__":
    main()
