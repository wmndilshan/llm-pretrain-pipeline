#!/usr/bin/env python3
"""
Training Monitor

Check status of ongoing or completed training runs.
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def format_time_ago(timestamp_str: str) -> str:
    """Format timestamp as 'X hours ago'."""
    try:
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        now = datetime.now()
        delta = now - timestamp
        
        days = delta.days
        hours = delta.seconds // 3600
        minutes = (delta.seconds % 3600) // 60
        
        if days > 0:
            return f"{days}d {hours}h ago"
        elif hours > 0:
            return f"{hours}h {minutes}m ago"
        else:
            return f"{minutes}m ago"
    except:
        return "unknown"


def check_preprocessing_status(processed_dir: Path):
    """Check preprocessing status."""
    state_path = processed_dir / "preprocessing_state.json"
    
    print("\n" + "=" * 70)
    print("PREPROCESSING STATUS")
    print("=" * 70)
    
    if not state_path.exists():
        print("No preprocessing state found")
        print(f"   Expected: {state_path}")
        return
    
    with open(state_path, 'r') as f:
        state = json.load(f)
    
    print(f"Dataset: {state['dataset_name']}")
    print(f"Dataset Hash: {state['dataset_hash']}")
    print(f"Vocab Size: {state['vocab_size']:,}")
    print(f"Max Sequence Length: {state['max_seq_length']}")
    
    print("\nStages:")
    print(f"  Tokenizer Trained: {'Yes' if state['tokenizer_trained'] else 'No'}")
    print(f"  Data Tokenized: {'Yes' if state['data_tokenized'] else 'No'}")
    print(f"  Completed: {'Yes' if state['completed'] else 'No'}")
    
    if state['completed']:
        print("\nDataset Statistics:")
        print(f"  Train: {state['train_samples']:,} samples")
        print(f"  Val:   {state['val_samples']:,} samples")
        print(f"  Test:  {state['test_samples']:,} samples")
        print(f"  Total: {state['train_samples'] + state['val_samples'] + state['test_samples']:,} samples")


def check_training_status(checkpoint_dir: Path):
    """Check training status."""
    metadata_path = checkpoint_dir / "latest_metadata.json"
    
    print("\n" + "=" * 70)
    print("TRAINING STATUS")
    print("=" * 70)
    
    if not metadata_path.exists():
        print("No training metadata found")
        print(f"   Expected: {metadata_path}")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Find all checkpoints
    checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
    checkpoint_steps = []
    for ckpt in checkpoints:
        try:
            step = int(ckpt.stem.split('_')[-1])
            checkpoint_steps.append((step, ckpt))
        except ValueError:
            continue
    
    checkpoint_steps.sort(key=lambda x: x[0])
    
    print(f"Current Step: {metadata['step']:,}")
    print(f"Current Epoch: {metadata['epoch']:.2f}")
    print(f"Train Loss: {metadata['train_loss']:.4f}")
    
    if metadata['val_loss'] is not None:
        print(f"Val Loss: {metadata['val_loss']:.4f}")
        if metadata.get('is_best', False):
            print("   BEST MODEL")
    
    print(f"Learning Rate: {metadata['learning_rate']:.2e}")
    print(f"Last Updated: {metadata['timestamp']} ({format_time_ago(metadata['timestamp'])})")
    
    print(f"\nCheckpoints: {len(checkpoints)} saved")
    if checkpoint_steps:
        print(f"  Earliest: step {checkpoint_steps[0][0]:,}")
        print(f"  Latest:   step {checkpoint_steps[-1][0]:,}")
    
    best_model = checkpoint_dir / "best_model.pt"
    if best_model.exists():
        size_mb = best_model.stat().st_size / (1024 ** 2)
        print(f"\nBest model saved: {size_mb:.2f} MB")


def check_logs(log_dir: Path):
    """Check log directory."""
    print("\n" + "=" * 70)
    print("LOGS")
    print("=" * 70)
    
    if not log_dir.exists():
        print("No logs found")
        return
    
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    
    if event_files:
        print(f"TensorBoard logs: {len(event_files)} files")
        print(f"\nTo view logs, run:")
        print(f"  tensorboard --logdir {log_dir}")
    else:
        print("No TensorBoard event files found")


def check_inference_bundle(bundle_dir: Path):
    """Check Docker inference bundle status."""
    print("\n" + "=" * 70)
    print("DOCKER INFERENCE BUNDLE")
    print("=" * 70)

    model_path = bundle_dir / "best_model.pt"
    tokenizer_path = bundle_dir / "tokenizer.json"
    manifest_path = bundle_dir / "manifest.json"

    if model_path.exists():
        print(f"Model bundle: {model_path}")
    else:
        print(f"Missing model bundle: {model_path}")

    if tokenizer_path.exists():
        print(f"Tokenizer bundle: {tokenizer_path}")
    else:
        print(f"Missing tokenizer bundle: {tokenizer_path}")

    if manifest_path.exists():
        print(f"Manifest: {manifest_path}")
    else:
        print(f"Missing manifest: {manifest_path}")


def main():
    # Default paths
    processed_dir = Path("data/processed")
    checkpoint_dir = Path("models/checkpoints")
    log_dir = Path("logs")
    bundle_dir = Path("models/current")
    
    print("=" * 70)
    print("LLM TRAINING PIPELINE - STATUS MONITOR")
    print("=" * 70)
    
    # Check preprocessing
    check_preprocessing_status(processed_dir)
    
    # Check training
    check_training_status(checkpoint_dir)
    
    # Check logs
    check_logs(log_dir)

    # Check Docker bundle
    check_inference_bundle(bundle_dir)
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
