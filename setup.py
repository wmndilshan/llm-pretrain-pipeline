#!/usr/bin/env python3
"""
Automated Setup Script for LLM Training Pipeline

Sets up the complete environment:
1. Install dependencies
2. Configure Modal
3. Validate setup
4. Run test training
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
from src.utils.logging import log_fail, log_info, log_ok, log_step, log_warn


class Colors:
    """ANSI color codes."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print section header."""
    log_step(''.join(ch for ch in text if ord(ch) < 128).strip())
    return


def print_step(step: str):
    """Print step."""
    log_info(''.join(ch for ch in step if ord(ch) < 128).strip())
    return


def print_success(msg: str):
    """Print success message."""
    log_ok(''.join(ch for ch in msg if ord(ch) < 128).strip())
    return


def print_error(msg: str):
    """Print error message."""
    log_fail(''.join(ch for ch in msg if ord(ch) < 128).strip())
    return


def print_warning(msg: str):
    """Print warning message."""
    log_warn(''.join(ch for ch in msg if ord(ch) < 128).strip())
    return


def run_command(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    """Run shell command."""
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=True,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        if check:
            sys.exit(1)
        return e


def check_python_version():
    """Check Python version."""
    print_step("Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        sys.exit(1)
    
    print_success(f"Python {version.major}.{version.minor}.{version.micro}")


def install_dependencies():
    """Install required dependencies."""
    print_step("Installing dependencies...")
    
    # Main requirements
    print("  Installing main requirements...")
    run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Inference requirements
    if Path("inference_requirements.txt").exists():
        print("  Installing inference requirements...")
        run_command([sys.executable, "-m", "pip", "install", "-r", "inference_requirements.txt"])
    
    # Modal
    print("  Installing Modal...")
    run_command([sys.executable, "-m", "pip", "install", "modal"])
    
    print_success("Dependencies installed")


def configure_modal(token_id: str = None, token_secret: str = None):
    """Configure Modal credentials."""
    print_step("Configuring Modal...")
    
    if token_id and token_secret:
        # Use provided credentials
        cmd = [
            "modal", "token", "set",
            "--token-id", token_id,
            "--token-secret", token_secret
        ]
        result = run_command(cmd, check=False)
        
        if result.returncode == 0:
            print_success("Modal configured")
        else:
            print_error("Modal configuration failed")
            print_warning("You can configure manually with:")
            print(f"  modal token set --token-id <id> --token-secret <secret>")
    else:
        # Check if already configured
        result = run_command(["modal", "token", "show"], check=False)
        
        if result.returncode == 0:
            print_success("Modal already configured")
        else:
            print_warning("Modal not configured")
            print("Configure manually with:")
            print("  modal token set --token-id <id> --token-secret <secret>")


def validate_setup():
    """Validate setup is correct."""
    print_step("Validating setup...")
    
    # Check imports
    try:
        import torch
        print_success(f"PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print_success(f"CUDA {torch.version.cuda} available")
        else:
            print_warning("CUDA not available (CPU only)")
    except ImportError:
        print_error("PyTorch not installed")
        return False
    
    try:
        import modal
        print_success("Modal installed")
    except ImportError:
        print_error("Modal not installed")
        return False
    
    # Check directory structure
    required_dirs = ['src', 'configs', 'scripts', 'data', 'models']
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print_success(f"Directory: {dir_name}/")
        else:
            print_error(f"Missing directory: {dir_name}/")
            return False
    
    # Check key files
    required_files = [
        'main.py',
        'configs/config.yaml',
        'src/tokenizer.py',
        'src/model.py',
        'src/trainer.py'
    ]
    for file_name in required_files:
        if Path(file_name).exists():
            print_success(f"File: {file_name}")
        else:
            print_error(f"Missing file: {file_name}")
            return False
    
    return True


def run_quick_test():
    """Run quick test to verify everything works."""
    print_step("Running quick validation test...")
    
    print("  This will:")
    print("  - Test data preprocessing")
    print("  - Create a tiny test config")
    print("  - Train for 10 steps (local CPU)")
    print("  - Verify pipeline works end-to-end")
    
    # Create tiny test config
    test_config = """
dataset:
  name: "roneneldan/TinyStories"
  split_ratios:
    train: 0.9
    validation: 0.05
    test: 0.05
  max_seq_length: 64
  cache_dir: "./data/cache"
  processed_dir: "./data/processed_test"

model:
  vocab_size: 1000
  d_model: 128
  num_heads: 2
  num_layers: 2
  d_ff: 512
  dropout: 0.1
  max_seq_length: 64

training:
  batch_size: 8
  learning_rate: 3.0e-4
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  grad_clip: 1.0
  warmup_steps: 5
  max_steps: 10
  eval_interval: 5
  save_interval: 10
  log_interval: 1

checkpoint:
  save_dir: "./models/checkpoints_test"
  keep_last_n: 1
  keep_best: true
  resume_from_latest: false

hardware:
  device: "cpu"
  mixed_precision: false
  num_workers: 2
  pin_memory: false

logging:
  log_dir: "./logs_test"
  tensorboard: false
  wandb: false
  wandb_project: null
  wandb_entity: null
"""
    
    # Write test config
    test_config_path = Path("configs/test_config.yaml")
    with open(test_config_path, 'w') as f:
        f.write(test_config)
    
    print(f"  Created test config: {test_config_path}")
    
    # Run test
    print("\n  Running test training (this may take a few minutes)...")
    result = run_command(
        [sys.executable, "main.py", "--config", str(test_config_path), "--local", "--yes"],
        check=False
    )
    
    if result.returncode == 0:
        print_success("Test training completed successfully!")
        
        # Cleanup
        print("\n  Cleaning up test files...")
        import shutil
        if Path("data/processed_test").exists():
            shutil.rmtree("data/processed_test")
        if Path("models/checkpoints_test").exists():
            shutil.rmtree("models/checkpoints_test")
        if Path("logs_test").exists():
            shutil.rmtree("logs_test")
        if test_config_path.exists():
            test_config_path.unlink()
        
        return True
    else:
        print_error("Test training failed")
        print("\nCheck the error messages above")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Setup LLM Training Pipeline"
    )
    parser.add_argument(
        '--modal-token-id',
        type=str,
        help='Modal token ID'
    )
    parser.add_argument(
        '--modal-token-secret',
        type=str,
        help='Modal token secret'
    )
    parser.add_argument(
        '--skip-test',
        action='store_true',
        help='Skip validation test'
    )
    parser.add_argument(
        '--skip-deps',
        action='store_true',
        help='Skip dependency installation'
    )
    
    args = parser.parse_args()
    
    print_header("LLM TRAINING PIPELINE SETUP")
    
    # Step 1: Check Python
    check_python_version()
    
    # Step 2: Install dependencies
    if not args.skip_deps:
        install_dependencies()
    else:
        print_step("Skipping dependency installation")
    
    # Step 3: Configure Modal
    configure_modal(args.modal_token_id, args.modal_token_secret)
    
    # Step 4: Validate
    if not validate_setup():
        print_error("Setup validation failed")
        sys.exit(1)
    
    # Step 5: Run test
    if not args.skip_test:
        if run_quick_test():
            print_header("SETUP COMPLETED SUCCESSFULLY")
            print("\nNext steps:")
            print("1. Edit configs/config.yaml for your use case")
            print("2. Run training: python main.py --config configs/config.yaml")
            print("3. Or use Modal GPU: python scripts/train_with_modal.py --config configs/config.yaml")
            print("4. Deploy: docker compose -f docker/docker-compose.yml up --build -d")
        else:
            print_header("SETUP COMPLETED WITH WARNINGS")
            print("\nThe test failed but dependencies are installed.")
            print("You may need to debug the issue before training.")
    else:
        print_header("SETUP COMPLETED")
        print("\nNext steps:")
        print("1. Edit configs/config.yaml")
        print("2. Run: python main.py --config configs/config.yaml")


if __name__ == "__main__":
    main()
