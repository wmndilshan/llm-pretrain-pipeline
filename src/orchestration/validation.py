"""
Pre-Training Validation System

Validates everything before starting expensive GPU training:
- Environment and dependencies
- GPU availability and memory
- Dataset accessibility
- Tokenizer functionality
- Model architecture
- Budget constraints
- Checkpoint compatibility

Prevents wasted GPU costs from configuration errors.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..utils.config import load_yaml_config
from ..utils.logging import log_fail, log_info, log_ok, log_step, log_warn


class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    check_name: str
    passed: bool
    level: ValidationLevel
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report."""
    passed: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: int
    results: List[ValidationResult] = field(default_factory=list)
    can_proceed: bool = True  # False if any CRITICAL or ERROR
    estimated_cost: Optional[float] = None

    def add_result(self, result: ValidationResult):
        """Add a validation result."""
        self.results.append(result)
        self.total_checks += 1
        if result.passed:
            self.passed_checks += 1
        else:
            self.failed_checks += 1
            if result.level in (ValidationLevel.ERROR, ValidationLevel.CRITICAL):
                self.can_proceed = False
        if result.level == ValidationLevel.WARNING:
            self.warnings += 1

    def summary(self) -> str:
        """Generate summary string."""
        status = "PASSED" if self.can_proceed else "FAILED"
        return (
            f"Validation {status}: "
            f"{self.passed_checks}/{self.total_checks} checks passed, "
            f"{self.warnings} warnings"
        )


class PreTrainingValidator:
    """
    Validates training configuration before GPU deployment.

    Checks:
    1. Environment: Python version, dependencies, tokens
    2. GPU: Availability, memory requirements
    3. Data: Dataset access, tokenizer
    4. Model: Architecture compatibility
    5. Budget: Cost estimation and limits
    6. Checkpoints: Previous checkpoint compatibility
    """

    def __init__(
        self,
        config: Dict[str, Any],
        budget_tracker=None,
        verbose: bool = True
    ):
        self.config = config
        self.budget_tracker = budget_tracker
        self.verbose = verbose
        self.report = ValidationReport(
            passed=True,
            total_checks=0,
            passed_checks=0,
            failed_checks=0,
            warnings=0,
        )

    def _log(self, message: str):
        """Log if verbose."""
        if self.verbose:
            log_info(message)

    def validate_all(self) -> ValidationReport:
        """Run all validation checks."""
        if self.verbose:
            log_step("PRE-TRAINING VALIDATION")

        # Run all checks
        self._validate_environment()
        self._validate_dependencies()
        self._validate_tokens()
        self._validate_gpu()
        self._validate_dataset()
        self._validate_tokenizer()
        self._validate_model_config()
        self._validate_budget()
        self._validate_checkpoint()
        self._validate_output_paths()

        # Update overall status
        self.report.passed = self.report.can_proceed

        # Print summary
        if self.verbose:
            log_step("VALIDATION SUMMARY")
            log_info(self.report.summary())

        if not self.report.can_proceed:
            log_fail("Cannot proceed with training. Fix errors above.")

        return self.report

    def _add_check(
        self,
        name: str,
        passed: bool,
        level: ValidationLevel,
        message: str,
        details: Optional[Dict] = None,
        suggestion: Optional[str] = None
    ):
        """Add a check result."""
        result = ValidationResult(
            check_name=name,
            passed=passed,
            level=level,
            message=message,
            details=details,
            suggestion=suggestion,
        )
        self.report.add_result(result)

        # Log result
        if self.verbose:
            if passed:
                log_ok(f"{name}: {message}")
            elif level == ValidationLevel.WARNING:
                log_warn(f"{name}: {message}")
            else:
                log_fail(f"{name}: {message}")
        if suggestion and not passed:
            log_info(f"Suggestion: {suggestion}")

    def _validate_environment(self):
        """Validate Python environment."""
        self._log("Checking environment...")

        # Python version
        py_version = sys.version_info
        py_ok = py_version >= (3, 9)
        self._add_check(
            "Python Version",
            py_ok,
            ValidationLevel.ERROR if not py_ok else ValidationLevel.INFO,
            f"Python {py_version.major}.{py_version.minor}.{py_version.micro}",
            suggestion="Requires Python 3.9+" if not py_ok else None
        )

        # Platform
        import platform
        self._add_check(
            "Platform",
            True,
            ValidationLevel.INFO,
            f"{platform.system()} {platform.release()}"
        )

    def _validate_dependencies(self):
        """Validate required dependencies."""
        self._log("Checking dependencies...")

        required = ["torch", "transformers", "tokenizers", "datasets", "yaml"]
        optional = ["modal", "wandb", "tensorboard"]

        for pkg in required:
            try:
                __import__(pkg)
                self._add_check(f"Package: {pkg}", True, ValidationLevel.INFO, "installed")
            except ImportError:
                self._add_check(
                    f"Package: {pkg}",
                    False,
                    ValidationLevel.ERROR,
                    "not installed",
                    suggestion=f"pip install {pkg}"
                )

        for pkg in optional:
            try:
                __import__(pkg)
                self._add_check(f"Optional: {pkg}", True, ValidationLevel.INFO, "installed")
            except ImportError:
                self._add_check(
                    f"Optional: {pkg}",
                    True,
                    ValidationLevel.WARNING,
                    "not installed (optional)",
                    suggestion=f"pip install {pkg}"
                )

    def _validate_tokens(self):
        """Validate API tokens."""
        self._log("Checking API tokens...")

        # HuggingFace token
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self._add_check(
            "HuggingFace Token",
            bool(hf_token),
            ValidationLevel.WARNING if not hf_token else ValidationLevel.INFO,
            "configured" if hf_token else "not set",
            suggestion="Set HF_TOKEN for private datasets" if not hf_token else None
        )

        # Modal token
        modal_token = os.environ.get("MODAL_TOKEN_ID")
        self._add_check(
            "Modal Token",
            bool(modal_token),
            ValidationLevel.WARNING if not modal_token else ValidationLevel.INFO,
            "configured" if modal_token else "not set",
            suggestion="Run 'modal token new' to authenticate" if not modal_token else None
        )

    def _validate_gpu(self):
        """Validate GPU availability."""
        self._log("Checking GPU...")

        try:
            import torch
            cuda_available = torch.cuda.is_available()

            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                self._add_check(
                    "CUDA Available",
                    True,
                    ValidationLevel.INFO,
                    f"{gpu_name} ({gpu_memory:.1f} GB)"
                )

                # Check memory requirements
                model_config = self.config.get('model', {})
                estimated_memory = self._estimate_memory_gb(model_config)
                memory_ok = gpu_memory >= estimated_memory

                self._add_check(
                    "GPU Memory",
                    memory_ok,
                    ValidationLevel.WARNING if not memory_ok else ValidationLevel.INFO,
                    f"Required: ~{estimated_memory:.1f} GB, Available: {gpu_memory:.1f} GB",
                    suggestion="Consider reducing batch size or model size" if not memory_ok else None
                )
            else:
                # For Modal deployment, local GPU is not required
                self._add_check(
                    "CUDA Available",
                    True,
                    ValidationLevel.WARNING,
                    "No local GPU (will use Modal cloud GPU)",
                )
        except Exception as e:
            self._add_check(
                "GPU Check",
                True,
                ValidationLevel.WARNING,
                f"Could not check GPU: {e}"
            )

    def _estimate_memory_gb(self, model_config: Dict) -> float:
        """Estimate GPU memory requirements."""
        d_model = model_config.get('d_model', 768)
        n_layers = model_config.get('num_layers', model_config.get('n_layers', 12))
        vocab_size = model_config.get('vocab_size', 32000)
        batch_size = self.config.get('training', {}).get('batch_size', 32)
        seq_len = model_config.get('max_seq_length', model_config.get('max_seq_len', 1024))

        # Rough estimation
        params = vocab_size * d_model + n_layers * (4 * d_model * d_model + 8 * d_model * d_model)
        params_gb = params * 4 / (1024**3)  # float32

        # Activations and gradients
        activations_gb = batch_size * seq_len * d_model * n_layers * 4 / (1024**3)

        # Total with overhead
        return (params_gb * 3 + activations_gb) * 1.2

    def _validate_dataset(self):
        """Validate dataset accessibility."""
        self._log("Checking dataset...")

        dataset_config = self.config.get('dataset', {})
        dataset_name = dataset_config.get('name', 'roneneldan/TinyStories')

        try:
            from datasets import load_dataset_builder

            # Try to get dataset info without downloading
            builder = load_dataset_builder(dataset_name)

            self._add_check(
                "Dataset Access",
                True,
                ValidationLevel.INFO,
                f"'{dataset_name}' accessible"
            )

            # Check if streaming is configured for large datasets
            if 'openwebtext' in dataset_name.lower() or 'pile' in dataset_name.lower():
                streaming = dataset_config.get('streaming', False)
                self._add_check(
                    "Large Dataset Mode",
                    streaming,
                    ValidationLevel.WARNING if not streaming else ValidationLevel.INFO,
                    "streaming enabled" if streaming else "streaming disabled",
                    suggestion="Enable streaming for large datasets to save memory" if not streaming else None
                )

        except Exception as e:
            self._add_check(
                "Dataset Access",
                False,
                ValidationLevel.ERROR,
                f"Cannot access '{dataset_name}': {e}",
                suggestion="Check dataset name and HF_TOKEN for private datasets"
            )

    def _validate_tokenizer(self):
        """Validate tokenizer."""
        self._log("Checking tokenizer...")

        dataset_config = self.config.get('dataset', {})
        processed_dir = Path(dataset_config.get('processed_dir', 'data/processed'))
        tokenizer_path = Path(
            self.config.get('tokenizer', {}).get('path', processed_dir / 'tokenizer.json')
        )

        if tokenizer_path.exists():
            try:
                from tokenizers import Tokenizer
                tokenizer = Tokenizer.from_file(str(tokenizer_path))
                vocab_size = tokenizer.get_vocab_size()

                self._add_check(
                    "Tokenizer",
                    True,
                    ValidationLevel.INFO,
                        f"Loaded from {tokenizer_path} (vocab: {vocab_size:,})"
                )

                # Verify vocab size matches config
                config_vocab = self.config.get('model', {}).get('vocab_size', 32000)
                if vocab_size != config_vocab:
                    self._add_check(
                        "Vocab Size Match",
                        False,
                        ValidationLevel.ERROR,
                        f"Tokenizer vocab ({vocab_size}) != config ({config_vocab})",
                        suggestion="Update config.yaml model.vocab_size or retrain tokenizer"
                    )
            except Exception as e:
                self._add_check(
                    "Tokenizer",
                    False,
                    ValidationLevel.ERROR,
                    f"Failed to load: {e}",
                    suggestion="Run tokenizer training first"
                )
        else:
            self._add_check(
                "Tokenizer",
                True,
                ValidationLevel.WARNING,
                f"Not found at {tokenizer_path} (will train new)",
            )

    def _validate_model_config(self):
        """Validate model configuration."""
        self._log("Checking model config...")

        model_config = self.config.get('model', {})

        # Architecture
        architecture = model_config.get('architecture', 'base')
        valid_archs = ['base', 'enhanced', 'professional']
        arch_valid = architecture in valid_archs

        self._add_check(
            "Model Architecture",
            arch_valid,
            ValidationLevel.ERROR if not arch_valid else ValidationLevel.INFO,
            architecture,
            suggestion=f"Must be one of: {valid_archs}" if not arch_valid else None
        )

        # Dimensions
        d_model = model_config.get('d_model', 768)
        n_heads = model_config.get('num_heads', model_config.get('n_heads', 12))

        if d_model % n_heads != 0:
            self._add_check(
                "Head Dimensions",
                False,
                ValidationLevel.ERROR,
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})",
                suggestion=f"Set d_model to {n_heads * (d_model // n_heads)}"
            )
        else:
            self._add_check(
                "Head Dimensions",
                True,
                ValidationLevel.INFO,
                f"d_model={d_model}, n_heads={n_heads}, head_dim={d_model // n_heads}"
            )

        # Training params
        training_config = self.config.get('training', {})
        batch_size = training_config.get('batch_size', 32)
        grad_accum = training_config.get('gradient_accumulation_steps', 1)
        effective_batch = batch_size * grad_accum

        self._add_check(
            "Effective Batch Size",
            True,
            ValidationLevel.INFO,
            f"{effective_batch} (batch={batch_size} x accum={grad_accum})"
        )

    def _validate_budget(self):
        """Validate budget constraints."""
        self._log("Checking budget...")

        if not self.budget_tracker:
            self._add_check(
                "Budget Tracker",
                True,
                ValidationLevel.WARNING,
                "Not configured (no spending limits)"
            )
            return

        training_config = self.config.get('training', {})
        max_steps = training_config.get('max_steps', 10000)
        gpu_type = self.config.get('hardware', {}).get(
            'modal_gpu',
            training_config.get('gpu_type', 'A10G')
        )

        estimate = self.budget_tracker.estimate_cost(
            max_steps=max_steps,
            gpu_type=gpu_type,
            model_params_millions=self.budget_tracker.estimate_model_params_millions_from_config(self.config),
        )

        self.report.estimated_cost = estimate.estimated_cost

        self._add_check(
            "Estimated Cost",
            True,
            ValidationLevel.INFO,
                f"${estimate.estimated_cost:.2f} for {max_steps:,} steps on {gpu_type}"
        )

        self._add_check(
            "Budget Available",
            estimate.within_safety_margin,
            ValidationLevel.ERROR if not estimate.within_safety_margin else ValidationLevel.INFO,
            f"${estimate.remaining_budget:.2f} remaining (90% safety: ${estimate.safety_margin_budget:.2f})",
            suggestion="Reduce max_steps or wait for budget reset" if not estimate.within_safety_margin else None
        )

        if estimate.budget_utilization_pct > 80:
            self._add_check(
                "Budget Warning",
                True,
                ValidationLevel.WARNING,
                f"Will use {estimate.budget_utilization_pct:.1f}% of monthly budget"
            )

    def _validate_checkpoint(self):
        """Validate checkpoint if resuming."""
        self._log("Checking checkpoints...")

        checkpoint_path = self.config.get('training', {}).get('resume_from')

        if not checkpoint_path:
            self._add_check(
                "Resume Checkpoint",
                True,
                ValidationLevel.INFO,
                "Starting fresh (no resume)"
            )
            return

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            self._add_check(
                "Resume Checkpoint",
                False,
                ValidationLevel.ERROR,
                f"Not found: {checkpoint_path}",
                suggestion="Remove resume_from or provide valid checkpoint"
            )
            return

        try:
            import torch
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Check checkpoint contents
            required_keys = ['model_state_dict']
            has_keys = all(k in checkpoint for k in required_keys)

            self._add_check(
                "Checkpoint Format",
                has_keys,
                ValidationLevel.ERROR if not has_keys else ValidationLevel.INFO,
                "Valid format" if has_keys else f"Missing keys: {required_keys}",
            )

            # Check config compatibility
            if 'config' in checkpoint:
                ckpt_config = checkpoint['config']
                ckpt_model_config = ckpt_config.get('model', ckpt_config)
                model_config = self.config.get('model', {})

                mismatches = []
                key_pairs = [
                    ('d_model', 'd_model'),
                    ('num_heads', 'num_heads'),
                    ('n_heads', 'num_heads'),
                    ('num_layers', 'num_layers'),
                    ('n_layers', 'num_layers'),
                    ('vocab_size', 'vocab_size'),
                    ('max_seq_length', 'max_seq_length'),
                    ('max_seq_len', 'max_seq_length'),
                ]
                for ckpt_key, model_key in key_pairs:
                    if ckpt_key in ckpt_model_config and model_key in model_config:
                        if ckpt_model_config[ckpt_key] != model_config[model_key]:
                            mismatches.append(
                                f"{model_key}: {ckpt_model_config[ckpt_key]} vs {model_config[model_key]}"
                            )

                if mismatches:
                    self._add_check(
                        "Config Compatibility",
                        False,
                        ValidationLevel.ERROR,
                        f"Mismatches: {', '.join(mismatches)}",
                        suggestion="Update config to match checkpoint or start fresh"
                    )
                else:
                    self._add_check(
                        "Config Compatibility",
                        True,
                        ValidationLevel.INFO,
                        "Checkpoint matches current config"
                    )

        except Exception as e:
            self._add_check(
                "Checkpoint Load",
                False,
                ValidationLevel.ERROR,
                f"Failed to load: {e}"
            )

    def _validate_output_paths(self):
        """Validate output directories."""
        self._log("Checking output paths...")

        output_dir = Path(
            self.config.get('output', {}).get(
                'dir',
                self.config.get('checkpoint', {}).get('save_dir', 'models')
            )
        )
        log_dir = Path(self.config.get('logging', {}).get('log_dir', 'logs'))

        for name, path in [("Model Output", output_dir), ("Logs", log_dir)]:
            try:
                path.mkdir(parents=True, exist_ok=True)
                self._add_check(name, True, ValidationLevel.INFO, str(path))
            except Exception as e:
                self._add_check(
                    name,
                    False,
                    ValidationLevel.ERROR,
                    f"Cannot create {path}: {e}"
                )


def validate_before_training(
    config_path: str,
    budget_tracker=None,
    verbose: bool = True
) -> Tuple[bool, ValidationReport]:
    """
    Convenience function to validate before training.

    Args:
        config_path: Path to config.yaml
        budget_tracker: Optional BudgetTracker instance
        verbose: Print validation results

    Returns:
        (can_proceed, report)
    """
    config = load_yaml_config(config_path)

    validator = PreTrainingValidator(
        config=config,
        budget_tracker=budget_tracker,
        verbose=verbose
    )

    report = validator.validate_all()
    return report.can_proceed, report


if __name__ == "__main__":
    # Test validation
    can_proceed, report = validate_before_training("configs/config.yaml")
    print(f"\nCan proceed: {can_proceed}")
    print(f"Estimated cost: ${report.estimated_cost:.2f}" if report.estimated_cost else "")
