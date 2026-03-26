"""
Pre-Training Validation System

Tests the entire pipeline with minimal resources before Modal GPU training:
1. Data loading and preprocessing
2. Model architecture initialization
3. Forward/backward pass
4. Optimizer step
5. Checkpoint saving/loading
6. Generation capability

Runs on CPU with tiny dataset - completes in ~2 minutes.
Prevents wasting Modal credits on broken configurations.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import sys
from typing import Dict, Any, Tuple
import traceback
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_yaml_config
from src.utils.logging import log_fail, log_info, log_ok, log_step, log_warn


class ValidationReport:
    """Structured validation report."""
    
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
        self.warnings = []
        self.start_time = datetime.now()
        self.end_time = None
    
    def add_test(self, name: str, passed: bool, message: str = "", error: str = ""):
        """Add test result."""
        self.tests.append({
            'name': name,
            'passed': passed,
            'message': message,
            'error': error
        })
        
        if passed:
            self.passed += 1
            log_ok(name)
            if message:
                log_info(message)
        else:
            self.failed += 1
            log_fail(name)
            if message:
                log_info(message)
            if error:
                log_fail(f"Error: {error}")
    
    def add_warning(self, message: str):
        """Add warning."""
        self.warnings.append(message)
        log_warn(message)
    
    def finalize(self):
        """Finalize report."""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        log_step("VALIDATION REPORT")
        log_info(
            f"Total: {self.passed + self.failed}  Passed: {self.passed}  "
            f"Failed: {self.failed}  Warnings: {len(self.warnings)}"
        )
        log_info(f"Duration: {duration:.1f}s")
        
        return self.failed == 0
    
    def save(self, path: str):
        """Save report to file."""
        report = {
            'timestamp': self.start_time.isoformat(),
            'duration': (self.end_time - self.start_time).total_seconds() if self.end_time else 0,
            'tests': self.tests,
            'warnings': self.warnings,
            'passed': self.passed,
            'failed': self.failed
        }
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)


class PreTrainingValidator:
    """
    Validates entire training pipeline before Modal GPU training.
    
    Tests with minimal resources:
    - 100 samples
    - 10 steps
    - CPU only
    - Completes in ~2 minutes
    """
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.report = ValidationReport()
        self.validation_dir = Path("validation_results")
        self.validation_dir.mkdir(exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration."""
        return load_yaml_config(self.config_path)

    def _build_model(self):
        """Build the configured model through the canonical registry."""
        from src.core.models import ModelConfig, get_model

        model_cfg = self.config['model']
        enhanced_cfg = model_cfg.get('enhanced', {})
        runtime_config = ModelConfig(
            architecture=model_cfg.get('architecture', 'base'),
            model_name=model_cfg.get('model_name', ''),
            parameter_count=model_cfg.get('parameter_count', 0),
            architecture_family=model_cfg.get('architecture_family', 'decoder-only-transformer'),
            vocab_size=model_cfg['vocab_size'],
            d_model=model_cfg['d_model'],
            num_heads=model_cfg['num_heads'],
            num_layers=model_cfg['num_layers'],
            d_ff=model_cfg['d_ff'],
            max_seq_length=model_cfg['max_seq_length'],
            dropout=model_cfg.get('dropout', 0.1),
            use_rotary_embeddings=model_cfg.get(
                'use_rotary_embeddings',
                enhanced_cfg.get('use_rotary_embeddings', True),
            ),
            use_flash_attention=False,
            use_grouped_query_attention=model_cfg.get(
                'use_grouped_query_attention',
                enhanced_cfg.get('use_grouped_query_attention', True),
            ),
            gqa_num_kv_heads=model_cfg.get(
                'gqa_num_kv_heads',
                enhanced_cfg.get('gqa_num_kv_heads', 2),
            ),
            use_rms_norm=model_cfg.get(
                'use_rms_norm',
                enhanced_cfg.get('use_rms_norm', True),
            ),
            use_swiglu=model_cfg.get(
                'use_swiglu',
                enhanced_cfg.get('use_swiglu', True),
            ),
            gradient_checkpointing=model_cfg.get(
                'gradient_checkpointing',
                enhanced_cfg.get('gradient_checkpointing', False),
            ),
            compile_model=False,
            version_models=model_cfg.get('version_models', True),
        )
        return get_model(runtime_config.architecture, runtime_config)
    
    def run_all_tests(self) -> bool:
        """
        Run all validation tests.
        
        Returns:
            True if all tests passed
        """
        log_step("PRE-TRAINING VALIDATION")
        log_info(f"Config: {self.config_path}")
        log_info("This will test the entire pipeline with minimal resources")
        
        # Test 1: Configuration validation
        self.test_config_validation()
        
        # Test 2: Data pipeline
        self.test_data_pipeline()
        
        # Test 3: Model initialization
        self.test_model_initialization()
        
        # Test 4: Forward pass
        self.test_forward_pass()
        
        # Test 5: Backward pass
        self.test_backward_pass()
        
        # Test 6: Optimizer step
        self.test_optimizer_step()
        
        # Test 7: Training loop
        self.test_training_loop()
        
        # Test 8: Checkpoint save/load
        self.test_checkpoint_operations()
        
        # Test 9: Generation
        self.test_generation()
        
        # Test 10: Memory usage
        self.test_memory_usage()
        
        # Finalize
        success = self.report.finalize()
        
        # Save report
        report_path = self.validation_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.report.save(report_path)
        log_info(f"Report saved: {report_path}")
        
        return success
    
    def test_config_validation(self):
        """Test configuration is valid."""
        try:
            # Check required keys
            required = ['model', 'training', 'hardware']
            missing = [k for k in required if k not in self.config]
            
            if missing:
                self.report.add_test(
                    "Configuration Validation",
                    False,
                    f"Missing keys: {missing}"
                )
                return
            
            # Check model config
            model_required = ['vocab_size', 'd_model', 'num_heads', 'num_layers']
            model_missing = [k for k in model_required if k not in self.config['model']]
            
            if model_missing:
                self.report.add_test(
                    "Configuration Validation",
                    False,
                    f"Missing model keys: {model_missing}"
                )
                return
            
            # Validate dimensions
            d_model = self.config['model']['d_model']
            num_heads = self.config['model']['num_heads']
            
            if d_model % num_heads != 0:
                self.report.add_test(
                    "Configuration Validation",
                    False,
                    f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
                )
                return
            self.report.add_test(
                "Configuration Validation",
                True,
                f"Config valid: {d_model}d x {self.config['model']['num_layers']}L"
            )
            return
            
            self.report.add_test(
                "Configuration Validation",
                True,
                f"Config valid: {d_model}d x {self.config['model']['num_layers']}L"
            )
            
        except Exception as e:
            self.report.add_test(
                "Configuration Validation",
                False,
                error=str(e)
            )
    
    def test_data_pipeline(self):
        """Test data loading and preprocessing."""
        try:
            # Create tiny synthetic dataset
            vocab_size = self.config['model']['vocab_size']
            seq_len = 64  # Short for testing
            num_samples = 100
            
            # Generate random data
            data = np.random.randint(0, vocab_size, size=(num_samples, seq_len), dtype=np.int32)
            
            # Save to temporary file
            test_data_path = self.validation_dir / "test_data.bin"
            data.tofile(test_data_path)
            
            # Try to load
            loaded = np.memmap(test_data_path, dtype=np.int32, mode='r')
            loaded = loaded.reshape(-1, seq_len)
            
            if loaded.shape == data.shape:
                self.report.add_test(
                    "Data Pipeline",
                    True,
                    f"Successfully created and loaded {num_samples} samples"
                )
                
                # Store for later tests (writable + long for PyTorch indices/loss)
                self.test_data = torch.from_numpy(loaded.copy()).long()
            else:
                self.report.add_test(
                    "Data Pipeline",
                    False,
                    f"Shape mismatch: {loaded.shape} vs {data.shape}"
                )
                
        except Exception as e:
            self.report.add_test(
                "Data Pipeline",
                False,
                error=str(e)
            )
            self.test_data = None
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        try:
            self.model = self._build_model()
            
            params = self.model.count_parameters()
            
            self.report.add_test(
                "Model Initialization",
                True,
                f"Model created: {params:,} parameters"
            )
            
            # Check for NaN/Inf in initial weights
            has_nan = False
            has_inf = False
            
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any():
                    has_nan = True
                if torch.isinf(param).any():
                    has_inf = True
            
            if has_nan or has_inf:
                self.report.add_warning(
                    f"Initial weights contain NaN={has_nan}, Inf={has_inf}"
                )
            
        except Exception as e:
            self.report.add_test(
                "Model Initialization",
                False,
                error=str(e)
            )
            self.model = None
    
    def test_forward_pass(self):
        """Test forward pass."""
        if self.model is None or self.test_data is None:
            self.report.add_test(
                "Forward Pass",
                False,
                "Skipped: Previous tests failed"
            )
            return
        
        try:
            self.model.eval()
            
            # Take small batch
            batch = self.test_data[:4]  # 4 samples
            
            with torch.no_grad():
                if hasattr(self.model, 'forward'):
                    logits, loss, _ = self.model(batch)
                else:
                    logits = self.model(batch)
                    loss = None
            
            # Check output shape
            expected_shape = (4, batch.shape[1], self.config['model']['vocab_size'])
            
            if logits.shape == expected_shape:
                # Check for NaN/Inf
                if torch.isnan(logits).any():
                    self.report.add_test(
                        "Forward Pass",
                        False,
                        "Output contains NaN"
                    )
                elif torch.isinf(logits).any():
                    self.report.add_test(
                        "Forward Pass",
                        False,
                        "Output contains Inf"
                    )
                else:
                    self.report.add_test(
                        "Forward Pass",
                        True,
                        f"Output shape: {logits.shape}"
                    )
            else:
                self.report.add_test(
                    "Forward Pass",
                    False,
                    f"Shape mismatch: {logits.shape} vs {expected_shape}"
                )
                
        except Exception as e:
            self.report.add_test(
                "Forward Pass",
                False,
                error=str(e)
            )
    
    def test_backward_pass(self):
        """Test backward pass."""
        if self.model is None or self.test_data is None:
            self.report.add_test(
                "Backward Pass",
                False,
                "Skipped: Previous tests failed"
            )
            return
        
        try:
            self.model.train()
            
            batch = self.test_data[:4]
            targets = self.test_data[:4]
            
            # Forward
            if hasattr(self.model, 'forward') and len(self.model.forward.__code__.co_varnames) > 2:
                logits, loss, _ = self.model(batch, targets)
            else:
                logits = self.model(batch)
                loss = nn.CrossEntropyLoss()(
                    logits.view(-1, self.config['model']['vocab_size']),
                    targets.view(-1)
                )
            
            # Backward
            loss.backward()
            
            # Check gradients
            has_grad = False
            has_nan = False
            has_inf = False
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    has_grad = True
                    if torch.isnan(param.grad).any():
                        has_nan = True
                    if torch.isinf(param.grad).any():
                        has_inf = True
            
            if not has_grad:
                self.report.add_test(
                    "Backward Pass",
                    False,
                    "No gradients computed"
                )
            elif has_nan:
                self.report.add_test(
                    "Backward Pass",
                    False,
                    "Gradients contain NaN"
                )
            elif has_inf:
                self.report.add_test(
                    "Backward Pass",
                    False,
                    "Gradients contain Inf"
                )
            else:
                self.report.add_test(
                    "Backward Pass",
                    True,
                    f"Loss: {loss.item():.4f}"
                )
            
            # Clear gradients
            self.model.zero_grad()
            
        except Exception as e:
            self.report.add_test(
                "Backward Pass",
                False,
                error=str(e)
            )
    
    def test_optimizer_step(self):
        """Test optimizer can update weights."""
        if self.model is None:
            self.report.add_test(
                "Optimizer Step",
                False,
                "Skipped: Previous tests failed"
            )
            return
        
        try:
            # Create optimizer
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
            
            # Save initial weights
            initial_weights = {}
            for name, param in self.model.named_parameters():
                initial_weights[name] = param.data.clone()
            
            # Training step
            batch = self.test_data[:4]
            targets = self.test_data[:4]
            
            optimizer.zero_grad()
            
            if hasattr(self.model, 'forward') and len(self.model.forward.__code__.co_varnames) > 2:
                logits, loss, _ = self.model(batch, targets)
            else:
                logits = self.model(batch)
                loss = nn.CrossEntropyLoss()(
                    logits.view(-1, self.config['model']['vocab_size']),
                    targets.view(-1)
                )
            
            loss.backward()
            optimizer.step()
            
            # Check weights changed
            weights_changed = False
            for name, param in self.model.named_parameters():
                if not torch.allclose(param.data, initial_weights[name], atol=1e-6):
                    weights_changed = True
                    break
            
            if weights_changed:
                self.report.add_test(
                    "Optimizer Step",
                    True,
                    "Weights updated successfully"
                )
            else:
                self.report.add_test(
                    "Optimizer Step",
                    False,
                    "Weights did not change"
                )
                
        except Exception as e:
            self.report.add_test(
                "Optimizer Step",
                False,
                error=str(e)
            )
    
    def test_training_loop(self):
        """Test a few training steps."""
        if self.model is None or self.test_data is None:
            self.report.add_test(
                "Training Loop",
                False,
                "Skipped: Previous tests failed"
            )
            return
        
        try:
            self.model.train()
            
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate']
            )
            
            losses = []
            num_steps = 5
            
            for step in range(num_steps):
                batch = self.test_data[:8]
                targets = self.test_data[:8]
                
                optimizer.zero_grad()
                
                if hasattr(self.model, 'forward') and len(self.model.forward.__code__.co_varnames) > 2:
                    logits, loss, _ = self.model(batch, targets)
                else:
                    logits = self.model(batch)
                    loss = nn.CrossEntropyLoss()(
                        logits.view(-1, self.config['model']['vocab_size']),
                        targets.view(-1)
                    )
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
                
                optimizer.step()
                
                losses.append(loss.item())
            
            # Check loss is reasonable
            avg_loss = sum(losses) / len(losses)
            
            if any(np.isnan(l) for l in losses):
                self.report.add_test(
                    "Training Loop",
                    False,
                    "Loss became NaN"
                )
            elif any(np.isinf(l) for l in losses):
                self.report.add_test(
                    "Training Loop",
                    False,
                    "Loss became Inf"
                )
            elif avg_loss > 100:
                self.report.add_warning(
                    f"High loss: {avg_loss:.2f} (might be normal for random init)"
                )
                self.report.add_test(
                    "Training Loop",
                    True,
                    f"Completed {num_steps} steps, avg loss: {avg_loss:.2f}"
                )
            else:
                self.report.add_test(
                    "Training Loop",
                    True,
                    f"Completed {num_steps} steps, avg loss: {avg_loss:.2f}"
                )
                
        except Exception as e:
            self.report.add_test(
                "Training Loop",
                False,
                error=str(e)
            )
    
    def test_checkpoint_operations(self):
        """Test checkpoint save and load."""
        if self.model is None:
            self.report.add_test(
                "Checkpoint Operations",
                False,
                "Skipped: Previous tests failed"
            )
            return
        
        try:
            # Save checkpoint
            checkpoint_path = self.validation_dir / "test_checkpoint.pt"
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'step': 10
            }
            
            torch.save(checkpoint, checkpoint_path)
            
            # Load checkpoint
            loaded = torch.load(checkpoint_path, map_location='cpu')
            
            # Create new model and load
            new_model = self._build_model()
            new_model.load_state_dict(loaded['model_state_dict'])
            
            # Verify weights match
            match = True
            for (n1, p1), (n2, p2) in zip(self.model.named_parameters(), new_model.named_parameters()):
                if not torch.allclose(p1, p2):
                    match = False
                    break
            
            if match:
                self.report.add_test(
                    "Checkpoint Operations",
                    True,
                    f"Saved and loaded checkpoint ({checkpoint_path.stat().st_size / 1024:.1f} KB)"
                )
            else:
                self.report.add_test(
                    "Checkpoint Operations",
                    False,
                    "Loaded weights don't match saved weights"
                )
                
        except Exception as e:
            self.report.add_test(
                "Checkpoint Operations",
                False,
                error=str(e)
            )
    
    def test_generation(self):
        """Test model can generate text."""
        if self.model is None:
            self.report.add_test(
                "Generation",
                False,
                "Skipped: Previous tests failed"
            )
            return
        
        try:
            self.model.eval()
            
            # Start with single token
            start_token = torch.randint(0, self.config['model']['vocab_size'], (1, 1))
            
            with torch.no_grad():
                if hasattr(self.model, 'generate'):
                    # Use built-in generate
                    generated = self.model.generate(
                        start_token,
                        max_new_tokens=10,
                        temperature=1.0
                    )
                else:
                    # Manual generation
                    generated = start_token
                    for _ in range(10):
                        if hasattr(self.model, 'forward'):
                            logits, _, _ = self.model(generated)
                        else:
                            logits = self.model(generated)
                        
                        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                        generated = torch.cat([generated, next_token], dim=1)
            
            if generated.shape[1] > start_token.shape[1]:
                self.report.add_test(
                    "Generation",
                    True,
                    f"Generated {generated.shape[1] - start_token.shape[1]} tokens"
                )
            else:
                self.report.add_test(
                    "Generation",
                    False,
                    "No tokens generated"
                )
                
        except Exception as e:
            self.report.add_test(
                "Generation",
                False,
                error=str(e)
            )
    
    def test_memory_usage(self):
        """Test memory usage is reasonable."""
        try:
            if self.model is None:
                self.report.add_test(
                    "Memory Usage",
                    False,
                    "Skipped: Model not initialized"
                )
                return
            
            # Calculate model size
            param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
            total_mb = (param_size + buffer_size) / (1024**2)
            
            # Estimate GPU memory needed
            estimated_gpu_mb = total_mb * 4  # Model + gradients + optimizer + activations
            
            self.report.add_test(
                "Memory Usage",
                True,
                f"Model: {total_mb:.1f}MB, Est. GPU: {estimated_gpu_mb:.1f}MB"
            )
            
            # Warnings
            if estimated_gpu_mb > 40000:  # 40GB
                self.report.add_warning(
                    f"Model may not fit on A100 40GB: {estimated_gpu_mb:.0f}MB estimated"
                )
            
        except Exception as e:
            self.report.add_test(
                "Memory Usage",
                False,
                error=str(e)
            )


def validate_before_training(config_path: str) -> bool:
    """
    Run pre-training validation.
    
    Args:
        config_path: Path to config file
        
    Returns:
        True if all tests passed
    """
    validator = PreTrainingValidator(config_path)
    return validator.run_all_tests()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate pipeline before training")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    success = validate_before_training(args.config)
    
    if success:
        log_ok("All tests passed. Safe to train on Modal GPU.")
        sys.exit(0)
    else:
        log_fail("Some tests failed. Fix issues before training.")
        sys.exit(1)
