"""
Data Preprocessing Pipeline with State Management

This module handles:
1. Dataset downloading and caching
2. Tokenization with BPE
3. Train/Val/Test splitting
4. Binary file serialization
5. State tracking for incremental processing
"""

import json
import os
import pickle
import hashlib
import threading
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

from datasets import load_dataset
from tokenizers import Tokenizer

from ..core.tokenization import TokenizerManager
from ..utils.config import load_env, get_hf_token
from ..utils.logging import log_step, log_info, log_ok, log_warn, log_fail

# Load environment variables
load_env()


@dataclass
class PreprocessingState:
    """
    Track preprocessing state for reproducibility and resumption.

    Hash Invariants:
    - dataset_hash: H(dataset_name, split_ratios, max_seq_length)
    - tokenizer_hash: H(vocab_size, training_texts_hash)
    """
    dataset_name: str
    dataset_hash: str
    split_ratios: Dict[str, float]
    max_seq_length: int
    vocab_size: int
    tokenizer_trained: bool
    tokenizer_hash: Optional[str]
    data_tokenized: bool
    train_samples: int
    val_samples: int
    test_samples: int
    completed: bool

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'PreprocessingState':
        return cls(**data)


class DataPreprocessor:
    """
    Production-grade data preprocessing pipeline.

    Pipeline Stages:
    1. Dataset Loading & Validation
    2. Tokenizer Training (if needed)
    3. Data Tokenization
    4. Train/Val/Test Splitting
    5. Binary Serialization

    State Management:
    - Tracks completion of each stage
    - Supports resumption from any stage
    - Validates data integrity with hashes
    """

    def __init__(
        self,
        dataset_name: str,
        cache_dir: str,
        processed_dir: str,
        split_ratios: Dict[str, float],
        max_seq_length: int,
        vocab_size: int,
        max_samples: Optional[int] = None,
        tokenizer_backend: str = "hf",
    ):
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir).resolve()
        self.processed_dir = Path(processed_dir).resolve()
        self.split_ratios = split_ratios
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.max_samples = max_samples  # limit for fast first run (None = full)
        self.tokenizer_backend = tokenizer_backend
        self.split_seed = 42
        self.split_strategy_version = 2

        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Paths
        self.state_path = self.processed_dir / "preprocessing_state.json"
        self.tokenizer_path = self.processed_dir / "tokenizer.json"
        self.train_path = self.processed_dir / "train.bin"
        self.val_path = self.processed_dir / "val.bin"
        self.test_path = self.processed_dir / "test.bin"
        self.meta_path = self.processed_dir / "meta.pkl"

        # State
        self.state = self._load_or_create_state()
        self.tokenizer: Optional[Tokenizer] = None

    def _compute_dataset_hash(self) -> str:
        """
        Compute deterministic hash of dataset configuration.

        H(dataset_name, split_ratios, max_seq_length, vocab_size)
        """
        config_str = (
            f"{self.dataset_name}_{self.split_ratios}_{self.max_seq_length}_"
            f"{self.vocab_size}_{self.max_samples}_{self.split_seed}_{self.split_strategy_version}"
        )
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _load_or_create_state(self) -> PreprocessingState:
        """Load existing state or create new one."""
        if self.state_path.exists():
            with open(self.state_path, 'r') as f:
                state_dict = json.load(f)
                state = PreprocessingState.from_dict(state_dict)

            # Validate hash
            current_hash = self._compute_dataset_hash()
            if state.dataset_hash != current_hash:
                log_warn("Configuration changed. Creating new preprocessing state.")
                return self._create_new_state()

            return state
        else:
            return self._create_new_state()

    def _create_new_state(self) -> PreprocessingState:
        """Create new preprocessing state."""
        return PreprocessingState(
            dataset_name=self.dataset_name,
            dataset_hash=self._compute_dataset_hash(),
            split_ratios=self.split_ratios,
            max_seq_length=self.max_seq_length,
            vocab_size=self.vocab_size,
            tokenizer_trained=False,
            tokenizer_hash=None,
            data_tokenized=False,
            train_samples=0,
            val_samples=0,
            test_samples=0,
            completed=False
        )

    def _save_state(self):
        """Persist state to disk."""
        with open(self.state_path, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def run(self, force_reprocess: bool = False) -> Tuple[str, str, str]:
        """
        Run complete preprocessing pipeline.

        Args:
            force_reprocess: Force reprocessing even if completed

        Returns:
            (train_path, val_path, test_path)
        """
        if not force_reprocess and self._can_reuse_artifacts():
            log_ok("Preprocessed data found. Using existing files.")
            log_info(f"  Train: {self.train_path}")
            log_info(f"  Val: {self.val_path}")
            log_info(f"  Test: {self.test_path}")
            log_info(f"  Tokenizer: {self.tokenizer_path}")
            self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
            return str(self.train_path), str(self.val_path), str(self.test_path)

        log_step("DATA PREPROCESSING")
        log_info(f"Cache dir: {self.cache_dir}")
        log_info(f"Processed dir: {self.processed_dir}")

        # Stage 1: Download dataset
        log_step("[1/4] Download dataset")
        texts = self._load_dataset()

        # Stage 2: Train tokenizer
        log_step("[2/4] Training tokenizer")
        self.tokenizer = self._train_or_load_tokenizer(texts)

        # Stage 3: Tokenize data
        log_step("[3/4] Tokenizing data")
        train_ids, val_ids, test_ids = self._tokenize_and_split(texts)

        # Stage 4: Save to binary files
        log_step("[4/4] Saving binary files")
        self._save_binary_files(train_ids, val_ids, test_ids)

        self.state.completed = True
        self._save_state()
        log_ok("PREPROCESSING COMPLETED")
        self._print_statistics()

        return str(self.train_path), str(self.val_path), str(self.test_path)

    def _can_reuse_artifacts(self) -> bool:
        """Reuse artifacts only when state and metadata match the active config."""
        files_exist = all([
            self.state_path.exists(),
            self.train_path.exists(),
            self.val_path.exists(),
            self.test_path.exists(),
            self.tokenizer_path.exists(),
            self.meta_path.exists(),
        ])
        if not files_exist:
            return False

        if not (self.state.completed and self.state.tokenizer_trained and self.state.data_tokenized):
            return False

        try:
            with open(self.meta_path, 'rb') as f:
                meta = pickle.load(f)

            tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
            tokenizer_vocab_size = tokenizer.get_vocab_size()
        except Exception as exc:
            log_warn(f"Cannot validate existing preprocessing artifacts: {exc}")
            return False

        expected_hash = self._compute_dataset_hash()
        if self.state.dataset_hash != expected_hash:
            return False

        if meta.get('max_seq_length') != self.max_seq_length:
            return False

        if meta.get('vocab_size') != tokenizer_vocab_size:
            return False

        if tokenizer_vocab_size < self.vocab_size:
            return False

        return True

    def _load_dataset(self) -> List[str]:
        """Load dataset - use cache if available, otherwise download."""
        hf_token = get_hf_token()
        max_samples = self.max_samples or 500000  # Default limit to avoid memory issues

        # First try to load from cache (fast, no network needed)
        log_info(f"Loading dataset: {self.dataset_name}")
        log_info(f"Cache dir: {self.cache_dir}")

        try:
            # Try loading from cache first (no download)
            split = f"train[:{max_samples}]" if max_samples else "train"

            dataset = load_dataset(
                self.dataset_name,
                split=split,
                cache_dir=str(self.cache_dir),
                token=hf_token if hf_token else None,
            )

            n = len(dataset)
            log_ok(f"Loaded from cache: {n:,} samples")

            # Detect text column
            text_col = None
            first = dataset[0] if n else {}
            for key in ("text", "content", "story", "sentence"):
                if key in first:
                    text_col = key
                    break
            if text_col is None:
                text_col = list(dataset.features.keys())[0] if dataset.features else "text"
                log_warn(f"No 'text' column; using '{text_col}'")

            log_info(f"Using column: '{text_col}'")
            log_info(f"Extracting text from {n:,} samples...")

            # Extract texts (fast - data already loaded)
            texts: List[str] = []
            for item in tqdm(dataset, desc="Extracting", unit=" samples"):
                if text_col in item:
                    raw = item[text_col]
                    if isinstance(raw, str) and raw.strip():
                        texts.append(raw)

            log_ok(f"Extracted {len(texts):,} text samples")

        except Exception as e:
            log_warn(f"Cache load failed: {e}")
            log_info("Trying streaming mode...")

            try:
                # Fallback to streaming
                dataset = load_dataset(
                    self.dataset_name,
                    split="train",
                    streaming=True,
                    token=hf_token if hf_token else None,
                )

                first_sample = next(iter(dataset))
                text_col = None
                for key in ("text", "content", "story", "sentence"):
                    if key in first_sample:
                        text_col = key
                        break
                if text_col is None:
                    text_col = list(first_sample.keys())[0]

                log_info(f"Streaming samples (max {max_samples:,})...")

                texts: List[str] = []
                for i, item in enumerate(tqdm(dataset, desc="Loading", unit=" samples", total=max_samples)):
                    if i >= max_samples:
                        break
                    if text_col in item:
                        raw = item[text_col]
                        if isinstance(raw, str) and raw.strip():
                            texts.append(raw)

                log_ok(f"Loaded {len(texts):,} samples via streaming")

            except Exception as e2:
                log_fail(f"Failed to load dataset: {e2}")
                raise

        if not texts:
            log_fail("Extracted 0 text samples. Check dataset format.")
            raise ValueError("No text samples extracted from dataset")

        log_ok(f"Total samples: {len(texts):,}")
        return texts

    def _log_cache_contents(self) -> None:
        """Log what was written to cache."""
        try:
            if not self.cache_dir.exists():
                return
            total_size = 0
            files: List[str] = []
            for p in self.cache_dir.rglob("*"):
                if p.is_file():
                    total_size += p.stat().st_size
                    files.append(str(p.relative_to(self.cache_dir)))
            if files:
                log_info(f"Cache: {len(files)} files, {total_size / (1024*1024):.2f} MB")
        except Exception:
            pass

    def _train_or_load_tokenizer(self, texts: List[str]):
        """Train new tokenizer or load existing one using fast HF tokenizers."""
        if self.tokenizer_path.exists() and self.state.tokenizer_trained:
            log_info("Loading existing tokenizer")
            self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
            log_ok(f"Loaded tokenizer (vocab_size={self.tokenizer.get_vocab_size()})")
            return self.tokenizer

        train_texts = texts[:min(100000, len(texts))]
        log_info(
            f"Training BPE tokenizer (vocab_size={self.vocab_size}) with backend={self.tokenizer_backend}"
        )

        manager = TokenizerManager(vocabulary_size=self.vocab_size)
        manager.train(train_texts, output_path=str(self.processed_dir), tokenizer_type="bpe")
        self.tokenizer = manager.tokenizer
        self.state.tokenizer_trained = True
        self.state.tokenizer_hash = self._compute_dataset_hash()
        self._save_state()
        log_ok(f"Trained and saved tokenizer (vocab_size={self.tokenizer.get_vocab_size()})")
        return self.tokenizer

    def _tokenize_and_split(
        self,
        texts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize all texts using fast parallel batch encoding."""
        log_info(f"Tokenizing {len(texts):,} samples (parallel batch encoding)")

        # Enable padding and truncation
        self.tokenizer.enable_padding(
            pad_id=self.tokenizer.token_to_id("<PAD>"),
            pad_token="<PAD>",
            length=self.max_seq_length
        )
        self.tokenizer.enable_truncation(max_length=self.max_seq_length)

        # Batch encode - uses Rust parallel backend (10-100x faster)
        batch_size = 10000  # Process in chunks to show progress
        all_ids = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing batches"):
            batch = texts[i:i + batch_size]
            encodings = self.tokenizer.encode_batch(batch, add_special_tokens=True)
            batch_ids = [e.ids for e in encodings]
            all_ids.extend(batch_ids)

        log_ok(f"Tokenized {len(all_ids):,} samples")

        # Convert to numpy array
        all_token_ids = np.array(all_ids, dtype=np.int32)

        # Shuffle deterministically before splitting so validation is representative
        # and preprocessing remains reproducible across runs.
        rng = np.random.default_rng(self.split_seed)
        permutation = rng.permutation(len(all_token_ids))
        all_token_ids = all_token_ids[permutation]

        # Calculate split indices
        n = len(all_token_ids)
        train_idx = int(n * self.split_ratios['train'])
        val_idx = train_idx + int(n * self.split_ratios['validation'])

        # Split data
        train_ids = all_token_ids[:train_idx]
        val_ids = all_token_ids[train_idx:val_idx]
        test_ids = all_token_ids[val_idx:]

        # Update state
        self.state.train_samples = len(train_ids)
        self.state.val_samples = len(val_ids)
        self.state.test_samples = len(test_ids)
        self.state.data_tokenized = True
        self._save_state()

        log_ok(f"Split: Train={len(train_ids):,}, Val={len(val_ids):,}, Test={len(test_ids):,}")

        return train_ids, val_ids, test_ids

    def _save_binary_files(
        self,
        train_ids: np.ndarray,
        val_ids: np.ndarray,
        test_ids: np.ndarray
    ):
        """Save tokenized data to efficient binary format."""
        log_info("Saving train.bin")
        train_ids.tofile(str(self.train_path))

        log_info("Saving val.bin")
        val_ids.tofile(str(self.val_path))

        log_info("Saving test.bin")
        test_ids.tofile(str(self.test_path))

        # Save metadata
        meta = {
            'vocab_size': self.tokenizer.get_vocab_size(),
            'max_seq_length': self.max_seq_length,
            'train_samples': len(train_ids),
            'val_samples': len(val_ids),
            'test_samples': len(test_ids),
            'pad_token_id': self.tokenizer.token_to_id('<PAD>'),
            'bos_token_id': self.tokenizer.token_to_id('<BOS>'),
            'eos_token_id': self.tokenizer.token_to_id('<EOS>'),
            'unk_token_id': self.tokenizer.token_to_id('<UNK>'),
        }

        with open(self.meta_path, 'wb') as f:
            pickle.dump(meta, f)

        log_ok("Saved binary files")

    def _print_statistics(self):
        """Print preprocessing statistics."""
        train_size = self.train_path.stat().st_size / (1024 ** 2)
        val_size = self.val_path.stat().st_size / (1024 ** 2)
        test_size = self.test_path.stat().st_size / (1024 ** 2)

        log_info(f"Train: {self.state.train_samples:,} samples ({train_size:.2f} MB)")
        log_info(f"Val: {self.state.val_samples:,} samples ({val_size:.2f} MB)")
        log_info(f"Test: {self.state.test_samples:,} samples ({test_size:.2f} MB)")

    def clean(self, keep_tokenizer: bool = True):
        """
        Clean preprocessed files.

        Args:
            keep_tokenizer: Whether to keep trained tokenizer
        """
        log_step("Cleaning preprocessed files")

        files_to_remove = [
            self.train_path,
            self.val_path,
            self.test_path,
            self.meta_path,
            self.state_path
        ]

        if not keep_tokenizer:
            files_to_remove.append(self.tokenizer_path)

        for path in files_to_remove:
            if path.exists():
                path.unlink()
                log_ok(f"Removed {path.name}")
        log_ok("Cleanup completed")
