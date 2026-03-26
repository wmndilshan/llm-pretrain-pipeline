"""
Efficient Dataset Loader for Binary Token Files

Implements memory-mapped file reading for large-scale datasets.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Dict
import pickle


class TokenDataset(Dataset):
    """
    Memory-mapped dataset for efficient loading of tokenized data.

    Memory Efficiency:
    - Uses np.memmap for lazy loading
    - Only loads requested samples into RAM
    - Supports datasets larger than RAM

    Time Complexity:
    - __getitem__: O(seq_len) - constant for fixed sequence length
    - Memory: O(1) per access (lazy loading)
    """

    def __init__(
        self,
        data_path: str,
        max_seq_length: int,
        pad_token_id: int = 256
    ):
        self.data_path = Path(data_path)
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Memory-map the file
        self.data = np.memmap(
            str(self.data_path),
            dtype=np.int32,
            mode='r'
        )

        # Calculate number of samples
        self.num_samples = len(self.data) // max_seq_length

        # Reshape for easier indexing
        self.data = self.data.reshape(self.num_samples, max_seq_length)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            {
                'input_ids': [seq_len],
                'labels': [seq_len],
                'attention_mask': [seq_len]
            }
        """
        # Get sequence (PyTorch expects long for indices/embeddings)
        sequence = self.data[idx].astype(np.int64)
        input_ids = torch.from_numpy(sequence.copy()).long()
        labels = torch.from_numpy(sequence.copy()).long()

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.pad_token_id).long()

        # Set padding positions in labels to -1 (ignored in loss)
        labels[attention_mask == 0] = -1

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


def create_dataloader(
    data_path: str,
    meta_path: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for the dataset.

    Args:
        data_path: Path to binary data file
        meta_path: Path to metadata pickle
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    # Load metadata
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    # Create dataset
    dataset = TokenDataset(
        data_path=data_path,
        max_seq_length=meta['max_seq_length'],
        pad_token_id=meta['pad_token_id']
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    return dataloader


class InfiniteDataLoader:
    """
    Wrapper for infinite iteration over dataloader.

    Useful for training with step-based rather than epoch-based loops.
    """

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            # Reset iterator
            self.iterator = iter(self.dataloader)
            return next(self.iterator)
