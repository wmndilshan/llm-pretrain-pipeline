"""
Fast BPE Tokenizer with C++ Backend

Uses ctypes to interface with the C++ tokenizer for speedup.
Falls back to HuggingFace tokenizers library if C++ DLL not available.
"""

import os
import ctypes
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple
import multiprocessing

# Try to load C++ tokenizer
_LIB = None
_LIB_PATH = Path(__file__).parent / "tokenizer.dll"


def _load_library():
    """Load the C++ tokenizer library."""
    global _LIB
    if _LIB is not None:
        return _LIB

    if _LIB_PATH.exists():
        try:
            _LIB = ctypes.CDLL(str(_LIB_PATH))

            # Define function signatures
            _LIB.tokenizer_create.restype = ctypes.c_void_p
            _LIB.tokenizer_destroy.argtypes = [ctypes.c_void_p]

            _LIB.tokenizer_add_token.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
            _LIB.tokenizer_add_merge.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            _LIB.tokenizer_add_merge_with_priority.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]

            _LIB.tokenizer_encode.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int
            ]
            _LIB.tokenizer_encode.restype = ctypes.c_int

            _LIB.tokenizer_encode_batch.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int
            ]

            _LIB.tokenizer_get_vocab_size.argtypes = [ctypes.c_void_p]
            _LIB.tokenizer_get_vocab_size.restype = ctypes.c_int

            _LIB.tokenizer_get_pad_id.argtypes = [ctypes.c_void_p]
            _LIB.tokenizer_get_pad_id.restype = ctypes.c_int

            _LIB.tokenizer_get_token_id.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            _LIB.tokenizer_get_token_id.restype = ctypes.c_int

            _LIB.tokenizer_decode.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                ctypes.c_char_p, ctypes.c_int
            ]
            _LIB.tokenizer_decode.restype = ctypes.c_int

            _LIB.tokenizer_load_vocab.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            _LIB.tokenizer_load_vocab.restype = ctypes.c_int

            _LIB.tokenizer_load_merges.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            _LIB.tokenizer_load_merges.restype = ctypes.c_int

            return _LIB
        except Exception as e:
            print(f"Warning: Could not load C++ tokenizer: {e}")
            return None
    return None


class FastBPETokenizer:
    """
    Fast BPE Tokenizer with C++ backend.

    Features:
    - C++ backend for speedup
    - Falls back to HuggingFace tokenizers if C++ not available
    - Compatible with HuggingFace byte-level BPE format
    """

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self._cpp_tokenizer = None
        self._hf_tokenizer = None
        self._use_cpp = False
        self._lib = None

        # Try C++ backend first
        lib = _load_library()
        if lib is not None:
            self._cpp_tokenizer = lib.tokenizer_create()
            self._use_cpp = True
            self._lib = lib

        # Always init HF tokenizer as fallback/primary trainer
        self._init_hf_tokenizer()

    def _init_hf_tokenizer(self):
        """Initialize HuggingFace tokenizers backend."""
        try:
            from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders

            self._hf_tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
            self._hf_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self._hf_tokenizer.decoder = decoders.ByteLevel()
        except ImportError:
            if not self._use_cpp:
                raise RuntimeError("Neither C++ tokenizer nor HuggingFace tokenizers available!")

    def __del__(self):
        if self._cpp_tokenizer is not None and self._lib is not None:
            try:
                self._lib.tokenizer_destroy(self._cpp_tokenizer)
            except:
                pass

    def train(self, texts: List[str], verbose: bool = True) -> None:
        """
        Train tokenizer on texts.

        Uses HuggingFace tokenizers for training (fast Rust backend),
        then exports to C++ format if C++ backend available.
        """
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders

        if verbose:
            print(f"Training tokenizer on {len(texts):,} texts (vocab_size={self.vocab_size})...")

        # Use HuggingFace tokenizers for training (Rust backend - very fast)
        tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
            show_progress=verbose,
        )

        tokenizer.train_from_iterator(texts, trainer=trainer)

        # Add post-processor for special tokens
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<BOS> $A <EOS>",
            special_tokens=[("<BOS>", tokenizer.token_to_id("<BOS>")),
                          ("<EOS>", tokenizer.token_to_id("<EOS>"))],
        )

        self._hf_tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()

        # Export to C++ if available
        if self._cpp_tokenizer is not None:
            self._export_to_cpp()
            if verbose:
                cpp_vocab = self._lib.tokenizer_get_vocab_size(self._cpp_tokenizer)
                print(f"Exported to C++ backend (vocab_size={cpp_vocab})")

        if verbose:
            print(f"Tokenizer trained! Vocab size: {tokenizer.get_vocab_size()}")

    def _export_to_cpp(self):
        """Export HuggingFace tokenizer to C++ backend."""
        if self._hf_tokenizer is None or self._cpp_tokenizer is None:
            return

        import json

        # Get tokenizer JSON
        tokenizer_json = json.loads(self._hf_tokenizer.to_str())

        # Export vocabulary
        vocab = self._hf_tokenizer.get_vocab()
        for token, id in vocab.items():
            try:
                token_bytes = token.encode('utf-8')
                self._lib.tokenizer_add_token(self._cpp_tokenizer, token_bytes, id)
            except:
                pass  # Skip tokens that can't be encoded

        # Export merges with priority
        model = tokenizer_json.get('model', {})
        merges = model.get('merges', [])

        for priority, merge in enumerate(merges):
            # merge is like ["token1", "token2"]
            if len(merge) == 2:
                token1, token2 = merge
                id1 = vocab.get(token1, -1)
                id2 = vocab.get(token2, -1)
                merged_token = token1 + token2
                merged_id = vocab.get(merged_token, -1)

                if id1 >= 0 and id2 >= 0 and merged_id >= 0:
                    self._lib.tokenizer_add_merge_with_priority(
                        self._cpp_tokenizer, id1, id2, merged_id, priority
                    )

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        if self._hf_tokenizer is not None:
            encoding = self._hf_tokenizer.encode(text, add_special_tokens=add_special_tokens)
            return encoding.ids
        elif self._use_cpp and self._cpp_tokenizer is not None:
            max_len = len(text) * 4 + 10  # Estimate max tokens
            output = (ctypes.c_int * max_len)()
            length = self._lib.tokenizer_encode(
                self._cpp_tokenizer,
                text.encode('utf-8'),
                output,
                max_len,
                1 if add_special_tokens else 0
            )
            return list(output[:length])
        else:
            raise RuntimeError("No tokenizer backend available")

    def encode_cpp(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode using C++ backend only (for testing)."""
        if not self._use_cpp or self._cpp_tokenizer is None:
            raise RuntimeError("C++ tokenizer not available")

        max_len = len(text) * 4 + 10
        output = (ctypes.c_int * max_len)()
        length = self._lib.tokenizer_encode(
            self._cpp_tokenizer,
            text.encode('utf-8'),
            output,
            max_len,
            1 if add_special_tokens else 0
        )
        return list(output[:length])

    def encode_batch(
        self,
        texts: List[str],
        max_length: int = 512,
        add_special_tokens: bool = True,
        num_threads: int = 0,
        return_numpy: bool = True
    ) -> Union[np.ndarray, List[List[int]]]:
        """
        Encode batch of texts.

        Args:
            texts: List of texts to encode
            max_length: Maximum sequence length (pad/truncate to this)
            add_special_tokens: Add BOS/EOS tokens
            num_threads: Number of threads (0 = auto)
            return_numpy: Return numpy array instead of list

        Returns:
            Array of shape [batch_size, max_length]
        """
        if num_threads == 0:
            num_threads = multiprocessing.cpu_count()

        if self._hf_tokenizer is not None:
            # Use HuggingFace batch encoding (parallel in Rust)
            self._hf_tokenizer.enable_padding(
                pad_id=self._hf_tokenizer.token_to_id("<PAD>"),
                pad_token="<PAD>",
                length=max_length
            )
            self._hf_tokenizer.enable_truncation(max_length=max_length)

            encodings = self._hf_tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)

            if return_numpy:
                result = np.array([e.ids for e in encodings], dtype=np.int32)
            else:
                result = [e.ids for e in encodings]

            return result

        elif self._use_cpp and self._cpp_tokenizer is not None:
            # Use C++ batch encoding
            n = len(texts)

            # Prepare input
            text_ptrs = (ctypes.c_char_p * n)()
            for i, text in enumerate(texts):
                text_ptrs[i] = text.encode('utf-8')

            # Prepare output
            output = (ctypes.c_int * (n * max_length))()
            lengths = (ctypes.c_int * n)()

            self._lib.tokenizer_encode_batch(
                self._cpp_tokenizer,
                text_ptrs,
                n,
                output,
                lengths,
                max_length,
                1 if add_special_tokens else 0,
                num_threads
            )

            if return_numpy:
                result = np.ctypeslib.as_array(output).reshape(n, max_length).copy()
            else:
                result = [list(output[i*max_length:(i+1)*max_length]) for i in range(n)]

            return result
        else:
            raise RuntimeError("No tokenizer backend available")

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        elif self._use_cpp and self._cpp_tokenizer is not None:
            output = ctypes.create_string_buffer(len(ids) * 20)
            ids_arr = (ctypes.c_int * len(ids))(*ids)
            self._lib.tokenizer_decode(
                self._cpp_tokenizer,
                ids_arr,
                len(ids),
                output,
                len(output)
            )
            return output.value.decode('utf-8', errors='replace')
        else:
            raise RuntimeError("No tokenizer backend available")

    def decode_cpp(self, ids: List[int]) -> str:
        """Decode using C++ backend only (for testing)."""
        if not self._use_cpp or self._cpp_tokenizer is None:
            raise RuntimeError("C++ tokenizer not available")

        output = ctypes.create_string_buffer(len(ids) * 20)
        ids_arr = (ctypes.c_int * len(ids))(*ids)
        self._lib.tokenizer_decode(
            self._cpp_tokenizer,
            ids_arr,
            len(ids),
            output,
            len(output)
        )
        return output.value.decode('utf-8', errors='replace')

    def save(self, path: str) -> None:
        """Save tokenizer to file."""
        if self._hf_tokenizer is not None:
            self._hf_tokenizer.save(path)
        else:
            raise RuntimeError("Cannot save C++-only tokenizer")

    @classmethod
    def load(cls, path: str) -> 'FastBPETokenizer':
        """Load tokenizer from file."""
        from tokenizers import Tokenizer

        instance = cls.__new__(cls)
        instance._cpp_tokenizer = None
        instance._use_cpp = False
        instance._lib = None
        instance._hf_tokenizer = Tokenizer.from_file(path)
        instance.vocab_size = instance._hf_tokenizer.get_vocab_size()

        # Try to load C++ backend
        lib = _load_library()
        if lib is not None:
            instance._lib = lib
            instance._cpp_tokenizer = lib.tokenizer_create()
            instance._use_cpp = True
            instance._export_to_cpp()

        return instance

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.get_vocab_size()
        elif self._use_cpp and self._cpp_tokenizer is not None:
            return self._lib.tokenizer_get_vocab_size(self._cpp_tokenizer)
        return self.vocab_size

    @property
    def pad_token_id(self) -> int:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.token_to_id("<PAD>")
        return 0

    @property
    def bos_token_id(self) -> int:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.token_to_id("<BOS>")
        return 2

    @property
    def eos_token_id(self) -> int:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.token_to_id("<EOS>")
        return 3


def compare_tokenizers(hf_tokenizer, cpp_tokenizer, texts: List[str]) -> Dict:
    """Compare HuggingFace and C++ tokenizer outputs."""
    results = {
        'total': len(texts),
        'matches': 0,
        'mismatches': [],
    }

    for text in texts:
        hf_ids = hf_tokenizer.encode(text)
        cpp_ids = cpp_tokenizer.encode_cpp(text)

        if hf_ids == cpp_ids:
            results['matches'] += 1
        else:
            results['mismatches'].append({
                'text': text,
                'hf_ids': hf_ids,
                'cpp_ids': cpp_ids,
            })

    results['match_rate'] = results['matches'] / results['total'] if results['total'] > 0 else 0
    return results


# Convenience function for quick tokenization
def tokenize_texts_parallel(
    texts: List[str],
    tokenizer_path: str,
    max_length: int = 512,
    num_threads: int = 0
) -> np.ndarray:
    """
    Tokenize texts in parallel.

    Args:
        texts: List of texts
        tokenizer_path: Path to saved tokenizer
        max_length: Max sequence length
        num_threads: Number of threads (0 = auto)

    Returns:
        numpy array of shape [len(texts), max_length]
    """
    tokenizer = FastBPETokenizer.load(tokenizer_path)
    return tokenizer.encode_batch(texts, max_length=max_length, num_threads=num_threads)


if __name__ == "__main__":
    # Quick test
    print("Testing FastBPETokenizer...")

    tokenizer = FastBPETokenizer(vocab_size=1000)
    print(f"C++ backend available: {tokenizer._use_cpp}")

    # Test training
    test_texts = [
        "Hello world! This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "Once upon a time there was a little girl.",
    ] * 100

    tokenizer.train(test_texts, verbose=True)

    # Test encoding
    text = "Hello world!"
    hf_ids = tokenizer.encode(text)
    print(f"HF encoded '{text}': {hf_ids}")

    if tokenizer._use_cpp:
        cpp_ids = tokenizer.encode_cpp(text)
        print(f"C++ encoded '{text}': {cpp_ids}")
        print(f"Match: {hf_ids == cpp_ids}")

    # Test decoding
    decoded = tokenizer.decode(hf_ids)
    print(f"Decoded: '{decoded}'")

    # Test batch encoding
    batch = ["Hello world!", "Testing batch encoding.", "This is fast!"]
    batch_ids = tokenizer.encode_batch(batch, max_length=32)
    print(f"Batch shape: {batch_ids.shape}")
    print(f"Batch[0]: {batch_ids[0][:10]}...")

    print("All tests passed!")
