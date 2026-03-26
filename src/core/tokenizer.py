"""
Byte Pair Encoding (BPE) Tokenizer Implementation
Based on GPT-2 tokenization with proper merging rules
"""

import json
import regex as re
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from pathlib import Path


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer with GPT-2 style tokenization.

    Mathematical Foundation:
    - Token space: V = {v_1, v_2, ..., v_n} where n = vocab_size
    - Encoding function: E: Sigma* -> V* (string to token sequence)
    - Decoding function: D: V* -> Sigma* (token sequence to string)
    - Invariant: D(E(s)) = s for all valid strings s
    """

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.vocab: Dict[int, bytes] = {}
        self.merges: Dict[Tuple[bytes, bytes], int] = {}
        self.hf_tokenizer = None
        self.pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        # Initialize base vocabulary with byte-level tokens
        self._init_base_vocab()

    def _init_base_vocab(self):
        """Initialize vocabulary with 256 byte tokens."""
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_tokens = {
            '<PAD>': 256,
            '<UNK>': 257,
            '<BOS>': 258,
            '<EOS>': 259,
        }
        # Add special tokens to vocab
        for token, idx in self.special_tokens.items():
            self.vocab[idx] = token.encode('utf-8')

    def train(self, texts: List[str], verbose: bool = True) -> None:
        """
        Train BPE tokenizer using byte pair statistics.

        Algorithm:
        1. Initialize vocab V_0 = {byte_0, ..., byte_255}
        2. For i = 1 to vocab_size - |V_0|:
            a. Find most frequent pair (p_1, p_2) in corpus
            b. Create new token v_i = merge(p_1, p_2)
            c. Add merge rule: (p_1, p_2) -> v_i
            d. Update corpus statistics

        Time Complexity: O(n x m x vocab_size)
        where n = corpus size, m = avg sequence length
        """
        # Tokenize all texts into byte sequences
        word_freqs = Counter()
        for text in texts:
            words = self.pattern.findall(text)
            for word in words:
                word_bytes = tuple(word.encode('utf-8'))
                word_freqs[word_bytes] += 1

        # Convert to splits (each byte as separate token initially)
        splits = {
            word: [bytes([b]) for b in word]
            for word in word_freqs.keys()
        }

        num_merges = self.vocab_size - len(self.vocab)

        for i in range(num_merges):
            # Count all pairs
            pair_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                split = splits[word]
                for j in range(len(split) - 1):
                    pair = (split[j], split[j + 1])
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)

            # Create new token
            new_token_id = len(self.vocab)
            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token_id] = new_token
            self.merges[best_pair] = new_token_id

            # Update splits
            for word in word_freqs:
                split = splits[word]
                new_split = []
                j = 0
                while j < len(split):
                    if j < len(split) - 1 and (split[j], split[j + 1]) == best_pair:
                        new_split.append(new_token)
                        j += 2
                    else:
                        new_split.append(split[j])
                        j += 1
                splits[word] = new_split

            if verbose and (i + 1) % 100 == 0:
                print(f"Merge {i + 1}/{num_merges}: {best_pair[0]} + {best_pair[1]} = {new_token}")

    def _encode_word(self, word_bytes: bytes) -> List[int]:
        """
        Encode a single word using learned merge rules.

        Greedy Algorithm:
        - Start with byte-level tokens
        - Iteratively apply merge rules in learned order
        - Return final token sequence
        """
        tokens = [bytes([b]) for b in word_bytes]

        while len(tokens) > 1:
            # Find valid pairs and their merge priorities
            pairs = [(i, (tokens[i], tokens[i + 1]))
                    for i in range(len(tokens) - 1)]

            # Get pair with highest priority (earliest in training)
            valid_pairs = [(i, pair) for i, pair in pairs if pair in self.merges]

            if not valid_pairs:
                break

            # Find earliest merge
            min_idx, min_pair = min(valid_pairs, key=lambda x: self.merges[x[1]])

            # Apply merge
            new_token = self.vocab[self.merges[min_pair]]
            tokens = tokens[:min_idx] + [new_token] + tokens[min_idx + 2:]

        # Convert tokens to IDs
        token_to_id = {v: k for k, v in self.vocab.items()}
        return [token_to_id.get(t, self.special_tokens['<UNK>']) for t in tokens]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text string
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        if self.hf_tokenizer is not None:
            encoding = self.hf_tokenizer.encode(text, add_special_tokens=add_special_tokens)
            return encoding.ids

        words = self.pattern.findall(text)
        ids = []

        if add_special_tokens:
            ids.append(self.special_tokens['<BOS>'])

        for word in words:
            word_bytes = word.encode('utf-8')
            ids.extend(self._encode_word(word_bytes))

        if add_special_tokens:
            ids.append(self.special_tokens['<EOS>'])

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded text string
        """
        if self.hf_tokenizer is not None:
            return self.hf_tokenizer.decode(ids, skip_special_tokens=True)

        tokens = []
        for idx in ids:
            if idx in self.vocab:
                tokens.append(self.vocab[idx])
            elif idx in self.special_tokens.values():
                # Skip special tokens in decode
                continue

        # Concatenate all byte sequences
        byte_string = b''.join(tokens)

        # Decode to UTF-8 (handle errors gracefully)
        try:
            return byte_string.decode('utf-8')
        except UnicodeDecodeError:
            return byte_string.decode('utf-8', errors='replace')

    def save(self, path: str) -> None:
        """Save tokenizer to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert bytes to base64 for JSON serialization
        import base64
        vocab_serializable = {
            str(k): base64.b64encode(v).decode('utf-8')
            for k, v in self.vocab.items()
        }
        merges_serializable = {
            f"{base64.b64encode(k[0]).decode('utf-8')}|{base64.b64encode(k[1]).decode('utf-8')}": v
            for k, v in self.merges.items()
        }

        data = {
            'vocab_size': self.vocab_size,
            'vocab': vocab_serializable,
            'merges': merges_serializable,
            'special_tokens': self.special_tokens,
        }

        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """Load tokenizer from disk."""
        import base64

        with open(path, 'r') as f:
            data = json.load(f)

        # HuggingFace tokenizers JSON format
        if "model" in data and "pre_tokenizer" in data:
            from tokenizers import Tokenizer

            tokenizer = cls(vocab_size=0)
            tokenizer.hf_tokenizer = Tokenizer.from_file(str(path))
            tokenizer.vocab_size = tokenizer.hf_tokenizer.get_vocab_size()
            tokenizer.special_tokens = {
                '<PAD>': tokenizer.hf_tokenizer.token_to_id('<PAD>') or 0,
                '<UNK>': tokenizer.hf_tokenizer.token_to_id('<UNK>') or 0,
                '<BOS>': tokenizer.hf_tokenizer.token_to_id('<BOS>') or 0,
                '<EOS>': tokenizer.hf_tokenizer.token_to_id('<EOS>') or 0,
            }
            return tokenizer

        tokenizer = cls(vocab_size=data['vocab_size'])

        # Deserialize vocab
        tokenizer.vocab = {
            int(k): base64.b64decode(v.encode('utf-8'))
            for k, v in data['vocab'].items()
        }

        # Deserialize merges
        tokenizer.merges = {
            tuple(base64.b64decode(part.encode('utf-8')) for part in k.split('|')): v
            for k, v in data['merges'].items()
        }

        tokenizer.special_tokens = data['special_tokens']

        return tokenizer
