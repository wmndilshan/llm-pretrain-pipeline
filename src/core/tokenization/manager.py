"""
Unified tokenizer manager merged from the enterprise API shape and canonical backends.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union


class TokenizerManager:
    """Unified train/load/encode/decode wrapper for tokenizer backends."""

    def __init__(
        self,
        tokenizer_path: Optional[Union[str, Path]] = None,
        vocabulary_size: int = 50000,
        special_tokens: Optional[Dict[str, str]] = None,
    ):
        self.vocabulary_size = vocabulary_size
        self.special_tokens = special_tokens or {
            "pad_token": "<PAD>",
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
            "unk_token": "<UNK>",
        }
        self.tokenizer = None
        self.backend = None
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else None

        if self.tokenizer_path and self.tokenizer_path.exists():
            self.load(self.tokenizer_path)

    def train(self, texts: List[str], output_path: str, tokenizer_type: str = "bpe") -> None:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer_type = tokenizer_type.lower()
        if tokenizer_type == "bpe":
            tokenizer = Tokenizer(models.BPE(unk_token=self.special_tokens["unk_token"]))
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer.decoder = decoders.ByteLevel()
            trainer = trainers.BpeTrainer(
                vocab_size=self.vocabulary_size,
                special_tokens=list(self.special_tokens.values()),
                show_progress=True,
            )
        elif tokenizer_type == "wordpiece":
            tokenizer = Tokenizer(models.WordPiece(unk_token=self.special_tokens["unk_token"]))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            trainer = trainers.WordPieceTrainer(
                vocab_size=self.vocabulary_size,
                special_tokens=list(self.special_tokens.values()),
                show_progress=True,
            )
        elif tokenizer_type == "unigram":
            tokenizer = Tokenizer(models.Unigram())
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            trainer = trainers.UnigramTrainer(
                vocab_size=self.vocabulary_size,
                special_tokens=list(self.special_tokens.values()),
                show_progress=True,
            )
        else:
            raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type}")

        tokenizer.train_from_iterator(texts, trainer=trainer)
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.special_tokens['bos_token']} $A {self.special_tokens['eos_token']}",
            pair=(
                f"{self.special_tokens['bos_token']} $A {self.special_tokens['eos_token']} "
                f"$B:1 {self.special_tokens['eos_token']}:1"
            ),
            special_tokens=[
                (self.special_tokens["bos_token"], tokenizer.token_to_id(self.special_tokens["bos_token"])),
                (self.special_tokens["eos_token"], tokenizer.token_to_id(self.special_tokens["eos_token"])),
            ],
        )

        tokenizer.save(str(output_dir / "tokenizer.json"))
        with open(output_dir / "special_tokens.json", "w", encoding="utf-8") as f:
            json.dump(self.special_tokens, f, indent=2)

        self.tokenizer = tokenizer
        self.backend = tokenizer_type
        self.tokenizer_path = output_dir

    def load(self, tokenizer_path: Union[str, Path]) -> None:
        from tokenizers import Tokenizer

        tokenizer_path = Path(tokenizer_path)
        if tokenizer_path.is_dir():
            tokenizer_file = tokenizer_path / "tokenizer.json"
            special_tokens_file = tokenizer_path / "special_tokens.json"
        else:
            tokenizer_file = tokenizer_path
            special_tokens_file = tokenizer_path.with_name("special_tokens.json")

        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self.backend = "hf"
        self.vocabulary_size = self.tokenizer.get_vocab_size()
        self.tokenizer_path = tokenizer_path if tokenizer_path.is_dir() else tokenizer_path.parent

        if special_tokens_file.exists():
            with open(special_tokens_file, "r", encoding="utf-8") as f:
                self.special_tokens = json.load(f)

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        if max_length is not None:
            if truncation:
                self.tokenizer.enable_truncation(max_length=max_length)
            elif hasattr(self.tokenizer, "no_truncation"):
                self.tokenizer.no_truncation()

            if padding:
                self.tokenizer.enable_padding(
                    pad_id=self.pad_token_id or 0,
                    pad_token=self.special_tokens["pad_token"],
                    length=max_length,
                )
            elif hasattr(self.tokenizer, "no_padding"):
                self.tokenizer.no_padding()
        else:
            if hasattr(self.tokenizer, "no_truncation"):
                self.tokenizer.no_truncation()
            if hasattr(self.tokenizer, "no_padding"):
                self.tokenizer.no_padding()

        if isinstance(text, str):
            return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids
        return [enc.ids for enc in self.tokenizer.encode_batch(text, add_special_tokens=add_special_tokens)]

    def decode(self, token_ids, skip_special_tokens: bool = True):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        if not token_ids:
            return ""
        if token_ids and isinstance(token_ids[0], int):
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        return [self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens) for ids in token_ids]

    def get_vocab_size(self) -> int:
        if self.tokenizer is None:
            return self.vocabulary_size
        return self.tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> Optional[int]:
        if self.tokenizer is None:
            return None
        return self.tokenizer.token_to_id(token)

    def get_special_token_id(self, token_name: str) -> Optional[int]:
        token = self.special_tokens.get(token_name)
        if token is None:
            raise ValueError(f"Unknown special token: {token_name}")
        return self.token_to_id(token)

    @property
    def pad_token_id(self) -> Optional[int]:
        return self.get_special_token_id("pad_token")

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.get_special_token_id("bos_token")

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.get_special_token_id("eos_token")

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> "TokenizerManager":
        manager = cls()
        manager.load(path)
        return manager
