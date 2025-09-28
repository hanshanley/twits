"""Utility classes and helpers for working with the Twits toxicity datasets.

This module bundles the small amount of boilerplate we need for
preprocessing CSV/TSV inputs, tokenising text with HuggingFace
transformers, and constructing the PyTorch ``Dataset`` objects consumed by
our training and inference pipelines.

All classes deliberately avoid performing any on-the-fly random
augmentations so that evaluation and inference remain deterministic.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


DEFAULT_CONTEXT_TOKENIZER = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_TOXICITY_TOKENIZER = "microsoft/deberta-v3-base"


def preprocess_text(text: str) -> str:
    """Collapse excess whitespace and strip control characters from ``text``."""

    return " ".join(text.replace("\n", " ").replace("\t", " ").split())


class QueryContentDataset(Dataset):
    """Dataset of (query, context, label) tuples used for contrastive pretraining."""

    def __init__(
        self,
        samples: Sequence[Tuple[str, str, int]],
        tokenizer_name: str = DEFAULT_CONTEXT_TOKENIZER,
        max_length: int = 256,
    ) -> None:
        self.samples: List[Tuple[str, str, int]] = list(samples)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[str, str, int]:
        return self.samples[index]

    def _encode(self, texts: Sequence[str]) -> Tuple[torch.LongTensor, torch.LongTensor]:
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return encoded["input_ids"].long(), encoded["attention_mask"].long()

    def collate_fn(self, batch: Sequence[Tuple[str, str, int]]):
        queries = [item[0] for item in batch]
        contexts = [item[1] for item in batch]
        labels = torch.tensor([int(item[2]) - 1 for item in batch], dtype=torch.long)

        token_ids_1, attention_mask_1 = self._encode(queries)
        token_ids_2, attention_mask_2 = self._encode(contexts)

        return {
            "token_ids_1": token_ids_1,
            "attention_mask_1": attention_mask_1,
            "token_ids_2": token_ids_2,
            "attention_mask_2": attention_mask_2,
            "labels": labels,
        }


class ToxicityClassificationDataset(Dataset):
    """Classification dataset with integer toxicity labels."""

    def __init__(
        self,
        samples: Sequence[Tuple[str, int]],
        tokenizer_name: str = DEFAULT_TOXICITY_TOKENIZER,
        max_length: int = 196,
    ) -> None:
        self.samples: List[Tuple[str, int]] = list(samples)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[str, int]:
        return self.samples[index]

    def collate_fn(self, batch: Sequence[Tuple[str, int]]):
        texts = [item[0] for item in batch]
        labels = torch.tensor([int(item[1]) for item in batch], dtype=torch.long)

        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "token_ids_1": encoded["input_ids"].long(),
            "attention_mask_1": encoded["attention_mask"].long(),
            "labels": labels,
        }


class ToxicityRegressionDataset(Dataset):
    """Regression dataset with continuous toxicity scores."""

    def __init__(
        self,
        samples: Sequence[Tuple[str, float]],
        tokenizer_name: str = DEFAULT_TOXICITY_TOKENIZER,
        max_length: int = 128,
    ) -> None:
        self.samples: List[Tuple[str, float]] = list(samples)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[str, float]:
        return self.samples[index]

    def collate_fn(self, batch: Sequence[Tuple[str, float]]):
        texts = [item[0] for item in batch]
        labels = torch.tensor([float(item[1]) for item in batch], dtype=torch.float)

        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "token_ids_1": encoded["input_ids"].long(),
            "attention_mask_1": encoded["attention_mask"].long(),
            "labels": labels,
        }


class ToxicityInferenceDataset(Dataset):
    """Dataset of (id, text) pairs used during model inference."""

    def __init__(
        self,
        samples: Sequence[Tuple[str, str]],
        tokenizer_name: str = DEFAULT_TOXICITY_TOKENIZER,
        max_length: int = 512,
    ) -> None:
        self.samples: List[Tuple[str, str]] = list(samples)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        return self.samples[index]

    def collate_fn(self, batch: Sequence[Tuple[str, str]]):
        identifiers = [item[0] for item in batch]
        texts = [item[1] for item in batch]

        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "token_ids_1": encoded["input_ids"].long(),
            "attention_mask_1": encoded["attention_mask"].long(),
            "ids": identifiers,
        }


# ---------------------------------------------------------------------------
# CSV loaders


def _read_rows(file_path: Path, delimiter: str = "\t") -> Iterable[List[str]]:
    with file_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        yield from reader


def load_query_context_data(file_path: Path, delimiter: str = "\t") -> Tuple[List[Tuple[str, str, int]], int]:
    rows = list(_read_rows(Path(file_path), delimiter))
    samples = [(row[0], row[1], int(row[2])) for row in rows]
    labels = {row[2] for row in rows}
    return samples, len(labels)


def load_toxic_regression_data(file_path: Path, delimiter: str = "\t") -> List[Tuple[str, float]]:
    rows = list(_read_rows(Path(file_path), delimiter))
    return [(preprocess_text(row[1]), float(row[2])) for row in rows]


def load_toxic_inference_data(file_path: Path, delimiter: str = "\t") -> List[Tuple[str, str]]:
    rows = list(_read_rows(Path(file_path), delimiter))
    return [(row[0], preprocess_text(row[1])) for row in rows]


__all__ = [
    "QueryContentDataset",
    "ToxicityClassificationDataset",
    "ToxicityRegressionDataset",
    "ToxicityInferenceDataset",
    "load_query_context_data",
    "load_toxic_regression_data",
    "load_toxic_inference_data",
    "preprocess_text",
]
