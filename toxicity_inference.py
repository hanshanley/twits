"""Inference and evaluation helpers for the Twits contrastive toxicity model."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel

from toxicity_data import (
    DEFAULT_TOXICITY_TOKENIZER,
    ToxicityInferenceDataset,
    ToxicityRegressionDataset,
    load_toxic_inference_data,
    load_toxic_regression_data,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContrastiveToxicityRegressor(nn.Module):
    """DeBERTa-based regressor with an auxiliary contrastive objective."""

    def __init__(
        self,
        config: argparse.Namespace,
        base_model: str = DEFAULT_TOXICITY_TOKENIZER,
        cache_dir: str | None = "cache",
    ) -> None:
        super().__init__()
        self.num_labels = getattr(config, "num_labels", 1)

        self.model = AutoModel.from_pretrained(base_model, cache_dir=cache_dir)
        self.toxicity_agn = AutoModel.from_pretrained(base_model, cache_dir=cache_dir)

        for encoder in (self.model, self.toxicity_agn):
            for param in encoder.parameters():
                param.requires_grad = True

        hidden_size = self.model.config.hidden_size
        self.temperature = 0.07
        self.scale = math.sqrt(float(hidden_size))
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
        self.factcheck_head = nn.Linear(hidden_size * 2, hidden_size)
        self.average_factcheck_head = nn.Linear(hidden_size * 2, hidden_size)
        self.cosine = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.attention_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor) -> torch.Tensor:
        primary = self.model(input_ids, attention_mask=attention_mask)["last_hidden_state"][:, 0, :]
        primary = self.dropout(primary)
        aux = self.toxicity_agn(input_ids, attention_mask=attention_mask)["last_hidden_state"][:, 0, :]
        aux = self.dropout(aux)

        attended = self._attention(primary, self.attention_linear(aux))
        combined = torch.cat((attended, primary), dim=-1)
        combined = self.average_factcheck_head(combined)
        combined = self.relu(combined)
        combined = self.batch_norm(combined)
        combined = self.output_layer(combined)
        combined = self.relu(combined)
        return combined.squeeze(-1)

    def compute_training_loss(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, bool]:
        contrastive_labels = labels > 0.5
        has_positive = bool(torch.any(contrastive_labels))
        has_negative = bool(torch.any(~contrastive_labels))

        aux_embeddings = self.toxicity_agn(input_ids, attention_mask=attention_mask)[
            "last_hidden_state"
        ][:, 0, :]
        aux_embeddings = self.dropout(aux_embeddings)

        contrastive_loss = torch.tensor(0.0, device=input_ids.device)
        if has_positive and has_negative:
            dot = aux_embeddings @ aux_embeddings.t()
            mask = self._pair_mask(contrastive_labels.int())
            square_norm = torch.diag(dot)
            denominator = torch.sqrt(square_norm).unsqueeze(0) * torch.sqrt(square_norm).unsqueeze(1)
            exp_logits = torch.exp(dot / (self.temperature * denominator + 1e-16))

            positive_sum = torch.sum(mask * exp_logits, dim=1) + 1e-16
            negative_mask = (~torch.eye(dot.size(0), device=input_ids.device).bool()).float()
            negative_sum = torch.sum(negative_mask * exp_logits, dim=1) + 1e-16
            contrastive_loss = -torch.log(positive_sum / negative_sum).mean()

        predictions = self.forward(input_ids, attention_mask)
        regression_loss = F.mse_loss(predictions, labels, reduction="mean")
        total_loss = regression_loss + contrastive_loss
        return total_loss, has_positive and has_negative

    def predict(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(input_ids, attention_mask)

    def _attention(self, inputs: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        # ``inputs``: (batch, hidden); we add a singleton length dimension
        inputs = inputs.unsqueeze(1)
        sim = torch.einsum("blh,bh->bl", inputs, query) / self.scale
        weights = torch.softmax(sim, dim=1)
        context = torch.einsum("blh,bl->bh", inputs, weights)
        return context

    def _pair_mask(self, labels: torch.Tensor) -> torch.Tensor:
        equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        indices_equal = torch.eye(labels.size(0), device=labels.device, dtype=torch.bool)
        mask = equal & ~indices_equal
        return mask.float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run toxicity model inference.")
    parser.add_argument("model_checkpoint", type=Path, help="Path to the trained model checkpoint (.pt)")
    parser.add_argument("input_file", type=Path, help="Tab-separated file containing text and optionally a label")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size to use during inference (default: 64)"
    )
    parser.add_argument(
        "--tokenizer", type=str, default=DEFAULT_TOXICITY_TOKENIZER, help="Tokenizer to load for inference"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional path to save predictions as TSV"
    )
    parser.add_argument(
        "--has-labels",
        action="store_true",
        help="Set if the input file contains labels in the third column for evaluation",
    )
    return parser.parse_args()


def _build_dataloader(
    samples: List[Tuple[str, float]] | List[Tuple[str, str]],
    tokenizer_name: str,
    batch_size: int,
    inference: bool,
) -> DataLoader:
    if inference:
        dataset = ToxicityInferenceDataset(samples, tokenizer_name=tokenizer_name)
    else:
        dataset = ToxicityRegressionDataset(samples, tokenizer_name=tokenizer_name)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)


def load_model(checkpoint_path: Path) -> Tuple[ContrastiveToxicityRegressor, argparse.Namespace]:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    config = checkpoint["model_config"]
    model = ContrastiveToxicityRegressor(config)
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()
    return model, config


def run_inference(
    model: ContrastiveToxicityRegressor,
    dataloader: DataLoader,
    has_labels: bool,
) -> Tuple[List[str] | None, np.ndarray, np.ndarray | None]:
    predictions: List[float] = []
    labels: List[float] = []
    identifiers: List[str] | None = [] if has_labels else []

    with torch.no_grad():
        for batch in dataloader:
            ids = batch.get("ids")
            input_ids = batch["token_ids_1"].to(DEVICE)
            attention_mask = batch["attention_mask_1"].to(DEVICE)
            batch_preds = model.predict(input_ids, attention_mask).cpu().numpy()
            predictions.extend(batch_preds.tolist())

            if has_labels:
                labels.extend(batch["labels"].cpu().numpy().tolist())
            if ids is not None:
                identifiers.extend(ids)

    pred_array = np.asarray(predictions)
    label_array = np.asarray(labels) if has_labels else None
    id_list = identifiers if identifiers else None
    return id_list, pred_array, label_array


def save_predictions(path: Path, identifiers: List[str] | None, predictions: np.ndarray, labels: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if identifiers is None:
            identifiers = [str(i) for i in range(len(predictions))]
        if labels is None:
            for sample_id, score in zip(identifiers, predictions):
                handle.write(f"{sample_id}\t{score:.6f}\n")
        else:
            for sample_id, score, label in zip(identifiers, predictions, labels):
                handle.write(f"{sample_id}\t{label:.6f}\t{score:.6f}\n")


def main() -> None:
    args = parse_args()
    model, config = load_model(args.model_checkpoint)

    if args.has_labels:
        samples = load_toxic_regression_data(args.input_file)
    else:
        samples = load_toxic_inference_data(args.input_file)

    dataloader = _build_dataloader(samples, args.tokenizer, args.batch_size, inference=not args.has_labels)
    identifiers, predictions, labels = run_inference(model, dataloader, has_labels=args.has_labels)

    if labels is not None:
        mae = float(np.mean(np.abs(predictions - labels)))
        rmse = float(np.sqrt(np.mean((predictions - labels) ** 2)))
        print(f"MAE: {mae:.4f}\tRMSE: {rmse:.4f}")

    if args.output is not None:
        save_predictions(args.output, identifiers, predictions, labels)

    if labels is None:
        for score in predictions:
            print(f"{score:.6f}")


if __name__ == "__main__":
    main()
