from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import binary_roc_auc, classification_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_loader(dataset: Any, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        generator=generator,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray, list[str]]:
    model.eval()
    losses: list[float] = []
    labels: list[float] = []
    probabilities: list[float] = []
    paths: list[str] = []
    for inputs, targets, batch_paths in loader:
        logits = model(inputs)
        loss = criterion(logits, targets)
        losses.append(float(loss.item()) * inputs.size(0))
        labels.extend(targets.numpy().tolist())
        probabilities.extend(torch.sigmoid(logits).numpy().tolist())
        paths.extend(batch_paths)
    label_array = np.asarray(labels, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    metrics = classification_metrics(label_array, probability_array)
    metrics["loss"] = float(sum(losses) / max(1, len(label_array)))
    return metrics, label_array, probability_array, paths


def train_model(
    model: nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    positive_weight: float,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
) -> tuple[nn.Module, list[dict[str, float]], float]:
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    optimiser = torch.optim.Adam(
        model.parameters(), learning_rate, weight_decay=weight_decay
    )
    best_state = copy.deepcopy(model.state_dict())
    best_auc = -float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    training_started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        labels: list[float] = []
        probabilities: list[float] = []
        epoch_started = time.perf_counter()
        for inputs, targets, _ in training_loader:
            optimiser.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimiser.step()
            running_loss += float(loss.item()) * inputs.size(0)
            labels.extend(targets.detach().numpy().tolist())
            probabilities.extend(torch.sigmoid(logits.detach()).numpy().tolist())

        training_auc = binary_roc_auc(np.asarray(labels), np.asarray(probabilities))
        validation_metrics, _, _, _ = evaluate(model, validation_loader, criterion)
        row = {
            "epoch": float(epoch),
            "training_loss": running_loss / max(1, len(labels)),
            "training_auc": training_auc,
            "validation_loss": float(validation_metrics["loss"]),
            "validation_auc": float(validation_metrics["roc_auc"]),
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_auc={row['training_auc']:.4f} "
            f"val_auc={row['validation_auc']:.4f} seconds={row['epoch_seconds']:.1f}",
            flush=True,
        )

        if row["validation_auc"] > best_auc + 1e-5:
            best_auc = row["validation_auc"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epoch >= 4 and epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return model, history, time.perf_counter() - training_started

