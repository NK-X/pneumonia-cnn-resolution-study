from __future__ import annotations

import csv
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .data import (
    ChestXrayDataset,
    ImageRecord,
    audit_images,
    discover_images,
    load_split_manifest,
    make_stratified_split,
    save_split_manifest,
)
from .metrics import bootstrap_auc_interval
from .models import FORMAL_SPECIFICATIONS, ModelSpecification, build_model, count_parameters
from .training import evaluate, make_loader, seed_everything, train_model
from .visuals import (
    create_deep_dream,
    create_feature_visualisation,
    plot_confusion_matrix,
    plot_history,
    plot_performance_parameters,
    plot_roc,
)


def _write_json(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, ensure_ascii=False)


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def _read_history(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _records_for(records: list[ImageRecord], split: str) -> list[ImageRecord]:
    return [record for record in records if record.split == split]


def _positive_weight(training_records: list[ImageRecord]) -> float:
    positives = sum(record.label == 1 for record in training_records)
    negatives = len(training_records) - positives
    return negatives / positives


def _select_model(results: list[dict[str, Any]]) -> dict[str, Any]:
    best_auc = max(float(row["best_validation_auc"]) for row in results)
    eligible = [row for row in results if float(row["best_validation_auc"]) >= best_auc - 0.005]
    return min(eligible, key=lambda row: int(row["parameters"]))


def run_experiment(
    project_root: Path,
    mode: str,
    normal_dir: Path,
    pneumonia_dir: Path,
    image_size: int,
    batch_size: int,
    epochs_override: int | None,
    seed: int,
    allow_repeat_test: bool,
    skip_wide: bool,
) -> None:
    artifact_root = project_root / "artifacts"
    figure_dir = artifact_root / "figures"
    result_dir = artifact_root / "results"
    model_dir = artifact_root / "models"
    split_dir = artifact_root / "splits"
    for directory in (figure_dir, result_dir, model_dir, split_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = split_dir / "split_manifest.csv"
    if manifest_path.exists():
        records = load_split_manifest(manifest_path)
    else:
        records = make_stratified_split(discover_images(normal_dir, pneumonia_dir), seed)
        save_split_manifest(records, manifest_path)
    audit_counts = audit_images(records)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "seed": seed,
        "image_size": image_size,
        "batch_size": batch_size,
        "optimizer": "Adam",
        "primary_model_selection_metric": "validation ROC-AUC",
        "selection_rule": "highest validation AUC; if within 0.005, choose fewer parameters",
        "split_policy": "deterministic stratified image-level 70/15/15 split",
        "patient_level_split_available": False,
        "audit_counts": audit_counts,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
        "skip_wide_residual": skip_wide,
    }
    _write_json(result_dir / f"run_metadata_{mode}.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)

    training_records = _records_for(records, "training")
    validation_records = _records_for(records, "validation")
    test_records = _records_for(records, "test")
    resolution_sizes = {
        "resolution_256_pilot": 256,
        "resolution_256": 256,
        "resolution_320_pilot": 320,
        "resolution_320": 320,
    }
    if mode in resolution_sizes and image_size != resolution_sizes[mode]:
        raise ValueError(
            f"{mode} requires --image-size {resolution_sizes[mode]}; received {image_size}."
        )

    if mode == "formal":
        specifications = FORMAL_SPECIFICATIONS
    elif mode == "pretrained_pilot" or mode in resolution_sizes:
        specifications = FORMAL_SPECIFICATIONS[-1:]
    else:
        specifications = FORMAL_SPECIFICATIONS[:1]
    if skip_wide:
        specifications = tuple(item for item in specifications if item.name != "wide_residual")
        _write_json(
            result_dir / "deviation_compute_stop_wide_residual.json",
            {
                "candidate": "wide_residual",
                "status": "excluded after engineering compute-limit stop; not ranked",
                "reason": (
                    "The third epoch exceeded 20 minutes after the first two epochs required "
                    "297.8 and 521.6 seconds under concurrent CPU load."
                ),
                "partial_observations_not_used_for_selection": [
                    {"epoch": 1, "training_auc": 0.8841, "validation_auc": 0.9043, "seconds": 297.8},
                    {"epoch": 2, "training_auc": 0.9005, "validation_auc": 0.9121, "seconds": 521.6},
                ],
            },
        )
    full_training_modes = {"formal", "resolution_256", "resolution_320"}
    max_epochs = epochs_override if epochs_override is not None else (12 if mode in full_training_modes else 1)
    patience = 3
    positive_weight = _positive_weight(training_records)

    results: list[dict[str, Any]] = []
    for specification in specifications:
        print(f"\nmodel={specification.name}", flush=True)
        model_path = model_dir / f"{specification.name}_{mode}.pt"
        history_path = result_dir / f"history_{specification.name}_{mode}.csv"
        if mode == "formal" and model_path.exists() and history_path.exists():
            history = _read_history(history_path)
            model = build_model(specification)
            parameters = count_parameters(model)
            elapsed_seconds = sum(row["epoch_seconds"] for row in history)
            best_validation_auc = max(row["validation_auc"] for row in history)
            print(
                f"resume=existing best_val_auc={best_validation_auc:.4f} epochs={len(history)}",
                flush=True,
            )
            results.append(
                {
                    "model": specification.name,
                    "parameters": parameters,
                    "best_validation_auc": best_validation_auc,
                    "epochs_completed": len(history),
                    "training_seconds": elapsed_seconds,
                    "seconds_per_epoch": elapsed_seconds / len(history),
                    "specification": asdict(specification),
                    "model_path": str(model_path),
                }
            )
            continue
        seed_everything(seed)
        training_dataset = ChestXrayDataset(
            training_records,
            image_size=image_size,
            augment=specification.augmentation,
            noise_std=specification.noise_std,
        )
        validation_dataset = ChestXrayDataset(
            validation_records, image_size=image_size, augment=False, noise_std=0.0
        )
        training_loader = make_loader(training_dataset, batch_size, True, seed)
        validation_loader = make_loader(validation_dataset, batch_size, False, seed)
        model = build_model(specification)
        parameters = count_parameters(model)
        model, history, elapsed_seconds = train_model(
            model=model,
            training_loader=training_loader,
            validation_loader=validation_loader,
            positive_weight=positive_weight,
            learning_rate=1e-3,
            weight_decay=specification.weight_decay,
            max_epochs=max_epochs,
            patience=patience,
        )
        best_validation_auc = max(row["validation_auc"] for row in history)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "specification": asdict(specification),
                "image_size": image_size,
                "seed": seed,
            },
            model_path,
        )
        _write_history(history_path, history)
        plot_history(history, specification.name, figure_dir / f"training_validation_{specification.name}_{mode}.png")
        results.append(
            {
                "model": specification.name,
                "parameters": parameters,
                "best_validation_auc": best_validation_auc,
                "epochs_completed": len(history),
                "training_seconds": elapsed_seconds,
                "seconds_per_epoch": elapsed_seconds / len(history),
                "specification": asdict(specification),
                "model_path": str(model_path),
            }
        )

    _write_json(result_dir / f"model_comparison_{mode}.json", results)
    plot_performance_parameters(results, figure_dir / f"performance_vs_parameters_{mode}.png")

    if mode != "formal":
        print(
            "Validation-only run complete. The held-out test split has not been evaluated.",
            flush=True,
        )
        return

    selected = _select_model(results)
    _write_json(result_dir / "selected_model.json", selected)
    test_lock = result_dir / "test_evaluation.lock"
    if test_lock.exists() and not allow_repeat_test:
        raise RuntimeError(
            "The formal test set has already been evaluated. Use --allow-repeat-test only with an explicit reason."
        )

    specification = next(item for item in specifications if item.name == selected["model"])
    selected_model = build_model(specification)
    checkpoint = torch.load(selected["model_path"], map_location="cpu", weights_only=True)
    selected_model.load_state_dict(checkpoint["state_dict"])
    test_dataset = ChestXrayDataset(test_records, image_size=image_size, augment=False, noise_std=0.0)
    test_loader = make_loader(test_dataset, batch_size, False, seed)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    test_metrics, test_labels, test_probabilities, test_paths = evaluate(
        selected_model, test_loader, criterion
    )
    lower, upper = bootstrap_auc_interval(test_labels, test_probabilities, seed)
    test_metrics["roc_auc_bootstrap_95_ci_lower"] = lower
    test_metrics["roc_auc_bootstrap_95_ci_upper"] = upper
    test_metrics["selected_model"] = selected["model"]
    test_metrics["test_images"] = len(test_records)
    _write_json(result_dir / "held_out_test_metrics.json", test_metrics)

    with (result_dir / "held_out_test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "label", "probability_pneumonia"))
        writer.writeheader()
        for path, label, probability in zip(test_paths, test_labels, test_probabilities):
            writer.writerow({"path": path, "label": int(label), "probability_pneumonia": float(probability)})

    plot_roc(test_labels, test_probabilities, float(test_metrics["roc_auc"]), figure_dir / "held_out_test_roc.png")
    plot_confusion_matrix(test_metrics, figure_dir / "held_out_test_confusion_matrix.png")
    representative_index = int(np.argmin(np.abs(test_probabilities - 0.5)))
    representative_path = Path(test_paths[representative_index])
    create_feature_visualisation(
        selected_model, representative_path, image_size, figure_dir / "feature_visualisation.png"
    )
    create_deep_dream(selected_model, representative_path, image_size, figure_dir / "deepdream.png")
    test_lock.write_text(
        f"Formal test evaluated once at {datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )
    print(json.dumps(test_metrics, indent=2), flush=True)
