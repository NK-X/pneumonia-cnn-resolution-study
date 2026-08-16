from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .data import NORMALISATION_MEAN, NORMALISATION_STD, load_raw_image, normalise
from .metrics import roc_curve_points
from .models import last_convolution


COLOURS = {"training": "#2E6F9E", "validation": "#E69F00", "test": "#009E73"}


def plot_history(history: list[dict[str, float]], model_name: str, output_path: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(
        epochs,
        [row["training_loss"] for row in history],
        marker="o",
        label="Training",
        color=COLOURS["training"],
    )
    axes[0].plot(
        epochs,
        [row["validation_loss"] for row in history],
        marker="o",
        label="Validation",
        color=COLOURS["validation"],
    )
    axes[0].set(xlabel="Epoch", ylabel="Binary cross-entropy loss", title="Loss")
    axes[1].plot(
        epochs,
        [row["training_auc"] for row in history],
        marker="o",
        label="Training",
        color=COLOURS["training"],
    )
    axes[1].plot(
        epochs,
        [row["validation_auc"] for row in history],
        marker="o",
        label="Validation",
        color=COLOURS["validation"],
    )
    axes[1].set(xlabel="Epoch", ylabel="ROC-AUC", title="Classification performance", ylim=(0.45, 1.01))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle(f"Training and validation: {model_name.replace('_', ' ')}", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_performance_parameters(results: list[dict[str, Any]], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    display_names = {
        "tiny_baseline": "tiny baseline",
        "deeper_baseline": "deeper baseline",
        "regularised_cnn": "regularised CNN",
        "compact_residual": "compact residual",
        "pretrained_mobilenet_v3_small": "MobileNetV3-Small (pretrained)",
    }
    annotation_offsets = {
        "tiny_baseline": (8, 8, "left"),
        "deeper_baseline": (8, 12, "left"),
        "regularised_cnn": (8, -18, "left"),
        "compact_residual": (8, 8, "left"),
        "pretrained_mobilenet_v3_small": (-8, 8, "right"),
    }
    for row in results:
        pretrained = bool(row["specification"].get("pretrained", False))
        axis.scatter(
            row["parameters"],
            row["best_validation_auc"],
            s=95,
            marker="D" if pretrained else "o",
            color="#E69F00" if pretrained else "#2E6F9E",
        )
        offset_x, offset_y, horizontal_alignment = annotation_offsets[row["model"]]
        axis.annotate(
            display_names[row["model"]],
            (row["parameters"], row["best_validation_auc"]),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=9,
            ha=horizontal_alignment,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Total model parameters (log scale)")
    axis.set_ylabel("Best validation ROC-AUC")
    axis.set_ylim(0.90, 1.005)
    axis.margins(x=0.08)
    axis.grid(alpha=0.25, which="both")
    axis.set_title("Performance versus model size", fontweight="bold")
    custom_handle = plt.Line2D([], [], marker="o", linestyle="None", color="#2E6F9E", label="Trained from scratch")
    pretrained_handle = plt.Line2D([], [], marker="D", linestyle="None", color="#E69F00", label="ImageNet pretrained")
    axis.legend(handles=(custom_handle, pretrained_handle), frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_roc(labels: np.ndarray, probabilities: np.ndarray, auc: float, output_path: Path) -> None:
    false_positive_rate, true_positive_rate = roc_curve_points(labels, probabilities)
    figure, axis = plt.subplots(figsize=(6.2, 5.3))
    axis.plot(false_positive_rate, true_positive_rate, color="#2E6F9E", linewidth=2, label=f"Selected CNN (AUC = {auc:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Chance")
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.01))
    axis.set_title("Held-out test ROC curve", fontweight="bold")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(metrics: dict[str, float | int], output_path: Path) -> None:
    matrix = np.asarray(
        [
            [metrics["true_negative"], metrics["false_positive"]],
            [metrics["false_negative"], metrics["true_positive"]],
        ]
    )
    figure, axis = plt.subplots(figsize=(5.4, 4.8))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center", fontsize=14)
    axis.set_xticks((0, 1), labels=("Normal", "Pneumonia"))
    axis.set_yticks((0, 1), labels=("Normal", "Pneumonia"))
    axis.set(xlabel="Predicted class", ylabel="True class", title="Held-out test confusion matrix")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _capture_activation(model: nn.Module, inputs: torch.Tensor) -> np.ndarray:
    captured: list[torch.Tensor] = []

    def hook(_: nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured.append(output.detach().cpu())

    handles = [
        module.register_forward_hook(hook)
        for module in model.modules()
        if isinstance(module, nn.Conv2d)
    ]
    with torch.no_grad():
        model(inputs)
    for handle in handles:
        handle.remove()
    spatially_informative = [
        activation
        for activation in captured
        if activation.ndim == 4 and min(activation.shape[-2:]) >= 8
    ]
    selected = spatially_informative[-1] if spatially_informative else captured[-1]
    return selected[0].numpy()


def create_feature_visualisation(
    model: nn.Module, image_path: Path, image_size: int, output_path: Path
) -> None:
    raw = load_raw_image(image_path, image_size)
    activation = _capture_activation(model, normalise(raw).unsqueeze(0))
    variances = activation.reshape(activation.shape[0], -1).var(axis=1)
    selected = np.argsort(-variances)[:12]
    figure, axes = plt.subplots(3, 5, figsize=(11, 7))
    axes[0, 0].imshow(raw.squeeze(0).numpy(), cmap="gray")
    axes[0, 0].set_title("Input X-ray")
    axes[0, 0].axis("off")
    flat_axes = list(axes.flat)[1:]
    for axis, channel in zip(flat_axes, selected):
        axis.imshow(activation[channel], cmap="viridis")
        axis.set_title(f"Feature {channel}")
        axis.axis("off")
    for axis in flat_axes[len(selected):]:
        axis.axis("off")
    figure.suptitle("Deep-layer feature visualisation", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_deep_dream(
    model: nn.Module,
    image_path: Path,
    image_size: int,
    output_path: Path,
    steps: int = 60,
) -> None:
    model.eval()
    original = load_raw_image(image_path, image_size)
    dreamed = (original + 0.03 * torch.randn_like(original)).clamp(0.0, 1.0).requires_grad_(True)
    target_layer = last_convolution(model)
    captured: dict[str, torch.Tensor] = {}

    def hook(_: nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured["activation"] = output

    handle = target_layer.register_forward_hook(hook)
    optimiser = torch.optim.Adam([dreamed], lr=0.025)
    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        model(normalise(dreamed).unsqueeze(0))
        activation_objective = captured["activation"].square().mean()
        horizontal_variation = (dreamed[:, :, 1:] - dreamed[:, :, :-1]).abs().mean()
        vertical_variation = (dreamed[:, 1:, :] - dreamed[:, :-1, :]).abs().mean()
        loss = -activation_objective + 0.02 * (horizontal_variation + vertical_variation)
        loss.backward()
        optimiser.step()
        with torch.no_grad():
            dreamed.clamp_(0.0, 1.0)
            smoothed = torch.nn.functional.avg_pool2d(
                dreamed.unsqueeze(0), kernel_size=3, stride=1, padding=1
            ).squeeze(0)
            dreamed.mul_(0.85).add_(smoothed, alpha=0.15)
    handle.remove()

    figure, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(original.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original X-ray")
    axes[1].imshow(dreamed.detach().squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("DeepDream image")
    difference = dreamed.detach().squeeze(0).numpy() - original.squeeze(0).numpy()
    maximum = max(1e-6, float(np.abs(difference).max()))
    axes[2].imshow(difference, cmap="coolwarm", vmin=-maximum, vmax=maximum)
    axes[2].set_title("Optimised change")
    for axis in axes:
        axis.axis("off")
    figure.suptitle("Features amplified by gradient ascent", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
