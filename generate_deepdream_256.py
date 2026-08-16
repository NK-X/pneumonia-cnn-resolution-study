from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from src.pneumonia_cnn.data import load_raw_image, load_split_manifest, normalise
from src.pneumonia_cnn.models import ModelSpecification, build_model, last_convolution


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


IMAGE_SIZE = 256
STEPS = 60
LEARNING_RATE = 0.025
TOTAL_VARIATION_WEIGHT = 0.02
SMOOTHING_WEIGHT = 0.15
DREAM_SEED = 20260816


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_grayscale(array: np.ndarray, path: Path) -> None:
    clipped = np.clip(array, 0.0, 1.0)
    pixels = np.rint(clipped * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(path)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    split_path = project_root / "artifacts" / "splits" / "split_manifest.csv"
    checkpoint_path = (
        project_root
        / "artifacts"
        / "models"
        / "pretrained_mobilenet_v3_small_resolution_256.pt"
    )
    figure_dir = project_root / "artifacts" / "figures"
    result_dir = project_root / "artifacts" / "results"
    submission_dir = project_root / "submission_images"
    for directory in (figure_dir, result_dir, submission_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records = load_split_manifest(split_path)
    candidates = sorted(
        (record for record in records if record.split == "validation" and record.label == 1),
        key=lambda record: str(record.path),
    )
    if not candidates:
        raise RuntimeError("No pneumonia image is available in the validation split.")
    source_record = candidates[0]
    if source_record.split != "validation":
        raise RuntimeError("DeepDream source selection must not access the held-out test split.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(checkpoint["image_size"]) != IMAGE_SIZE:
        raise RuntimeError("The checkpoint was not trained at 256 x 256 resolution.")
    specification = ModelSpecification(**checkpoint["specification"])
    model = build_model(specification)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    target_layer = last_convolution(model)
    target_layer_name = next(
        name for name, module in model.named_modules() if module is target_layer
    )
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured["activation"] = output

    handle = target_layer.register_forward_hook(hook)
    torch.manual_seed(DREAM_SEED)
    np.random.seed(DREAM_SEED)
    original = load_raw_image(source_record.path, IMAGE_SIZE)
    dreamed = (
        original + 0.03 * torch.randn_like(original)
    ).clamp(0.0, 1.0).requires_grad_(True)
    optimiser = torch.optim.Adam([dreamed], lr=LEARNING_RATE)
    activation_objectives: list[float] = []
    started = time.perf_counter()

    for _step in range(STEPS):
        optimiser.zero_grad(set_to_none=True)
        model(normalise(dreamed).unsqueeze(0))
        squared_activation = captured["activation"].square()
        activation_objective = squared_activation.sum() / squared_activation.numel()
        horizontal_differences = (dreamed[:, :, 1:] - dreamed[:, :, :-1]).abs()
        vertical_differences = (dreamed[:, 1:, :] - dreamed[:, :-1, :]).abs()
        horizontal_variation = horizontal_differences.sum() / horizontal_differences.numel()
        vertical_variation = vertical_differences.sum() / vertical_differences.numel()
        loss = -activation_objective + TOTAL_VARIATION_WEIGHT * (
            horizontal_variation + vertical_variation
        )
        loss.backward()
        optimiser.step()
        with torch.no_grad():
            dreamed.clamp_(0.0, 1.0)
            smoothed = torch.nn.functional.avg_pool2d(
                dreamed.unsqueeze(0), kernel_size=3, stride=1, padding=1
            ).squeeze(0)
            dreamed.mul_(1.0 - SMOOTHING_WEIGHT).add_(smoothed, alpha=SMOOTHING_WEIGHT)
        activation_objectives.append(float(activation_objective.detach()))

    elapsed_seconds = time.perf_counter() - started
    handle.remove()

    original_array = original.squeeze(0).numpy()
    dreamed_array = dreamed.detach().squeeze(0).numpy()
    difference = dreamed_array - original_array
    maximum_change = max(1e-6, float(np.abs(difference).max()))
    stem = "08_deepdream_mobilenet_v3_small_256"

    _save_grayscale(original_array, figure_dir / f"{stem}_original_256.png")
    _save_grayscale(dreamed_array, figure_dir / f"{stem}_dream_only_256.png")
    _save_grayscale(dreamed_array, submission_dir / f"{stem}_dream_only_256.png")
    np.savez_compressed(
        result_dir / f"{stem}_arrays.npz",
        original=original_array,
        dream=dreamed_array,
        difference=difference,
        activation_objective=np.asarray(activation_objectives, dtype=np.float64),
    )

    figure, axes = plt.subplots(1, 3, figsize=(7.09, 2.55), constrained_layout=True)
    figure.patch.set_facecolor("white")
    axes[0].imshow(original_array, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Original validation X-ray")
    axes[1].imshow(dreamed_array, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("DeepDream (60 steps)")
    change_image = axes[2].imshow(
        difference,
        cmap="RdBu_r",
        vmin=-maximum_change,
        vmax=maximum_change,
    )
    axes[2].set_title("Pixel change (dream − original)")
    for label, axis in zip(("a", "b", "c"), axes):
        axis.text(
            0.015,
            0.985,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
            color="white" if label != "c" else "black",
        )
        axis.set_axis_off()
    colour_bar = figure.colorbar(change_image, ax=axes[2], fraction=0.047, pad=0.025)
    colour_bar.set_label("Pixel-intensity change")
    colour_bar.ax.tick_params(labelsize=6)
    figure.suptitle(
        "DeepDream visualisation: MobileNetV3-Small at 256 × 256",
        fontsize=9,
        fontweight="bold",
    )

    figure.savefig(figure_dir / f"{stem}.svg", facecolor="white", bbox_inches="tight")
    figure.savefig(figure_dir / f"{stem}.pdf", facecolor="white", bbox_inches="tight")
    figure.savefig(
        figure_dir / f"{stem}.tiff",
        facecolor="white",
        bbox_inches="tight",
        dpi=600,
    )
    figure.savefig(
        figure_dir / f"{stem}.png",
        facecolor="white",
        bbox_inches="tight",
        dpi=600,
    )
    figure.savefig(
        submission_dir / f"{stem}.png",
        facecolor="white",
        bbox_inches="tight",
        dpi=600,
    )
    plt.close(figure)

    metadata = {
        "artifact": "DeepDream visualisation",
        "model": specification.name,
        "input_resolution": [IMAGE_SIZE, IMAGE_SIZE],
        "checkpoint_training_seed": int(checkpoint["seed"]),
        "deepdream_seed": DREAM_SEED,
        "source_split": source_record.split,
        "source_class": "pneumonia",
        "source_selection_rule": (
            "lexicographically first pneumonia image in the fixed validation manifest"
        ),
        "source_manifest_index_zero_based": records.index(source_record),
        "source_image_sha256": _sha256(source_record.path),
        "target_layer": target_layer_name,
        "target_layer_type": type(target_layer).__name__,
        "steps": STEPS,
        "learning_rate": LEARNING_RATE,
        "total_variation_weight": TOTAL_VARIATION_WEIGHT,
        "smoothing_weight": SMOOTHING_WEIGHT,
        "initial_activation_objective": activation_objectives[0],
        "final_activation_objective": activation_objectives[-1],
        "maximum_absolute_pixel_change": maximum_change,
        "elapsed_seconds": elapsed_seconds,
        "interpretation_boundary": (
            "The image displays patterns amplified to increase a late convolutional-layer "
            "activation. It is not evidence of diagnostic localisation or medical causality."
        ),
        "test_split_accessed": False,
    }
    (result_dir / f"{stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    caption = (
        "Figure 8 | DeepDream visualisation for the 256 × 256 MobileNetV3-Small model. "
        "a, A validation-split pneumonia X-ray selected by a deterministic rule before "
        "optimisation. b, The image after 60 gradient-ascent steps that maximise the "
        "normalised sum of squared activations in the model's final convolutional layer, "
        "with total-variation "
        "regularisation and local smoothing. c, The signed pixel-intensity change. This "
        "visualisation shows features favoured by the optimisation objective; it does not "
        "establish diagnostic localisation or medical causality. The held-out test split was "
        "not accessed.\n"
    )
    (figure_dir / f"{stem}_caption.txt").write_text(caption, encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(caption, end="")


if __name__ == "__main__":
    main()
