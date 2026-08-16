from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _read_json(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _plain_integer_tick(value: float, _position: float) -> str:
    if value < 1:
        return ""
    return f"{int(value):,}"


def main() -> None:
    project_root = Path(__file__).resolve().parent
    result_dir = project_root / "artifacts" / "results"
    figure_dir = project_root / "artifacts" / "figures"
    submission_dir = project_root / "submission_images"
    figure_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)

    formal_rows = _read_json(result_dir / "model_comparison_formal.json")
    resolution_rows = _read_json(result_dir / "model_comparison_resolution_256.json")
    if len(resolution_rows) != 1 or resolution_rows[0]["model"] != "pretrained_mobilenet_v3_small":
        raise RuntimeError("The 256 x 256 result must contain exactly one MobileNetV3-Small row.")

    display_names = {
        "tiny_baseline": "Tiny CNN",
        "deeper_baseline": "Deeper CNN",
        "regularised_cnn": "Regularised CNN",
        "compact_residual": "Compact residual CNN",
        "pretrained_mobilenet_v3_small": "MobileNetV3-Small",
    }
    custom_rows = [row for row in formal_rows if row["model"] != "pretrained_mobilenet_v3_small"]
    mobile_128 = next(row for row in formal_rows if row["model"] == "pretrained_mobilenet_v3_small")
    mobile_256 = resolution_rows[0]

    if int(mobile_128["parameters"]) != int(mobile_256["parameters"]):
        raise RuntimeError("Resolution comparison changed the model parameter count.")

    source_rows = [
        {
            "model": display_names[str(row["model"])],
            "input_resolution": "128 x 128",
            "parameters": int(row["parameters"]),
            "best_validation_roc_auc": float(row["best_validation_auc"]),
            "training_seed": 20260815,
        }
        for row in formal_rows
    ]
    source_rows.append(
        {
            "model": "MobileNetV3-Small",
            "input_resolution": "256 x 256",
            "parameters": int(mobile_256["parameters"]),
            "best_validation_roc_auc": float(mobile_256["best_validation_auc"]),
            "training_seed": 20260815,
        }
    )
    source_data_path = result_dir / "resolution_comparison_source_data.csv"
    with source_data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=source_rows[0].keys())
        writer.writeheader()
        writer.writerows(source_rows)

    all_parameter_counts = [int(row["parameters"]) for row in source_rows]
    if not 0 < min(all_parameter_counts):
        raise ValueError("All parameter counts must be positive before applying a log scale.")

    figure, axis = plt.subplots(figsize=(7.09, 4.25), constrained_layout=False)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    figure.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.22)

    custom_x = [int(row["parameters"]) for row in custom_rows]
    custom_y = [float(row["best_validation_auc"]) for row in custom_rows]
    axis.scatter(
        custom_x,
        custom_y,
        s=42,
        marker="o",
        color="#3B6FB6",
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )

    label_offsets = {
        "tiny_baseline": (7, 5),
        "deeper_baseline": (7, 5),
        "regularised_cnn": (7, -12),
        "compact_residual": (7, 5),
    }
    for row in custom_rows:
        axis.annotate(
            display_names[str(row["model"])],
            (int(row["parameters"]), float(row["best_validation_auc"])),
            xytext=label_offsets[str(row["model"])],
            textcoords="offset points",
            color="#234A78",
            fontsize=7.2,
        )

    mobile_x = int(mobile_128["parameters"])
    auc_128 = float(mobile_128["best_validation_auc"])
    auc_256 = float(mobile_256["best_validation_auc"])
    axis.plot(
        [mobile_x, mobile_x],
        [auc_128, auc_256],
        color="#7A7A7A",
        linewidth=0.9,
        linestyle=(0, (2.5, 2.5)),
        zorder=1,
    )
    axis.scatter(
        [mobile_x],
        [auc_128],
        s=58,
        marker="D",
        facecolor="white",
        edgecolor="#D67A16",
        linewidth=1.4,
        zorder=4,
    )
    axis.scatter(
        [mobile_x],
        [auc_256],
        s=64,
        marker="D",
        color="#C44E52",
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
    )

    vertical_order = 1 if auc_256 >= auc_128 else -1
    axis.annotate(
        f"128 × 128: {auc_128:.4f}",
        (mobile_x, auc_128),
        xytext=(-8, -14 * vertical_order),
        textcoords="offset points",
        ha="right",
        color="#A45C0D",
        fontsize=7.5,
        fontweight="bold",
    )
    axis.annotate(
        f"256 × 256: {auc_256:.4f}",
        (mobile_x, auc_256),
        xytext=(-8, 7 * vertical_order),
        textcoords="offset points",
        ha="right",
        color="#9E363A",
        fontsize=7.5,
        fontweight="bold",
    )

    axis.set_xscale("log")
    axis.xaxis.set_major_locator(LogLocator(base=10.0))
    axis.xaxis.set_major_formatter(FuncFormatter(_plain_integer_tick))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xlim(3_000, 3_000_000)
    y_min = min(custom_y + [auc_128, auc_256])
    axis.set_ylim(max(0.90, y_min - 0.012), 1.001)
    axis.set_xlabel("Total model parameters (log scale)")
    axis.set_ylabel("Best validation ROC-AUC")
    axis.set_title("Validation performance versus model size")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    axis.grid(axis="x", which="major", color="#EEEEEE", linewidth=0.5)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.text(
        0.12,
        0.04,
        "Blue circles: custom CNNs at 128 × 128. Same split, optimizer, batch size and "
        "training seed; one seed per configuration. No test-set results are shown.",
        fontsize=6.7,
        color="#555555",
    )

    stem = "07_performance_vs_model_size_128_vs_256"
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

    delta = auc_256 - auc_128
    caption = (
        "Figure 7 | Validation ROC-AUC versus total model parameters. "
        "The horizontal axis uses a logarithmic scale with parameter counts displayed as "
        "plain integers. MobileNetV3-Small contains the same number of parameters at both "
        f"input resolutions ({mobile_x:,}); only the input resolution differs. Its best "
        f"validation ROC-AUC changed from {auc_128:.4f} at 128 × 128 to {auc_256:.4f} at "
        f"256 × 256 (difference {delta:+.4f}). Results use one fixed training seed per "
        "configuration; therefore, the observed difference is descriptive rather than an "
        "estimate of a general resolution effect. The held-out test set was not evaluated.\n"
    )
    (figure_dir / f"{stem}_caption.txt").write_text(caption, encoding="utf-8")
    print(caption, end="")


if __name__ == "__main__":
    main()
