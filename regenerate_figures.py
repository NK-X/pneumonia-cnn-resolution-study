from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.pneumonia_cnn.models import ModelSpecification, build_model
from src.pneumonia_cnn.visuals import (
    create_deep_dream,
    create_feature_visualisation,
    plot_performance_parameters,
)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    result_dir = project_root / "artifacts" / "results"
    figure_dir = project_root / "artifacts" / "figures"
    results = json.loads((result_dir / "model_comparison_formal.json").read_text(encoding="utf-8"))
    selected = json.loads((result_dir / "selected_model.json").read_text(encoding="utf-8"))
    plot_performance_parameters(results, figure_dir / "performance_vs_parameters_formal.png")

    specification_data = selected["specification"]
    specification_data["channels"] = tuple(specification_data["channels"])
    specification = ModelSpecification(**specification_data)
    model = build_model(specification)
    checkpoint = torch.load(selected["model_path"], map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])

    with (result_dir / "held_out_test_predictions.csv").open("r", newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    probabilities = np.asarray(
        [float(row["probability_pneumonia"]) for row in predictions], dtype=np.float64
    )
    representative_path = Path(predictions[int(np.argmin(np.abs(probabilities - 0.5)))]["path"])
    image_size = int(checkpoint["image_size"])
    create_feature_visualisation(
        model, representative_path, image_size, figure_dir / "feature_visualisation.png"
    )
    create_deep_dream(model, representative_path, image_size, figure_dir / "deepdream.png")
    print("Figures regenerated from locked model and saved predictions; no test evaluation was run.")


if __name__ == "__main__":
    main()

