from __future__ import annotations

import argparse
from pathlib import Path

from src.pneumonia_cnn.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare compact CNNs for pneumonia classification."
    )
    parser.add_argument(
        "--mode",
        choices=(
            "pilot",
            "pretrained_pilot",
            "formal",
            "resolution_256_pilot",
            "resolution_256",
            "resolution_320_pilot",
            "resolution_320",
        ),
        default="pilot",
        help=(
            "pilot trains the smallest custom model for one epoch; pretrained_pilot times "
            "MobileNetV3-Small; formal performs the prespecified comparison; "
            "resolution_256_pilot, resolution_256, resolution_320_pilot and resolution_320 "
            "run validation-only MobileNetV3-Small resolution checks without evaluating "
            "the test split."
        ),
    )
    parser.add_argument("--normal-dir", type=Path, default=Path("data/normal"))
    parser.add_argument("--pneumonia-dir", type=Path, default=Path("data/pneumonia"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--allow-repeat-test",
        action="store_true",
        help="Explicitly permit a repeated formal test evaluation after a lock file exists.",
    )
    parser.add_argument(
        "--skip-wide",
        action="store_true",
        help="Exclude the wide residual candidate after a documented compute-limit stop.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_experiment(
        project_root=Path(__file__).resolve().parent,
        mode=arguments.mode,
        normal_dir=arguments.normal_dir,
        pneumonia_dir=arguments.pneumonia_dir,
        image_size=arguments.image_size,
        batch_size=arguments.batch_size,
        epochs_override=arguments.epochs,
        seed=arguments.seed,
        allow_repeat_test=arguments.allow_repeat_test,
        skip_wide=arguments.skip_wide,
    )
