# Pneumonia CNN resolution study

This repository is a reproducible handover package for a binary chest X-ray
classification coursework project. It preserves the complete 128 × 128 and
256 × 256 code, checkpoints, numerical results and figure outputs, and prepares
a validation-only 320 × 320 continuation.

The work is an image-classification demonstration, not a clinical validation
study. No claim of medical deployment, diagnostic causality or patient-level
generalisation is supported.

## Established results

All comparisons below use the same deterministic image-level split, Adam,
batch size 128 and training seed `20260815`.

| Model/input | Total parameters | Best validation ROC-AUC | Status |
|---|---:|---:|---|
| Tiny CNN, 128 × 128 | 4,881 | 0.936037 | completed |
| Deeper CNN, 128 × 128 | 23,473 | 0.955923 | completed |
| Regularised CNN, 128 × 128 | 23,473 | 0.951651 | completed |
| Compact residual CNN, 128 × 128 | 77,169 | 0.978297 | completed |
| MobileNetV3-Small, 128 × 128 | 1,518,881 | 0.989988 | completed |
| MobileNetV3-Small, 256 × 256 | 1,518,881 | 0.993812 | completed, validation only |
| MobileNetV3-Small, 320 × 320 | 1,518,881 | not yet measured | next experiment |

The 256 × 256 result exceeded the 128 × 128 validation ROC-AUC by `0.003824`
in one fixed-seed run. This difference is descriptive rather than a general
resolution-effect estimate because no multi-seed replication has been
performed.

The 128 × 128 selected model was evaluated once on the held-out test split:
ROC-AUC `0.989850`. The lock file
`artifacts/results/test_evaluation.lock` prevents accidental repeat use. Its
SHA-256 at handover is:

```text
7DFC0B037BD06EA1AB2F9232B726CE29EA103D10EDA72B84440BDD749C90FD62
```

The 320 × 320 continuation must remain validation-only.

## Why model size remains constant

Changing input resolution changes the number of spatial positions processed,
not the number of convolutional weights. For a convolutional layer,

```text
parameters = output_channels × input_channels × kernel_height × kernel_width
```

MobileNetV3-Small uses adaptive pooling before its classifier, so the
classifier input length also remains fixed. Consequently, 128 × 128,
256 × 256 and 320 × 320 inputs use the same 1,518,881 model parameters while
their pixel counts and computational costs differ.

## Dataset contract

The raw chest X-rays are intentionally not included. Place the authorised
dataset in:

```text
data/
├── normal/
└── pneumonia/
```

Expected counts are 1,583 normal and 4,273 pneumonia images. The committed
portable split manifest preserves the original 70/15/15 image-level split:
4,100 training, 878 validation and 878 test images. Run commands from the
repository root so that the relative manifest paths resolve correctly.

The filenames do not provide reliable patient identifiers. Patient-level
leakage therefore cannot be ruled out and must remain an explicit limitation.

## Environment

The completed runs used Python 3.11.15 on Windows with CPU-only PyTorch. Create
a fresh environment rather than copying another machine's `.venv`:

```powershell
py -3.11 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
```

Exact completed-run package versions are recorded in
`requirements-lock.txt` and the run metadata JSON files. Torchvision may need
network access on first use to retrieve the standard ImageNet initial weights.

## Reproducing the completed figure outputs

The repository already includes the completed numerical results and models.
Figure-only regeneration does not retrain or access the test set:

```powershell
& '.\.venv\Scripts\python.exe' .\regenerate_figures.py
& '.\.venv\Scripts\python.exe' .\plot_resolution_comparison.py
& '.\.venv\Scripts\python.exe' .\generate_deepdream_256.py
```

## Running the 320 × 320 continuation

The side length is 2.5 times the 128 baseline, so the pixel-count factor is:

```text
(320 / 128)^2 = 6.25
```

Relative to 256 × 256, it contains `1.5625` times as many pixels. Run the
one-epoch timing pilot first:

```powershell
& '.\.venv\Scripts\python.exe' .\run_experiments.py `
  --mode resolution_320_pilot --image-size 320 --batch-size 128 --seed 20260815
```

If the pilot completes without memory or data errors, run the prespecified
validation-only experiment:

```powershell
& '.\.venv\Scripts\python.exe' .\run_experiments.py `
  --mode resolution_320 --image-size 320 --batch-size 128 --seed 20260815
```

Do not use `--mode formal`, do not pass `--allow-repeat-test`, and do not delete
or modify the test lock. If batch size 128 is infeasible, stop and document the
constraint before changing the design.

## Repository map

```text
artifacts/figures/     completed figures, including SVG/PDF/TIFF/PNG exports
artifacts/models/      trained checkpoints for completed runs
artifacts/results/     histories, metrics, metadata and test lock
artifacts/splits/      portable fixed split manifest
src/pneumonia_cnn/     data, models, training, metrics and visualisation code
submission_images/     convenient PNG copies for coursework submission
HANDOFF_PROMPT.md      copy-paste prompt for the receiving GPT
PROJECT_HANDOFF.md     eight-field technical and research handover
```

`ARTIFACT_MANIFEST_SHA256.csv` records the repository file hashes at handover.
The raw dataset and virtual environment are excluded.

## Research boundaries

- Validation ROC-AUC is the model-selection and resolution-comparison metric.
- The held-out test result belongs only to the completed 128 × 128 study.
- One seed does not quantify training variability.
- DeepDream and feature visualisations show optimisation-sensitive patterns;
  they do not establish diagnostic localisation or medical causality.
- The 256 × 256 improvement must not be described as established superiority
  without replication.

Read `PROJECT_HANDOFF.md` before modifying or running the project.
