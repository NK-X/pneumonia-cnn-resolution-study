# Copy-paste prompt for the receiving GPT

You are taking over a completed CPU-based CNN coursework repository. Act as an
independent machine-learning engineer and methodologically conservative research
assistant. Work from the repository files as the authoritative technical
record; do not invent missing results or reinterpret smoke tests as scientific
evidence.

## 1. Objective

Run the prepared MobileNetV3-Small experiment at 320 × 320 input resolution.
This contains 6.25 times the pixel count of the 128 × 128 baseline:

```text
(320 / 128)^2 = 6.25
```

The primary question is whether validation ROC-AUC changes when input
resolution increases while model architecture, parameter count, data split,
optimiser, batch size, seed and stopping rule remain fixed.

## 2. Read before acting

Read these files completely:

1. `README.md`
2. `PROJECT_HANDOFF.md`
3. `src/pneumonia_cnn/experiment.py`
4. `src/pneumonia_cnn/training.py`
5. `artifacts/results/model_comparison_formal.json`
6. `artifacts/results/model_comparison_resolution_256.json`
7. `artifacts/results/history_pretrained_mobilenet_v3_small_resolution_256.csv`
8. `artifacts/results/test_evaluation.lock`

Verify `ARTIFACT_MANIFEST_SHA256.csv` before modifying anything. Confirm that
the raw images are present under `data/normal` and `data/pneumonia` and that the
counts are 1,583 and 4,273 respectively. Do not regenerate the existing split.

## 3. Established baseline evidence

- MobileNetV3-Small parameters: 1,518,881.
- 128 × 128 best validation ROC-AUC: 0.9899879539485377.
- 256 × 256 best validation ROC-AUC: 0.9938124107242771.
- Observed difference: +0.00382445677573939.
- Training seed: 20260815.
- Adam learning rate: 1e-3.
- Batch size: 128.
- Maximum epochs: 12; early-stopping patience: 3.
- The 256 result is based on one seed and is descriptive.

## 4. Non-negotiable test boundary

The held-out test split was already used once for the completed 128 × 128
study. The 256 and 320 experiments are validation-only. Never:

- run `--mode formal`;
- pass `--allow-repeat-test`;
- delete or modify `artifacts/results/test_evaluation.lock`;
- use test predictions to tune, select or present the 320 model.

Record the test-lock SHA-256 before and after the experiment. It must remain:

```text
7DFC0B037BD06EA1AB2F9232B726CE29EA103D10EDA72B84440BDD749C90FD62
```

## 5. Execution sequence

Create a fresh Python 3.11 environment and install `requirements.txt`. Run from
the repository root.

First run the timing pilot:

```powershell
& '.\.venv\Scripts\python.exe' .\run_experiments.py `
  --mode resolution_320_pilot --image-size 320 --batch-size 128 --seed 20260815
```

Inspect the pilot metadata, history, model checkpoint and terminal output. If
it completes without memory or data errors, run:

```powershell
& '.\.venv\Scripts\python.exe' .\run_experiments.py `
  --mode resolution_320 --image-size 320 --batch-size 128 --seed 20260815
```

Do not change the experimental conditions merely to obtain a higher score. If
batch size 128 is infeasible, stop, preserve logs and request a design decision.

## 6. Required outputs

Preserve the automatic 320 checkpoint, history CSV, run metadata JSON,
comparison JSON and training/validation figure. Then create a new comparison
figure containing the completed custom CNNs and MobileNetV3-Small at 128,
256 and 320. Requirements:

- y-axis: best validation ROC-AUC;
- x-axis: total model parameters on a logarithmic scale;
- x-axis labels: ordinary counts such as `10,000`, `100,000`, `1,000,000`,
  never `10^n`;
- the 128/256/320 MobileNet points have the identical x-coordinate;
- no horizontal jitter that implies a parameter difference;
- directly label the three resolutions and exact AUC values;
- state that each configuration has one training seed and no test results are
  included;
- export SVG, PDF, TIFF at 600 dpi and PNG.

Generate a 320 DeepDream only after the training result is safely saved. Use
the same deterministic validation-image selection rule as
`generate_deepdream_256.py`; never select a test image.

## 7. Interpretation rules

Separate the final report into:

- established facts;
- supported inferences;
- assumptions;
- unresolved uncertainty;
- unsupported conclusions that must not be stated.

Do not call a validation difference an accuracy difference: the metric is
ROC-AUC. Do not claim that higher resolution is generally superior from one
seed. Do not interpret DeepDream textures as medical pathology localisation.

## 8. Completion criteria

The task is complete only when:

1. the pilot and full 320 validation run have completed or a documented
   resource limitation prevents the fixed protocol;
2. all numerical outputs and checkpoints exist and are readable;
3. the 128/256/320 figure has been visually inspected at final size;
4. the test-lock hash is unchanged;
5. no test-set inference has occurred;
6. the exact commands, runtime and limitations are reported;
7. the new files are committed without overwriting the 128/256 evidence.

Begin with a read-only audit and report whether the repository, dataset and
environment satisfy these prerequisites before starting the pilot.
