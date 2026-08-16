# Project handover

## 1. Objective and scope

The project compares convolutional neural-network performance and model size for
normal-versus-pneumonia chest X-ray classification. The immediate continuation
is a controlled 320 × 320 MobileNetV3-Small experiment. It tests whether the
descriptive validation improvement observed at 256 × 256 persists when the
input contains 6.25 times the pixel count of the 128 × 128 baseline.

## 2. Established completed state

- Fixed split: 4,100 training, 878 validation and 878 test images.
- Optimiser: Adam; learning rate: `1e-3`; batch size: `128`.
- Training seed: `20260815`; maximum epochs: `12`; early-stopping patience: `3`.
- MobileNetV3-Small parameters: `1,518,881` at both completed resolutions.
- Best validation ROC-AUC at 128 × 128: `0.9899879539485377`.
- Best validation ROC-AUC at 256 × 256: `0.9938124107242771`.
- Descriptive validation difference: `+0.00382445677573939`.
- The 256 run stopped after 7 epochs and required `1208.87995330001` seconds.
- One fixed seed was used; no uncertainty interval for the resolution effect is
  available.

## 3. Dataset and split

Raw images are excluded from version control. Place them under `data/normal`
and `data/pneumonia`. Preserve filenames exactly because
`artifacts/splits/split_manifest.csv` encodes the fixed assignments. The split
is stratified at image level, not patient level. No trustworthy patient ID is
available, so patient-level leakage remains unresolved.

## 4. Environment and reproducible commands

The completed environment used Python 3.11.15, PyTorch 2.13.0+cpu,
torchvision 0.28.0+cpu, NumPy 2.4.6, Pillow 12.3.0 and Matplotlib 3.11.1.
Create a fresh environment and install `requirements.txt`. Run the 320 pilot
and full validation commands exactly as shown in `README.md`.

## 5. Key evidence and file entry points

- `run_experiments.py`: command-line entry point; 320 modes are prepared.
- `src/pneumonia_cnn/experiment.py`: experiment conditions and test boundary.
- `artifacts/results/model_comparison_formal.json`: completed 128 results.
- `artifacts/results/model_comparison_resolution_256.json`: completed 256 result.
- `artifacts/results/history_pretrained_mobilenet_v3_small_resolution_256.csv`:
  epoch-level 256 evidence.
- `artifacts/models/pretrained_mobilenet_v3_small_resolution_256.pt`: best 256
  checkpoint.
- `artifacts/results/resolution_comparison_source_data.csv`: source data for the
  completed comparison plot.
- `artifacts/figures/07_performance_vs_model_size_128_vs_256.*`: completed
  resolution comparison.
- `artifacts/figures/08_deepdream_mobilenet_v3_small_256*`: 256 DeepDream
  outputs and caption.

## 6. Non-negotiable research-integrity boundaries

The test split has already been evaluated once for the completed 128 study.
Do not delete, overwrite or bypass `artifacts/results/test_evaluation.lock`.
Do not invoke `--allow-repeat-test`. The 256 and 320 studies are validation
comparisons only. Do not reinterpret a DeepDream image as a lesion map. Do not
report single-seed differences as statistically established effects.

Test-lock SHA-256 at handover:

```text
7DFC0B037BD06EA1AB2F9232B726CE29EA103D10EDA72B84440BDD749C90FD62
```

## 7. Next experiment and stopping criteria

Run `resolution_320_pilot` for one epoch. Continue only if data loading,
memory use and checkpoint writing succeed at batch size 128. Then run
`resolution_320`, which uses at most 12 epochs and patience 3. Stop when the
script completes, an unrecoverable resource error occurs, or preserving the
fixed design becomes impossible. Do not silently lower batch size, alter data,
change augmentation, change seed or increase the epoch budget.

After completion, add a new 128/256/320 comparison figure. All three
MobileNetV3-Small points must share the same x-coordinate because their
parameter counts are identical. The x-axis must remain logarithmic while its
tick labels display ordinary integer counts rather than powers of ten.

## 8. Reporting and unresolved questions

Report established facts, supported inferences, assumptions and unsupported
conclusions separately. Compare validation ROC-AUC, elapsed time and per-epoch
time. Preserve the one-seed limitation. Confirm that the test-lock hash is
unchanged before and after the run. Multi-seed replication and patient-level
splitting remain outside the current coursework scope unless explicitly
authorised.
