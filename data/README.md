# Dataset placement

Raw chest X-ray images are excluded from version control. Obtain the dataset
only from an authorised source and comply with its licence and usage terms.

Preserve the original filenames and place the images in:

```text
data/normal/
data/pneumonia/
```

Expected image counts:

- `normal`: 1,583
- `pneumonia`: 4,273

The fixed assignments in `../artifacts/splits/split_manifest.csv` use these
relative paths. Do not regenerate or reshuffle the manifest for the 320 × 320
comparison.
