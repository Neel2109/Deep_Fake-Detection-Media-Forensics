# Synthetic demonstration examples

Every image in this directory is a project-generated illustration. No image
depicts a real person, and none is a real deepfake, authentic-media reference,
or ground-truth training/evaluation sample.

| File | Provenance | Intended demonstration |
| --- | --- | --- |
| `sample_original_illustration.png` | Hand-drawn base illustration created for this project | Baseline media intake and evidence display |
| `sample_edited_illustration.png` | Project-generated edited version of the base illustration | Edited-demo workflow; not a face-swap example |
| `sample_color-shift_illustration.png` | Derived from the base illustration with color and contrast adjustments | Color/pixel changes in descriptive image measurements |
| `sample_recompressed_illustration.png` | Derived from the base illustration by JPEG re-encoding at low quality, then stored as PNG | Compression/re-encoding workflow demonstration |

These files are not sourced from FaceForensics++, DFDC, Celeb-DF, or any
third-party dataset. The app intentionally skips classifier inference for these
bundled illustrations and labels each report as a synthetic demonstration, not
ground truth. Do not use them to claim model accuracy, detector validation, or
real-world deepfake performance.
