# Task 1 — Measured results

Project: **23l-0829_A2_Ops**. Recorded 8 November 2026 (Asia/Karachi).

These figures come from executing the audit and split pipeline on Kaggle dataset version 1.

## Environment

Python **3.12.12**, TensorFlow **2.20.0**, Keras **3.15.1**. The installed TensorFlow CPU build reports one CPU and zero visible GPUs; it was not built with CUDA. Exact package versions are saved in `reports/generated/environment/packages.txt` and dependency resolution is pinned in `uv.lock`.

## Source dataset

| Class | Files | Width min / median / max (px) | Height min / median / max (px) | File size min / median / max (bytes) |
| --- | ---: | --- | --- | --- |
| real | 436 | 183 / 920 / 8,495 | 158 / 720 / 7,091 | 11,311 / 144,840 / 10,195,352 |
| ai | 539 | 225 / 1,024 / 6,000 | 135 / 800 / 4,001 | 11,738 / 199,760 / 5,626,183 |

**975 readable images; 0 corrupt/unreadable files; 1 duplicate beyond the first copy; 0 cross-class label conflicts.**

The duplicate is `AiArtData/AiArtData/image34.jpg`, identical to `image33.jpg` (SHA-256 `f3268a4fc0477b2f3d9728f118afe32bf5196c3b476def53e34229672f77dc97`). Both source files remain intact; only one enters the split lists.

Two valid PNG images have filenames starting with a dot. They are included in the final audit. Files with `.webp`, `.gif`, or no extension decode as JPEG/PNG; file content determines format.

| Class | JPEG | PNG |
| --- | ---: | ---: |
| real | 403 | 33 |
| ai | 429 | 110 |

## Split population

| Split | Real (0) | AI (1) | Total |
| --- | ---: | ---: | ---: |
| train | 305 | 376 | 681 |
| validation | 65 | 81 | 146 |
| test | 66 | 81 | 147 |

**974 unique images; seed 42; zero content-hash overlap between splits.** The requested proportions are 70/15/15; integer rounding gives the counts above.

## Figure interpretations

### Figure 1 — Sample images

The real-labelled sample includes photographs, a painting, a cartoon, and a magazine cover; it is not restricted to unedited camera photographs. The AI-labelled sample includes stylised portraits, abstract artwork, and an illustrated dog. This mixed content means a model could learn art style or publication layout instead of reliable evidence of AI generation.

### Figure 2 — Width histogram

Most images in both classes fall below roughly 2200 pixels wide, with a longer right tail among real images extending to 8495 pixels. The AI median is 1024 pixels, compared with 920 pixels for real images, and the distributions overlap substantially. A fixed-size input pipeline is necessary, but accuracy should still be interpreted with possible resolution-related shortcuts in mind.

### Figure 3 — Height histogram

Both classes are concentrated below roughly 2200 pixels high; real images include a sparse tail reaching 7091 pixels, while AI images reach 4001 pixels. Median heights are 720 pixels for real images and 800 pixels for AI images. The sample grid also contains both portrait and landscape images, so the training pipeline must explicitly explain its aspect-ratio policy.

### Figure 4 — File-size histogram

Both distributions are strongly right-skewed, with most images in the smallest size bins and a few much larger files. The medians are 141.4 KiB for real images and 195.1 KiB for AI images, while the largest real image is about 9.7 MiB. File size is influenced by resolution, image content, and compression, so it is not a reliable provenance label by itself.

Figures: `reports/generated/eda/samples.png`, `width_histogram.png`, `height_histogram.png`, and `file_size_bytes_histogram.png`.

## Verification and remaining evidence

- Ruff lint and formatting checks passed.
- Seven data tests passed, covering corrupt data, duplicates, label conflicts, hidden filenames, decoded formats, unknown folders, and deterministic splits without shared hashes.
- The DVC pipeline executed audit, EDA, and split stages and produced `dvc.lock`.
- The CSV toy exercise restored old data and recovered both versions from a local remote after clearing its cache.
- Local and DagsHub MLflow connectivity checks logged parameters, metrics, tags, and an artifact; see `tracking_setup.md`.
- DagsHub DVC upload completed successfully: `992 files pushed`, covering the dataset and current audit, EDA, and split outputs.
- A subsequent `dvc status --cloud -j 4` confirmed: `Cache and remote 'dagshub' are in sync.`
- The separate MLflow digits lab completed two local runs (84.17% and 95.00% validation accuracy) and repeated the 0.01 learning-rate run on DagsHub (95.00%). These are toy scores, not VisionGuard results.
- Required terminal/browser screenshots and publication of the toy commits to DagsHub remain outstanding.
- Subsequent model development and test evaluation are recorded in [Task 2 results](task2_results.md).
