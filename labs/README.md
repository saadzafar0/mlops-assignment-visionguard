# Tool labs

These exercises are separate from the real VisionGuard training and serving code.

## DVC CSV exercise (Task 1.4)

Run `python labs/dvc_toy.py` from the project root in the activated environment.
It refuses to reuse its output folder so an earlier exercise is not overwritten.
The nested Git repository and its local remote live under ignored `labs/work/`.

The completed exercise created two commits:

| Version | Commit | CSV change |
| --- | --- | --- |
| First | `851b3c4` | One real-image row |
| Second | `5a5fe62` | Add an AI-image row |

Both versions were pushed to a local DVC remote. Checking out the first Git commit
and running `dvc checkout` recovered the first CSV. After deleting that CSV and the
toy's DVC cache, `dvc pull` fetched it again from remote storage. Returning to
`main` and pulling recovered the second CSV.

The first implementation tried to use `dvc checkout` when returning to `main`
after clearing the cache. That failed because the second CSV version was no longer
cached. The final script correctly uses `dvc pull` at that point. The first attempt
is retained under ignored `labs/work/dvc-toy-first-attempt/`.

Evidence is in `reports/generated/dvc_toy/transcript.txt` and `commits.txt`.
Capture terminal screenshots of the commits and restoration steps for the report.
The toy commits have **not** been published to DagsHub; that requirement remains.

## Definitions

- **DVC pointer:** a small tracked description of data, including its path and
  checksum, which links a Git revision to bytes stored outside Git.
- **Cache:** local content-addressed copies of data that DVC can use to reconstruct
  a working dataset without downloading the same bytes again.
- **Remote:** storage used to share or recover cached data across machines.
- **`dvc checkout`:** materialises the version described by the current pointer
  from the local cache.
- **`dvc push` / `dvc pull`:** transfer cached objects to/from remote storage.

## MLflow Keras exercise (Task 3.1)

`python labs/mlflow_toy.py --target local` trains a small dense Keras classifier
on scikit-learn's bundled digits dataset, requiring no additional dataset download.
There are 1797 grayscale 8×8 images across ten digit classes. A stratified 80/20
split yields 1437 training and 360 validation images, with seed 42.

The model rescales pixel intensities by 1/16, flattens the image, uses 32 ReLU
units, and produces ten softmax outputs. Sparse categorical cross-entropy matches
the integer class labels. Both runs use Adam, batch size 64, six epochs, the same
split, and the same initialisation seed; the learning rate is the changed setting.

| Learning rate | Final validation loss | Final validation accuracy | Local run ID |
| --- | ---: | ---: | --- |
| 0.001 | 1.034715 | 84.17% | `165262983ff04ee0836bce8917dfc2c2` |
| 0.01 | 0.150037 | 95.00% | `8d8e6a75bf6b43919405a71e49d223ba` |

The larger learning rate converged faster during this short training budget. This
comparison does not establish that it is optimal for longer training or for
VisionGuard's two-class image problem.

Each run logs configuration parameters, per-epoch training and validation metrics,
a training plot, split indices, and the trained native MLflow TensorFlow model
with a signature and input example. Comparison files and plots are saved under
`reports/generated/mlflow_toy/local/`. `shuffle=False` in Keras disables a redundant
argument warning because the `tf.data` training dataset already handles shuffling.

Open the local MLflow UI using the README command and capture both run pages and
the comparison view. Repeat a run remotely with:

```bash
python labs/mlflow_toy.py --target dagshub --learning-rates 0.01
```

Remote results are saved separately under `reports/generated/mlflow_toy/dagshub/`.
Toy metrics must not be included in the VisionGuard model-comparison table.

The remote repetition completed successfully with final validation loss 0.150037
and accuracy 95.00%, matching the local run at the same learning rate.
[DagsHub run `2f366ee2f8c148b89b249bdb3da7b338`](https://dagshub.com/saadzafar0/mlops-assignment-visionguard.mlflow/#/experiments/0/runs/2f366ee2f8c148b89b249bdb3da7b338)
contains the epoch metrics, plot, split indices, source snapshot, and native model.

## Keras Tuner exercise (Task 2.7)

`python labs/tuner_toy.py` searches learning rate and hidden-unit count on the
bundled digits dataset. Random search ran four trials, each for four epochs,
using seed 42 and the same stratified 1437/360 training/validation split.

| Units | Learning rate | Best validation accuracy |
| ---: | ---: | ---: |
| 32 | 0.001 | 68.61% |
| 64 | 0.001 | 83.06% |
| 64 | 0.01 | 95.83% |
| 16 | 0.001 | 46.67% |

The best settings were **64 units and learning rate 0.01** for this short toy
budget. `reports/generated/tuner_toy/trials.csv` and `best.json` contain the
results. The VisionGuard search is separate: it warm-starts from the frozen
MobileNetV2 head and tunes learning rate, dropout, and trainable tail size over
at least eight completed trials, as described in `docs/task2.md`.
