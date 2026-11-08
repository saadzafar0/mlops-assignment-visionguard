# Task 2 — Model development and evaluation

Project: **23l-0829_A2_Ops**. These are report notes; measured results are recorded
separately after the experiments finish. Screenshots remain part of submission.

## 2.1 — Input pipeline

`visionguard.pipeline.make_dataset` reads the fixed CSV split lists from Task 1.
Each path is resolved under the dataset root and decoded by file contents using
TensorFlow, with three RGB channels. This supports the PNG/JPEG content found in
the audit even when a filename has a misleading extension. Grayscale and palette
images become RGB; alpha is discarded. The same `decode_and_resize` function is
available to future serving code.

`tf.image.resize_with_pad` produces **128×128** images with antialiasing. It keeps
the aspect ratio and pads the unused area with zero-valued pixels, preserving the
whole image without stretching it. This resolution and MobileNet width were
chosen to make repeatable CPU experiments practical on the available machine;
small forensic details may be lost when large originals are downsampled.

The pipeline maps decoding in parallel, fully caches resized pixels to disk,
shuffles only the training partition, batches 32 images, and prefetches one batch.
Cache keys include the split bytes, DVC dataset pointer, preprocessing source,
and image size. Fully warming the finite cache avoids discarding partial caches
when generating an eight-image preview before training.

Horizontal flips, rotations up to about 14.4 degrees, small zooms, and contrast
changes run inside the Keras model with `training=True` only. Validation, test,
and serving calls use `training=False`; the saved model disables augmentation
automatically. Caching happens before augmentation, so transformations can vary
each epoch. Each augmented candidate logs a sample batch figure.

## 2.2 — Custom baseline CNN

The custom model uses three blocks of 3×3 convolution, batch normalisation, ReLU,
and max pooling, with 16, 32, and 64 filters. Convolutions learn local patterns;
normalisation stabilises activations; ReLU introduces nonlinear behaviour; pooling
reduces the spatial workload. Filter counts increase as spatial dimensions shrink.
Three pooling operations reduce 128×128 inputs to 16×16 feature maps. Global
average pooling reduces the 16×16×64 tensor to 64 values, limiting parameter count.
A 64-unit dense layer learns nonlinear combinations of those pooled features;
dropout regularises the small training set before the single sigmoid output.

An internal rescaling layer maps raw 0–255 RGB pixels to 0–1. The sigmoid output
is `p(AI)` because the fixed label map is real=0 and AI=1. Binary cross-entropy is
appropriate for this binary target. Adam provides adaptive gradient updates.
Accuracy, precision, recall, and ROC-AUC are logged; exact final AUC is computed
from probabilities using scikit-learn rather than thresholded predictions.

| Model/stage | Total parameters | Trainable parameters |
| --- | ---: | ---: |
| Custom CNN | 28,145 | 27,921 |
| MobileNetV2 frozen head | 707,505 | 1,281 |
| MobileNetV2 four-convolution fine-tuning | 707,505 | 325,601 |

Expanded model summaries and per-layer shape/parameter tables are saved for every
training stage and uploaded as MLflow artifacts. Nested-model parent parameter
counts include their children and must not be summed twice.

## 2.3 — Transfer learning

MobileNetV2 uses ImageNet weights, `include_top=False`, width multiplier **0.5**,
and 128×128 RGB input. The width multiplier reduces computation and parameter
count. Its new head is global average pooling, dropout, and one sigmoid unit.

In the first stage all backbone layers are frozen while the new head trains.
In the second stage only up to four convolution layers in the final block are
unfrozen: the three `block_16` convolutions and final `Conv_1` projection. Every
other backbone layer stays frozen. Batch normalisation also remains frozen, and
the backbone is called with `training=False` to preserve its ImageNet statistics.
Changing trainability is followed by recompilation.

An internal `Rescaling(1/127.5, offset=-1)` exactly implements MobileNetV2's expected
`preprocess_input` range of −1 to 1. Including it in the saved model prevents the
Flask app from accidentally applying a different normalisation later.

The frozen head starts at learning rate 0.001; fine-tuning starts at 0.00001,
100 times smaller. This limits damage to pretrained features. Each stage has its
own curve, checkpoint, best epoch, architecture table, and trainable-layer list.
The lower-validation-loss checkpoint across both stages becomes that candidate's
saved model, even when fine-tuning fails to improve the frozen-head checkpoint.

## 2.4 — Training discipline

Seed 42 controls Python/NumPy/TensorFlow through Keras's seed helper. TensorFlow
deterministic operations are enabled, with bounded thread pools. Validation and
test ordering stays fixed. A saved source commit, DVC dataset hash, split lists,
package list, lockfile, parameters, and source snapshots accompany each run.

Every stage uses early stopping on validation loss (patience 4), checkpointing the
lowest validation loss, and learning-rate reduction by 0.5 after two non-improving
epochs. CNN runs allow up to 12 epochs; transfer learning allows up to eight head
epochs and six fine-tuning epochs. Curves mark the best epoch and training may
stop sooner than the maximum.

## 2.5 — Threshold choice and test evaluation

All model comparisons and tuning use training and validation data. The final
custom CNN and final transfer-learning candidate are selected separately by
validation ROC-AUC, using validation error cost to break ties. The serving champion
is selected by that same validation policy before any test inference.

Calling a real image AI-generated is treated as more harmful because the service
could falsely accuse a creator. The chosen illustrative policy assigns cost 3 to
each false positive and cost 1 to each false negative. Thresholds minimise
`3*FP+FN` on validation predictions; ties prefer higher F1 then proximity to 0.5.
This is a stated product assumption, not a measured universal cost ratio.

`python -m visionguard.evaluate` freezes model IDs and thresholds in
`selection.json` before reading test images. Each selected model then makes one
inference pass over the 147 held-out test images. Accuracy, precision, recall, F1,
ROC-AUC, confusion matrices, and ROC curves use those stored probabilities.
Repeat invocation preserves completed results; interrupted artifact generation
can reuse existing prediction CSVs without re-running inference. The test set
does not choose a new threshold or a new model after results are known.

## 2.6 — Error analysis

For each final model, the script saves up to ten wrong test predictions, ordered
by predicted-class confidence, with true label and `p(AI)`. The image paths and
scores are also saved in CSV form. Image-specific hypotheses are written after
actually inspecting these grids. High confidence is a model score and does not
establish that the folder label is incorrect.

## 2.7 — Keras Tuner

The toy exercise uses scikit-learn's offline digits dataset and random search over
learning rate and dense-unit count. Four trials each train for four epochs; see
the toy results table in the tool-lab notes.

The VisionGuard exercise performs **at least eight successful random-search
trials** on learning rate (1e-5, 3e-5, 1e-4), dropout (0.1, 0.3, 0.5), and the
number of unfrozen final-block convolution layers (1, 2, 4). Each trial starts
from the same trained frozen-head checkpoint, keeps batch normalisation frozen,
and fine-tunes for up to four epochs. This isolates fine-tuning choices without
retraining the frozen head for each trial. Validation loss is the objective;
test predictions are unavailable to the tuner.

The completed best trial used learning rate 1e-4, dropout 0.5, and only `Conv_1`
unfrozen; its best epoch was 4 with validation loss 0.449338. Learning rate had
the largest grouped mean-loss range in this search (0.0144), compared with
0.0069 for dropout and 0.0058 for the unfrozen-layer count. Faster adaptation during
the short fine-tuning budget is one plausible explanation; the uneven small
search prevents a causal ranking of these hyperparameters.

Each trial logs its settings and epoch metrics to a nested MLflow run. The trial
table, best settings, and descriptive mean-loss differences by parameter value
are saved. These differences suggest which setting mattered in this small search
but are not causal importance estimates because random search does not balance
every other parameter value. Selecting the best of many trials can fit validation
noise; the locked test set provides the independent final check.

## Experiment tracking and commands

The six controlled candidates change learning rate, augmentation, dropout, or the
number of unfrozen layers, with one setting changed from a reference in each
variant. Baseline runs use a manual epoch callback. Transfer runs use TensorFlow
autologging, with each training stage in its own child run to avoid conflicting
optimizer parameters when fine-tuning changes the learning rate.

Autologging records optimiser/training configuration and epoch metrics; manual
logging gives explicit names and control. Neither automatically establishes the
correct dataset version, source revision, split files, threshold policy, or
product error costs, so those are logged explicitly in both modes.

```bash
python scripts/train_comparison.py
python labs/tuner_toy.py
python -m visionguard.tune
python -m visionguard.evaluate
```

The comparison script resumes completed candidates only when their settings match.
Model files are outside Git under `models/`; evidence goes to
`reports/generated/training/`, `comparison/`, and `evaluation/`. Native MLflow
models and directly loadable `.keras` files are both logged to DagsHub.

References:
[TensorFlow transfer learning](https://www.tensorflow.org/guide/keras/transfer_learning),
[Keras MobileNetV2 preprocessing](https://keras.io/api/applications/mobilenet/),
[Keras Tuner random search](https://keras.io/guides/keras_tuner/getting_started/).
