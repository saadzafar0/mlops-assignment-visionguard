# Task 2 — Measured model results

Project: **23l-0829_A2_Ops**.

## Validation comparison

| candidate                |   validation_auc |   precision |   recall |     f1 |   threshold |   cost_3FP_FN |
|:-------------------------|-----------------:|------------:|---------:|-------:|------------:|--------------:|
| baseline-lower-lr        |           0.5953 |      0.7812 |   0.3086 | 0.4425 |      0.4968 |            77 |
| baseline-no-augmentation |           0.6821 |      1.0000 |   0.1358 | 0.2391 |      0.6635 |            70 |
| baseline-reference       |           0.5901 |      0.7857 |   0.2716 | 0.4037 |      0.4719 |            77 |
| transfer-fewer-unfrozen  |           0.8636 |      0.9583 |   0.5679 | 0.7132 |      0.7406 |            41 |
| transfer-lower-dropout   |           0.8655 |      0.9600 |   0.5926 | 0.7328 |      0.7362 |            39 |
| transfer-reference       |           0.8650 |      0.9583 |   0.5679 | 0.7132 |      0.7464 |            41 |
| transfer-tuned           |           0.8750 |      0.9412 |   0.5926 | 0.7273 |      0.7225 |            42 |

Six controlled runs plus one eight-trial tuned candidate were completed. The CNN variants use manual logging; transfer variants use autologging. Tuner trials use manual logging in nested runs. Models and thresholds were selected on validation data.

Disabling CNN augmentation produced validation AUC 0.6821, compared with 0.5901 for its reference and 0.5953 for the lower learning rate. The declared ranking selected **transfer-tuned** with validation AUC 0.8750. Its validation error cost was 42, versus the lowest observed cost 39 from transfer-lower-dropout. This tradeoff was resolved before test evaluation, with ranking quality taking priority.

## Best epochs

| candidate                | stage             |   best_epoch |   best_validation_loss |
|:-------------------------|:------------------|-------------:|-----------------------:|
| baseline-lower-lr        | baseline          |            2 |                 0.6932 |
| baseline-no-augmentation | baseline          |           11 |                 0.6604 |
| baseline-reference       | baseline          |            1 |                 0.7015 |
| transfer-fewer-unfrozen  | frozen_head       |            8 |                 0.4783 |
| transfer-fewer-unfrozen  | fine_tuning       |            5 |                 0.4666 |
| transfer-lower-dropout   | frozen_head       |            8 |                 0.4741 |
| transfer-lower-dropout   | fine_tuning       |            5 |                 0.4622 |
| transfer-reference       | frozen_head       |            8 |                 0.4783 |
| transfer-reference       | fine_tuning       |            5 |                 0.4647 |
| transfer-tuned           | tuned_fine_tuning |            4 |                 0.4493 |

The best epoch is the minimum validation-loss checkpoint within each stage. Candidate selection uses the declared validation ROC-AUC policy after checkpoint selection.

## Keras Tuner

|   trial_id | status    |   validation_loss |   dropout |   unfrozen_layers |   learning_rate |
|-----------:|:----------|------------------:|----------:|------------------:|----------------:|
|          0 | COMPLETED |          0.464460 |  0.300000 |                 1 |        0.000030 |
|          1 | COMPLETED |          0.458472 |  0.100000 |                 4 |        0.000030 |
|          2 | COMPLETED |          0.449338 |  0.500000 |                 1 |        0.000100 |
|          3 | COMPLETED |          0.473581 |  0.100000 |                 2 |        0.000010 |
|          4 | COMPLETED |          0.463051 |  0.100000 |                 2 |        0.000100 |
|          5 | COMPLETED |          0.451045 |  0.300000 |                 2 |        0.000030 |
|          6 | COMPLETED |          0.463328 |  0.100000 |                 1 |        0.000100 |
|          7 | COMPLETED |          0.471225 |  0.500000 |                 4 |        0.000010 |

The largest descriptive range of grouped mean validation losses was for **learning_rate** (0.0144). These grouped differences are exploratory: the search is small and the other settings are not balanced. Choosing the best trial can overfit validation noise, so test data stayed untouched until final selection.

The best trial used learning rate 0.0001, dropout 0.5, and 1 unfrozen convolution(s), reaching validation loss 0.4493. Dropout and limited backbone updates control the amount of adaptation on this small training set. The mixed search does not isolate a causal effect for either choice.

## Locked test evaluation

| model                    |   threshold |   accuracy |   precision |   recall |     f1 |   roc_auc |
|:-------------------------|------------:|-----------:|------------:|---------:|-------:|----------:|
| baseline-no-augmentation |      0.6635 |     0.4966 |      0.8182 |   0.1111 | 0.1957 |    0.6727 |
| transfer-tuned           |      0.7225 |     0.7551 |      0.9245 |   0.6049 | 0.7313 |    0.9126 |

| model                    |   true_negative |   false_positive |   false_negative |   true_positive |
|:-------------------------|----------------:|-----------------:|-----------------:|----------------:|
| baseline-no-augmentation |              64 |                2 |               72 |               9 |
| transfer-tuned           |              62 |                4 |               32 |              49 |

The validation-selected serving champion is **transfer-tuned**. The test comparison did not change this choice or the validation-chosen thresholds. Each model inferred the same 147 held-out test images once; prediction CSVs preserve the outputs. Precision/recall/F1 treat AI as the positive class. False positives mean real images called AI-generated.

## Evidence locations

- Expanded summaries, layer tables, trainability lists, checkpoints, and curves: `models/runs/<candidate>/<stage>/` and each nested MLflow run.
- Augmented training batches and validation predictions: `reports/generated/training/<candidate>/`.
- Validation comparison chart: `reports/generated/comparison/validation_comparison.png`.
- Confusion matrices, ROC curves, stored test predictions, and error grids: `reports/generated/evaluation/`.
- Serving artifact and metadata: `models/champion/model.keras` and `manifest.json`.

Required terminal/browser screenshots remain outstanding. The figures and recorded metrics support the final report; inspect the error grids and their accompanying hypotheses rather than treating accuracy as proof of general-purpose AI detection.
