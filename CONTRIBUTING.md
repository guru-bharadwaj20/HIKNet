# Contributing

Solo project by Guru R Bharadwaj (PES1UG24CS177). Work is split into phases and done in order, one commit per subtask.

✅ done &nbsp;&nbsp; ❌ not done

## Phase 1: Repo setup

| # | Subtask | Status |
|---|---------|:------:|
| 1.1 | README, LICENSE, .gitignore, CONTRIBUTING | ✅ |
| 1.2 | Folder structure and `requirements.txt` | ✅ |

## Phase 2: Data

| # | Subtask | Status |
|---|---------|:------:|
| 2.1 | Load mouthguard samples into 6 x 199 arrays with labels | ✅ |
| 2.2 | Check class balance (264 real / 263 false) | ✅ |
| 2.3 | Mock dataset generator (the lab data was never shared) | ✅ |

## Phase 3: Preprocessing

| # | Subtask | Status |
|---|---------|:------:|
| 3.1 | Per-channel standardization (mean / std) | ✅ |
| 3.2 | 70/30 train/eval split with fixed seed | ✅ |
| 3.3 | 10-fold cross validation helper | ✅ |

## Phase 4: EDA

| # | Subtask | Status |
|---|---------|:------:|
| 4.1 | Time domain plots of real vs false impacts | ✅ |
| 4.2 | FFT plots showing frequency content | ✅ |

## Phase 5: SVM baseline

| # | Subtask | Status |
|---|---------|:------:|
| 5.1 | Time and frequency domain feature extraction | ✅ |
| 5.2 | Sequential feature selection | ✅ |
| 5.3 | Train and evaluate SVM | ✅ |
| 5.4 | Accuracy, precision, specificity, sensitivity, ROC AUC, PR AUC | ✅ |

## Phase 6: HIKNet

| # | Subtask | Status |
|---|---------|:------:|
| 6.1 | Two 1D conv + max pool blocks, dropout 0.4 | ✅ |
| 6.2 | Late 2D conv, global average pooling, sigmoid output | ✅ |
| 6.3 | Early stopping (patience 5), save best epoch | ✅ |

## Phase 7: RecursiveNet

| # | Subtask | Status |
|---|---------|:------:|
| 7.1 | Deep 2D conv net with skip concatenations | ❌ |
| 7.2 | Compare with HIKNet under 10-fold CV | ❌ |

## Phase 8: Tuning

| # | Subtask | Status |
|---|---------|:------:|
| 8.1 | Last layer: global avg pool vs global max pool vs dense | ❌ |
| 8.2 | Sweep filters (15-200), kernel width, dropout (0-0.6) | ❌ |
| 8.3 | Repeat each run 10 times and average | ❌ |

## Phase 9: Evaluation

| # | Subtask | Status |
|---|---------|:------:|
| 9.1 | Final HIKNet vs SVM comparison table | ❌ |
| 9.2 | ROC and PR curves | ❌ |

## Phase 10: Deliverables

| # | Subtask | Status |
|---|---------|:------:|
| 10.1 | Two page write-up (PDF) | ❌ |
| 10.2 | Slide deck for review | ❌ |
| 10.3 | Demo script and final README update | ❌ |

## Workflow

- One commit per subtask, with a clear message
- Flip ❌ to ✅ in the same commit that finishes the subtask
- Don't commit `.mat` data files or model weights
