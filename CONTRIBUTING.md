# Contributing

Two people, one repo. Work is split into two halves and done in order: Guru finishes Part 1 first, then Rohan picks up Part 2 on top of it.

✅ done &nbsp;&nbsp; ❌ not done

## Part 1: Guru R Bharadwaj (PES1UG24CS177)

Data, preprocessing and the SVM baseline.

| # | Task | Subtask | Status |
|---|------|---------|:------:|
| 1 | Repo setup | README, LICENSE, .gitignore, CONTRIBUTING | ✅ |
| 2 | | Folder structure and `requirements.txt` | ✅ |
| 3 | Data loading | Load mouthguard samples into 6 x 199 arrays with labels | ✅ |
| 4 | | Check class balance (264 real / 263 false) | ✅ |
| 5 | Preprocessing | Per-channel standardization (mean / std) | ✅ |
| 6 | | 70/30 train/eval split with fixed seed | ✅ |
| 7 | | 10-fold cross validation helper | ✅ |
| 8 | EDA | Time domain plots of real vs false impacts | ✅ |
| 9 | | FFT plots showing frequency content | ✅ |
| 10 | Baseline | Time and frequency domain feature extraction | ✅ |
| 11 | | Sequential feature selection | ✅ |
| 12 | | Train and evaluate SVM | ❌ |
| 13 | Metrics | Accuracy, precision, specificity, sensitivity, ROC AUC, PR AUC | ❌ |

## Part 2: Chukkapalli Rohan (PES1UG24CS135)

Starts once Part 1 is merged. Neural networks, tuning and final deliverables.

| # | Task | Subtask | Status |
|---|------|---------|:------:|
| 1 | HIKNet | Two 1D conv + max pool blocks, dropout 0.4 | ❌ |
| 2 | | Late 2D conv, global average pooling, sigmoid output | ❌ |
| 3 | | Early stopping (patience 5), save best epoch | ❌ |
| 4 | RecursiveNet | Deep 2D conv net with skip concatenations | ❌ |
| 5 | | Compare with HIKNet under 10-fold CV | ❌ |
| 6 | Tuning | Last layer: global avg pool vs global max pool vs dense | ❌ |
| 7 | | Sweep filters (15-200), kernel width, dropout (0-0.6) | ❌ |
| 8 | | Repeat each run 10 times and average | ❌ |
| 9 | Evaluation | Final HIKNet vs SVM comparison table | ❌ |
| 10 | | ROC and PR curves | ❌ |
| 11 | Deliverables | Two page write-up (PDF) | ❌ |
| 12 | | Slide deck for review | ❌ |
| 13 | | Demo script and final README update | ❌ |

## Workflow

- Branch off `main` for each task, e.g. `data-loading`, `hiknet`
- Keep commits small with clear messages
- Open a PR and get the other person to look at it before merging
- Don't commit datasets, model weights or anything in `resources/`
- Flip ❌ to ✅ in the table in the same PR that finishes the task
