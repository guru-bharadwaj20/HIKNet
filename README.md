# HIKNet: Head Impact Detection from Kinematic Data

UE24CS352A Machine Learning mini project.

An instrumented mouthguard records linear acceleration and angular velocity whenever it sees a spike above 10g. A lot of those recordings are not real head impacts. Chewing, spitting, biting or dropping the mouthguard can trigger it too. This project trains a 1D/2D convolutional network (HIKNet, based on PerceptionNet) to separate real impacts from false positives, and compares it against an SVM baseline built on hand-picked features.

Reference: Fanton, Gaudio, Ling, *Neural Network for Detecting Head Impacts from Kinematic Data*, CS229, Stanford.

The authors' original code (linked at the end of their report) is in `resources/Project_Code_zip/`, along with the report, poster, course guidelines and the Wu et al. (2017) paper. It has `HIKNet.py`, `RecursiveNet.py` and the MATLAB scripts they used for preprocessing and plots. The data files it loads were never shared.

## Dataset

- 527 samples: 264 real impacts, 263 false positives
- Each sample is 6 x 199: linear acceleration (x, y, z) and angular velocity (x, y, z), sampled at 1000 Hz
- Real impacts sit mostly in the 20-30 Hz range, false positives carry more high frequency content
- 70/30 train/eval split for the final model, 10-fold cross validation for architecture selection

## Setup

```bash
git clone <repo-url>
cd <repo-name>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The dataset is not public and is not part of this repo. Put the two files from the Camarillo Lab in `data/` (`.mat` files are ignored by git):

```
data/
  data.mat     key "data", shape (527, 199, 6)
  labels.mat   key "label_impact_noimpact", 1 = impact, 0 = no impact
```

Channel order is lin_acc x, y, z then ang_vel x, y, z. Same layout the original HIKNet code reads.

The lab data was never shared, so I generate a synthetic stand-in with the same shape and class balance:

```bash
python -m scripts.make_mock_data      # writes data/data.mat and data/labels.mat
```

Real impacts are simulated as damped 18-32 Hz pulses. False positives ring faster and carry more 80-200 Hz energy. 16 samples are drawn from the opposite class so the task is not trivially separable. Every HIKNet number in this repo comes from this mock data.

For the SVM baseline I use the feature set Wu et al. (2017) released with their paper (Supplementary Dataset 1, [Sci Rep 8:855](https://www.nature.com/articles/s41598-017-17864-3)). It has 411 features for 387 collegiate training samples (156 impacts, 231 non-impacts) and 32 youth test samples. It is already in the repo as `data/wu2017_features.xlsx` (published under CC BY 4.0).

`src/features.py` also computes the time domain and PSD part of the same feature set from raw traces, for when `data.mat` is available. The wavelet and head-neck model features are only in the released file.

## Running

Run from the repo root.

SVM baseline on the Wu et al. features:

```bash
python -m scripts.select_features     # forward selection, writes results/selected_features.json
python -m scripts.train_svm           # 10-fold cv + youth test set, writes results/svm_metrics.json
```

Needs `data/data.mat` and `data/labels.mat`:

```bash
python -m scripts.check_data          # shapes, class balance, channel ranges
python -m scripts.plot_eda            # time and frequency plots in figures/
```

HIKNet and RecursiveNet (PyTorch, CPU is fine):

```bash
python -m scripts.train_hiknet        # 70/30 split, early stopping, weights in checkpoints/hiknet.pt
python -m scripts.compare_models      # HIKNet vs RecursiveNet, 10-fold cv
python -m scripts.tune_head           # gap vs gmp vs dense head, 10 repeats each
python -m scripts.tune_hparams        # filters, kernel width, dropout sweeps, figures/tuning.png
python -m scripts.evaluate            # final HIKNet vs SVM table on the eval split
python -m scripts.plot_curves         # figures/roc_curves.png and figures/pr_curves.png
python -m scripts.demo -n 10          # classify a few random samples with the trained model
```

The write-up and slides are generated from the result files:

```bash
python -m scripts.make_writeup        # docs/writeup.pdf
python -m scripts.make_slides         # docs/slides.pdf
```

### HIKNet results (mock data)

These come from the synthetic dataset, so they show the pipeline works, not how the model would do on real mouthguard data.

HIKNet vs RecursiveNet, 10-fold cross validation, 30 epochs max (`results/model_comparison.json`):

| Model | Accuracy | Precision | Specificity | Sensitivity | ROC AUC | PR AUC |
|---|---|---|---|---|---|---|
| HIKNet | 93.6% | 91.5% | 90.9% | 96.2% | 0.967 | 0.961 |
| RecursiveNet | 82.0% | 75.8% | 84.8% | 79.3% | 0.882 | 0.902 |

HIKNet wins on every metric with about 28x fewer parameters (0.36M vs 10.1M, most of RecursiveNet's sit in its dense layer).

Tuning, mean ROC AUC over 10 repeats of a 70/30 split (`results/tune_head.json`, `results/tune_hparams.json`):

- Head: global average pooling 0.956, global max pooling 0.943, dense 0.717 (dense often fails to train)
- Filters 30-200 all land within 0.950-0.956, 15 filters drops to 0.949
- Kernel width 15 is clearly best (0.956), every other width has unstable runs
- Dropout 0-0.5 is flat (0.951-0.957), 0.6 starts to hurt

So the original settings (gap head, 150 filters, kernel 15, dropout 0.4) hold up.

Final comparison on the 30% eval split (n=158), from `scripts.evaluate`:

| Model | Accuracy | Precision | Specificity | Sensitivity | ROC AUC | PR AUC |
|---|---|---|---|---|---|---|
| HIKNet | 89.9% | 89.9% | 89.9% | 89.9% | 0.950 | 0.923 |
| SVM (6 selected features) | 75.9% | 87.3% | 91.1% | 60.8% | 0.910 | 0.902 |

HIKNet early stops on the same split it is scored on, so its numbers are a little optimistic.

### SVM baseline results

| | Accuracy | Precision | Specificity | Sensitivity | ROC AUC | PR AUC |
|---|---|---|---|---|---|---|
| Collegiate, 10-fold CV (n=387) | 90.2% | 96.8% | 98.3% | 78.2% | 0.978 | 0.970 |
| Youth test set (n=32) | 96.9% | 100% | 100% | 93.8% | 1.000 | 1.000 |

Wu et al. report 87.2% sensitivity and 93.2% precision with leave-one-out CV. I use 10-fold CV and default RBF settings, so the numbers differ a bit.

## Metrics

Accuracy, precision, specificity, sensitivity, ROC AUC and PR AUC on the held out set.

## License

MIT. See [LICENSE](LICENSE).
