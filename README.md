# HIKNet: Head Impact Detection from Kinematic Data

UE24CS352A Machine Learning mini project.

An instrumented mouthguard records linear acceleration and angular velocity whenever it sees a spike above 10g. A lot of those recordings are not real head impacts. Chewing, spitting, biting or dropping the mouthguard can trigger it too. This project trains a 1D/2D convolutional network (HIKNet, based on PerceptionNet) to separate real impacts from false positives, and compares it against an SVM baseline built on hand-picked features.

Reference: Fanton, Gaudio, Ling, *Neural Network for Detecting Head Impacts from Kinematic Data*, CS229, Stanford.

The authors' original code (linked at the end of their report) is extracted locally under `resources/Project_Code_zip/` and kept out of git. It has `HIKNet.py`, `RecursiveNet.py` and the MATLAB scripts they used for preprocessing and plots. The data files it loads were never shared.

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

The dataset is not public and is not part of this repo. Put the two files from the Camarillo Lab in `data/` (ignored by git):

```
data/
  data.mat     key "data", shape (527, 199, 6)
  labels.mat   key "label_impact_noimpact", 1 = impact, 0 = no impact
```

Channel order is lin_acc x, y, z then ang_vel x, y, z. Same layout the original HIKNet code reads.

For the SVM baseline we use the feature set Wu et al. (2017) released with their paper (Supplementary Dataset 1, [Sci Rep 8:855](https://www.nature.com/articles/s41598-017-17864-3)). It has 411 features for 387 collegiate training samples (156 impacts, 231 non-impacts) and 32 youth test samples. Download it and save it as `data/wu2017_features.xlsx`:

```bash
curl -o data/wu2017_features.xlsx "https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-017-17864-3/MediaObjects/41598_2017_17864_MOESM2_ESM.xlsx"
```

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

HIKNet commands will be added in part 2.

### SVM baseline results

| | Accuracy | Precision | Specificity | Sensitivity | ROC AUC | PR AUC |
|---|---|---|---|---|---|---|
| Collegiate, 10-fold CV (n=387) | 90.2% | 96.8% | 98.3% | 78.2% | 0.978 | 0.970 |
| Youth test set (n=32) | 96.9% | 100% | 100% | 93.8% | 1.000 | 1.000 |

Wu et al. report 87.2% sensitivity and 93.2% precision with leave-one-out CV. We use 10-fold CV and default RBF settings, so the numbers differ a bit.

## Metrics

Accuracy, precision, specificity, sensitivity, ROC AUC and PR AUC on the held out set.

## Team

| Name | SRN |
|------|-----|
| Guru R Bharadwaj | PES1UG24CS177 |
| Chukkapalli Rohan | PES1UG24CS135 |

Work split is in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
