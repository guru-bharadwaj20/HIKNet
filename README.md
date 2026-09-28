# HIKNet: Head Impact Detection from Kinematic Data

UE24CS352A Machine Learning mini project.

An instrumented mouthguard records linear acceleration and angular velocity whenever it sees a spike above 10g. A lot of those recordings are not real head impacts. Chewing, spitting, biting or dropping the mouthguard can trigger it too. This project trains a 1D/2D convolutional network (HIKNet, based on PerceptionNet) to separate real impacts from false positives, and compares it against an SVM baseline built on hand-picked features.

Reference: Fanton, Gaudio, Ling, *Neural Network for Detecting Head Impacts from Kinematic Data*, CS229, Stanford.

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

Put the dataset files inside `data/`. The folder is ignored by git.

## Running

```bash
python train.py --model hiknet
python train.py --model svm
python evaluate.py --model hiknet
```

Commands and flags will be updated as the code is added.

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
