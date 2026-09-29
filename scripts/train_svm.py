import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from src.baseline import build_svm
from src.features import load_wu_features
from src.preprocess import kfold_splits

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def report(title, y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    print(f"\n{title}")
    print(f"  tp {tp}  fp {fp}  tn {tn}  fn {fn}")
    print(f"  accuracy    {(tp + tn) / len(y):.3f}")
    print(f"  sensitivity {tp / (tp + fn):.3f}")
    print(f"  precision   {tp / (tp + fp):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    selected = json.loads((RESULTS_DIR / "selected_features.json").read_text())
    idx = selected["indices"]
    print("features:", ", ".join(selected["names"]))

    splits, _ = load_wu_features()
    F_train, y_train = splits["train"]
    F_test, y_test = splits["test"]
    F_train, F_test = F_train[:, idx], F_test[:, idx]

    cv_pred = np.empty(len(y_train), dtype=int)
    for train, test in kfold_splits(y_train, k=args.folds, seed=args.seed):
        cv_pred[test] = build_svm().fit(F_train[train], y_train[train]).predict(F_train[test])
    report(f"collegiate, {args.folds}-fold cv (n={len(y_train)})", y_train, cv_pred)

    model = build_svm().fit(F_train, y_train)
    report(f"youth test set (n={len(y_test)})", y_test, model.predict(F_test))


if __name__ == "__main__":
    main()
