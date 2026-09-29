"""Phase 6 report: every perception table and figure, from one script with fixed seeds.

    python scripts/fetch_nsl_kdd.py          # once
    python scripts/perception_report.py      # writes figures/perception/
    python scripts/perception_report.py --quick   # smaller samples, for a fast check

Outputs
    model_comparison.csv      CV and test scores, complexity, and the 1-SE choice
    learning_curves.png       train / validation macro-F1 against training size
    validation_curves.png     against tree depth and SVM C
    decision_tree_rules.txt   rules extracted for the production system
    ttc_regression.csv        Ridge / Lasso / gradient boosting, raw and log target
    lasso_path.png            Lasso coefficients against alpha (distractors enter last)
    fgsm.png                  accuracy against epsilon, standard vs adversarially trained MLP
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegis_system.perception.adversarial_test import adversarial_training, robustness_curves  # noqa: E402
from aegis_system.perception.classifier import (  # noqa: E402
    calibrated_svc, compare_models, learning_curves, model_zoo, one_standard_error_choice, tree_paths,
    validation_curves,
)
from aegis_system.perception.datasets import (  # noqa: E402
    NSL_CATEGORICAL, NSL_MUTABLE, load_nsl_kdd, nsl_kdd_available, stratified_sample,
)
from aegis_system.perception.regressor import (  # noqa: E402
    compare_regressors, generate_ttc_dataset, lasso_coefficient_path,
)

SEED = 0
# Bias-variance curves use weighted F1: the training samples hold only a few u2r rows, and
# macro-F1 skips classes absent from a validation fold, which can put validation above training.
# Model selection (model_comparison.csv) uses macro-F1 as the plan specifies.
CURVE_SCORING = "f1_weighted"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def classification(out: Path, quick: bool) -> None:
    train_n, test_n = (8000, 4000) if quick else (30000, None)
    X, y = load_nsl_kdd("train", sample=train_n, seed=SEED)
    X_test, y_test = load_nsl_kdd("test", sample=test_n, seed=SEED) if test_n else load_nsl_kdd("test")
    log(f"NSL-KDD: {len(X)} training rows (stratified sample), {len(X_test)} test rows")

    rows, fitted = compare_models(X, y, X_test, y_test, model_zoo(SEED, fast=quick), seed=SEED,
                                  categorical=NSL_CATEGORICAL, n_jobs=-1)
    choice = one_standard_error_choice(rows)
    for r in rows:
        r["chosen_by_1se"] = r["model"] == choice
        r["params"] = json.dumps(r["params"])
    fields = ["model", "params", "train_rows", "cv_f1_macro", "cv_f1_std", "cv_f1_se", "test_f1_macro",
              "test_accuracy", "test_roc_auc_ovr", "complexity", "fit_seconds", "chosen_by_1se"]
    with (out / "model_comparison.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    log(f"model comparison written; one-standard-error choice: {choice}")

    rules = sorted(tree_paths(fitted["decision_tree"], min_samples=20), key=lambda r: -r.samples)
    (out / "decision_tree_rules.txt").write_text("\n".join(str(r) for r in rules) + "\n")
    log(f"{len(rules)} decision-tree rules with at least 20 samples")

    Xc, yc = stratified_sample(X, y, 4000 if quick else 12000, SEED)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, name in zip(axes, ["decision_tree", "mlp"]):
        curve = learning_curves(fitted[name], Xc, yc, seed=SEED, scoring=CURVE_SCORING)
        ax.plot(curve["sizes"], curve["train"], marker="o", label="training")
        ax.plot(curve["sizes"], curve["validation"], marker="o", label="cross-validation")
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("Training rows")
        ax.legend()
    axes[0].set_ylabel("Weighted F1")
    fig.suptitle("Learning curves (gap = variance, low plateau = bias)")
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=150)
    plt.close(fig)
    log("learning curves done")

    from sklearn.tree import DecisionTreeClassifier
    from aegis_system.perception.classifier import make_pipeline

    depth = validation_curves(make_pipeline(DecisionTreeClassifier(random_state=SEED), Xc, NSL_CATEGORICAL),
                              Xc, yc, "max_depth", [1, 2, 4, 6, 8, 12, 16, 24], seed=SEED,
                              scoring=CURVE_SCORING)
    Xs, ys = stratified_sample(X, y, 2000 if quick else 4000, SEED)
    svm_c = validation_curves(make_pipeline(calibrated_svc(SEED), Xs, NSL_CATEGORICAL), Xs, ys,
                              "estimator__C", [0.01, 0.1, 1, 10, 100], seed=SEED, scoring=CURVE_SCORING)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, curve, label, logx in ((axes[0], depth, "Tree max_depth", False), (axes[1], svm_c, "SVM C", True)):
        ax.plot(curve["values"], curve["train"], marker="o", label="training")
        ax.plot(curve["values"], curve["validation"], marker="o", label="cross-validation")
        ax.set_xlabel(label)
        if logx:
            ax.set_xscale("log")
        ax.legend()
    axes[0].set_ylabel("Weighted F1")
    fig.suptitle("Validation curves: model complexity against fit")
    fig.tight_layout()
    fig.savefig(out / "validation_curves.png", dpi=150)
    plt.close(fig)
    log("validation curves done")

    eps = [0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
    mlp = fitted["mlp"]
    robust = adversarial_training(mlp, X, y, NSL_MUTABLE, eps=1.0, seed=SEED)
    Xa, ya = stratified_sample(X_test, y_test, 2000 if quick else 5000, SEED)
    curves = robustness_curves(mlp, robust, Xa, ya, NSL_MUTABLE, eps)
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in (("standard", "Standard MLP"), ("adversarially_trained", "Adversarially trained (eps=1.0)")):
        ax.plot([r["epsilon"] for r in curves[key]], [r["accuracy"] for r in curves[key]], marker="o", label=label)
    ax.set_xlabel("FGSM epsilon (standard deviations, mutable features only)")
    ax.set_ylabel("Accuracy on KDDTest+")
    ax.set_title("White-box FGSM against the MLP detector")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fgsm.png", dpi=150)
    plt.close(fig)
    (out / "fgsm.json").write_text(json.dumps(curves, indent=2))
    log("FGSM done")


def regression(out: Path) -> None:
    X, y = generate_ttc_dataset(2000, seed=SEED)
    rows = []
    for log_target in (False, True):
        r, _ = compare_regressors(X, y, seed=SEED, log_target=log_target)
        rows += r
    with (out / "ttc_regression.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "target", "rmse", "mae", "r2", "alpha", "zeroed", "coefficients"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "zeroed": json.dumps(r.get("zeroed", "")),
                        "coefficients": json.dumps({k: round(v, 4) for k, v in r.get("coefficients", {}).items()})})
    path = lasso_coefficient_path(X, y)
    fig, ax = plt.subplots(figsize=(7, 4))
    for f, coefs in path["coefficients"].items():
        ax.plot(path["alphas"], coefs, label=f)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Lasso alpha (decreasing)")
    ax.set_ylabel("Standardized coefficient")
    ax.set_title("Lasso path for time-to-compromise")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "lasso_path.png", dpi=150)
    plt.close(fig)
    log("regression done")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("figures") / "perception")
    parser.add_argument("--quick", action="store_true", help="smaller samples and grids")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    warnings.filterwarnings("ignore", category=UserWarning)  # rare-class fold warnings (u2r has 52 rows)
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
    regression(args.out)
    if not nsl_kdd_available():
        sys.exit("NSL-KDD not found: run `python scripts/fetch_nsl_kdd.py` first (regression outputs were written).")
    classification(args.out, args.quick)
    for f in sorted(args.out.iterdir()):
        print(f)


if __name__ == "__main__":
    main()
