"""LightGBM 5-fold baseline for Predicting Irrigation Need.

Run from repo root:
    uv run python -m src.baseline
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

from src.data import encode, load_raw
from src.paths import CATEGORICAL, CLASSES, SUBMISSIONS
from src.threshold import apply_weights, tune_weights

N_FOLDS = 5
SEED = 42

LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "verbose": -1,
    "n_jobs": -1,
    "seed": SEED,
}
N_BOOST_ROUND = 3000
EARLY_STOP = 100


def run(use_class_weight: bool = True, tag: str = "baseline", use_features: bool = False) -> dict:
    t0 = time.time()
    print(f"\n=== Baseline run: tag={tag}  class_weight={'balanced' if use_class_weight else 'none'}  features={use_features} ===")

    tr_df, te_df = load_raw()
    X, X_test, y, cat_maps = encode(tr_df, te_df, use_features=use_features)
    print(f"feature columns ({len(X.columns)}): {list(X.columns)}")

    if use_class_weight:
        priors = np.bincount(y) / len(y)
        sample_weight_full = (1.0 / priors)[y]
    else:
        sample_weight_full = np.ones(len(y), dtype=float)

    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_pred = np.zeros((len(X_test), 3), dtype=np.float32)
    importance_gain = np.zeros(len(X.columns), dtype=np.float64)
    importance_split = np.zeros(len(X.columns), dtype=np.float64)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\n-- fold {fold+1}/{N_FOLDS} --")
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        wtr = sample_weight_full[tr_idx]
        wva = sample_weight_full[va_idx]

        dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr, categorical_feature=CATEGORICAL)
        dva = lgb.Dataset(Xva, label=yva, weight=wva, categorical_feature=CATEGORICAL, reference=dtr)

        booster = lgb.train(
            LGB_PARAMS,
            dtr,
            num_boost_round=N_BOOST_ROUND,
            valid_sets=[dva],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOP, verbose=False),
                lgb.log_evaluation(period=200),
            ],
        )
        oof[va_idx] = booster.predict(Xva, num_iteration=booster.best_iteration)
        test_pred += booster.predict(X_test, num_iteration=booster.best_iteration) / N_FOLDS

        importance_gain  += booster.feature_importance(importance_type="gain")
        importance_split += booster.feature_importance(importance_type="split")

        fold_pred = np.argmax(oof[va_idx], axis=1)
        print(f"fold {fold+1} balanced_acc (argmax): {balanced_accuracy_score(yva, fold_pred):.5f} "
              f"best_iter={booster.best_iteration}")

    # OOF metrics
    argmax_pred = np.argmax(oof, axis=1)
    ba_argmax = balanced_accuracy_score(y, argmax_pred)
    print("\n=== OOF results (argmax) ===")
    print(f"balanced_acc: {ba_argmax:.5f}")
    print(classification_report(y, argmax_pred, target_names=CLASSES, digits=4))
    print("Confusion matrix (rows=true, cols=pred, order=Low/Medium/High):")
    print(confusion_matrix(y, argmax_pred))

    # Threshold/weight tuning
    print("\n=== OOF results (tuned weights) ===")
    weights, ba_tuned = tune_weights(oof, y)
    tuned_pred = apply_weights(oof, weights)
    print(f"weights={weights.round(4).tolist()}  balanced_acc: {ba_tuned:.5f}")
    print(classification_report(y, tuned_pred, target_names=CLASSES, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y, tuned_pred))

    # Build submission with tuned weights
    SUBMISSIONS.mkdir(exist_ok=True, parents=True)
    test_argmax = np.argmax(test_pred, axis=1)
    test_tuned = apply_weights(test_pred, weights)

    sub_path = SUBMISSIONS / f"sub_{tag}_argmax.csv"
    pd.DataFrame({
        "id": te_df["id"].values,
        "Irrigation_Need": np.array(CLASSES)[test_argmax],
    }).to_csv(sub_path, index=False)
    print(f"wrote {sub_path}")

    sub_tuned_path = SUBMISSIONS / f"sub_{tag}_tuned.csv"
    pd.DataFrame({
        "id": te_df["id"].values,
        "Irrigation_Need": np.array(CLASSES)[test_tuned],
    }).to_csv(sub_tuned_path, index=False)
    print(f"wrote {sub_tuned_path}")

    elapsed = time.time() - t0
    summary = {
        "tag": tag,
        "use_class_weight": use_class_weight,
        "ba_argmax": float(ba_argmax),
        "ba_tuned": float(ba_tuned),
        "weights": weights.tolist(),
        "elapsed_sec": round(elapsed, 1),
    }
    print(f"\nsummary: {json.dumps(summary, indent=2)}")
    Path(SUBMISSIONS / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    np.save(SUBMISSIONS / f"oof_{tag}.npy", oof)
    np.save(SUBMISSIONS / f"test_pred_{tag}.npy", test_pred)

    imp_df = pd.DataFrame({
        "feature": list(X.columns),
        "gain": importance_gain / N_FOLDS,
        "split": importance_split / N_FOLDS,
    }).sort_values("gain", ascending=False)
    imp_df.to_csv(SUBMISSIONS / f"importance_{tag}.csv", index=False)
    print("\nTop 20 features by gain:")
    print(imp_df.head(20).to_string(index=False))
    return summary


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "v0"
    if mode == "v0":
        run(use_class_weight=False, tag="lgbm_v0_unweighted")
        run(use_class_weight=True, tag="lgbm_v0_balanced")
    elif mode == "v1":
        run(use_class_weight=False, tag="lgbm_v1_features", use_features=True)
    else:
        raise SystemExit(f"unknown mode: {mode}")
