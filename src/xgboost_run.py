"""XGBoost 5-fold runner for Predicting Irrigation Need.

Same fold seed as LightGBM/CatBoost so OOF arrays line up for stacking.

Run from repo root:
    uv run python -m src.xgboost_run
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

from src.data import encode, load_raw
from src.paths import CLASSES, SUBMISSIONS
from src.threshold import apply_weights, tune_weights

N_FOLDS = 5
SEED = 42

XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "device": "cuda",
    "seed": SEED,
}
N_ROUND = 3000
EARLY_STOP = 100


def run(tag: str = "xgb_v0") -> dict:
    t0 = time.time()
    print(f"\n=== XGBoost run: tag={tag} ===")

    tr_df, te_df = load_raw()
    X, X_test, y, _ = encode(tr_df, te_df, use_features=False)

    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_pred = np.zeros((len(X_test), 3), dtype=np.float32)
    importance = pd.Series(0.0, index=X.columns)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\n-- fold {fold+1}/{N_FOLDS} --")
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        dtr = xgb.DMatrix(Xtr, label=ytr)
        dva = xgb.DMatrix(Xva, label=yva)
        dte = xgb.DMatrix(X_test)

        booster = xgb.train(
            XGB_PARAMS,
            dtr,
            num_boost_round=N_ROUND,
            evals=[(dva, "val")],
            early_stopping_rounds=EARLY_STOP,
            verbose_eval=200,
        )
        oof[va_idx] = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
        test_pred += booster.predict(dte, iteration_range=(0, booster.best_iteration + 1)) / N_FOLDS

        score = booster.get_score(importance_type="gain")
        for f, v in score.items():
            importance[f] = importance.get(f, 0.0) + v / N_FOLDS

        fold_pred = np.argmax(oof[va_idx], axis=1)
        print(f"fold {fold+1} balanced_acc (argmax): {balanced_accuracy_score(yva, fold_pred):.5f} "
              f"best_iter={booster.best_iteration}")

    argmax_pred = np.argmax(oof, axis=1)
    ba_argmax = balanced_accuracy_score(y, argmax_pred)
    print("\n=== OOF results (argmax) ===")
    print(f"balanced_acc: {ba_argmax:.5f}")
    print(classification_report(y, argmax_pred, target_names=CLASSES, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y, argmax_pred))

    print("\n=== OOF results (tuned weights) ===")
    weights, ba_tuned = tune_weights(oof, y)
    tuned_pred = apply_weights(oof, weights)
    print(f"weights={weights.round(4).tolist()}  balanced_acc: {ba_tuned:.5f}")
    print(classification_report(y, tuned_pred, target_names=CLASSES, digits=4))

    SUBMISSIONS.mkdir(exist_ok=True, parents=True)
    test_argmax = np.argmax(test_pred, axis=1)
    test_tuned = apply_weights(test_pred, weights)

    pd.DataFrame({"id": te_df["id"].values, "Irrigation_Need": np.array(CLASSES)[test_argmax]}) \
        .to_csv(SUBMISSIONS / f"sub_{tag}_argmax.csv", index=False)
    pd.DataFrame({"id": te_df["id"].values, "Irrigation_Need": np.array(CLASSES)[test_tuned]}) \
        .to_csv(SUBMISSIONS / f"sub_{tag}_tuned.csv", index=False)
    print(f"wrote sub_{tag}_argmax.csv and sub_{tag}_tuned.csv")

    elapsed = time.time() - t0
    summary = {
        "tag": tag,
        "ba_argmax": float(ba_argmax),
        "ba_tuned": float(ba_tuned),
        "weights": weights.tolist(),
        "elapsed_sec": round(elapsed, 1),
    }
    print(f"\nsummary: {json.dumps(summary, indent=2)}")
    Path(SUBMISSIONS / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    np.save(SUBMISSIONS / f"oof_{tag}.npy", oof)
    np.save(SUBMISSIONS / f"test_pred_{tag}.npy", test_pred)

    imp_df = importance.sort_values(ascending=False).to_frame("gain")
    imp_df.to_csv(SUBMISSIONS / f"importance_{tag}.csv")
    print("\nTop 20 features by gain:")
    print(imp_df.head(20))
    return summary


if __name__ == "__main__":
    run(tag="xgb_v0")
