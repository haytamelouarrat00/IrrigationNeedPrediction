"""CatBoost 5-fold runner for Predicting Irrigation Need.

Run from repo root:
    uv run python -m src.catboost_run
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

from src.data import load_raw
from src.paths import CATEGORICAL, CLASSES, NUMERIC, SUBMISSIONS, TARGET
from src.threshold import apply_weights, tune_weights

N_FOLDS = 5
SEED = 42

CB_PARAMS = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "iterations": 3000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "random_seed": SEED,
    "early_stopping_rounds": 100,
    "task_type": "GPU",
    "devices": "0",
    "verbose": 200,
    "allow_writing_files": False,
}


def run(tag: str = "catboost_v0") -> dict:
    t0 = time.time()
    print(f"\n=== CatBoost run: tag={tag} ===")

    tr_df, te_df = load_raw()

    # CatBoost prefers native categorical handling — pass strings directly.
    cols = NUMERIC + CATEGORICAL
    X = tr_df[cols].copy()
    X_test = te_df[cols].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype(str)
        X_test[c] = X_test[c].astype(str)

    target_map = {c: i for i, c in enumerate(CLASSES)}
    y = tr_df[TARGET].map(target_map).to_numpy()

    cat_idx = [cols.index(c) for c in CATEGORICAL]

    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_pred = np.zeros((len(X_test), 3), dtype=np.float32)
    importance = np.zeros(len(cols), dtype=np.float64)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\n-- fold {fold+1}/{N_FOLDS} --")
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        train_pool = Pool(Xtr, ytr, cat_features=cat_idx)
        val_pool   = Pool(Xva, yva, cat_features=cat_idx)
        test_pool  = Pool(X_test, cat_features=cat_idx)

        model = CatBoostClassifier(**CB_PARAMS)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        oof[va_idx] = model.predict_proba(val_pool)
        test_pred += model.predict_proba(test_pool) / N_FOLDS
        importance += model.get_feature_importance()

        fold_pred = np.argmax(oof[va_idx], axis=1)
        print(f"fold {fold+1} balanced_acc (argmax): {balanced_accuracy_score(yva, fold_pred):.5f} "
              f"best_iter={model.tree_count_}")

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
    print("Confusion matrix:")
    print(confusion_matrix(y, tuned_pred))

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

    imp_df = pd.DataFrame({"feature": cols, "importance": importance / N_FOLDS}) \
        .sort_values("importance", ascending=False)
    imp_df.to_csv(SUBMISSIONS / f"importance_{tag}.csv", index=False)
    print("\nTop features by importance:")
    print(imp_df.to_string(index=False))
    return summary


if __name__ == "__main__":
    run(tag="catboost_v0")
