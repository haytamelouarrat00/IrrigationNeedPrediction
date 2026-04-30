"""LightGBM 5-fold with in-fold multiclass target encoding.

Adds two sources of signal on top of baseline.run(use_features=True):
- Bigram + trigram interactions of top categoricals (Crop_Growth_Stage,
  Mulching_Used, Crop_Type), factorized using combined train+test vocab.
- In-fold sklearn TargetEncoder(target_type='multiclass'): for each encoded
  feature it appends 3 columns with P(class_k | feature_value), cross-fitted
  inside the train fold to avoid target leakage.

Run:
    uv run python -m src.baseline_te
"""
from __future__ import annotations

import json
import time
from itertools import combinations
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
from sklearn.preprocessing import TargetEncoder

from src.data import load_raw
from src.features import NEW_NUMERIC, add_features
from src.paths import CATEGORICAL, CLASSES, NUMERIC, SUBMISSIONS, TARGET
from src.threshold import apply_weights, tune_weights

N_FOLDS = 5
SEED = 42
TE_CV = 3  # internal CV in TargetEncoder; 3 instead of 5 for speed

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

NGRAM_CATS = ["Crop_Growth_Stage", "Mulching_Used", "Crop_Type"]


def make_ngrams(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Add bigrams + trigram of NGRAM_CATS, factorized over combined train+test vocab."""
    n_tr = len(tr)
    ngram_cols: list[str] = []

    for c1, c2 in combinations(NGRAM_CATS, 2):
        name = f"bg_{c1}__{c2}"
        joined = pd.concat([
            tr[c1].astype(str) + "_" + tr[c2].astype(str),
            te[c1].astype(str) + "_" + te[c2].astype(str),
        ], ignore_index=True)
        codes, _ = pd.factorize(joined)
        tr[name] = codes[:n_tr].astype(np.int32)
        te[name] = codes[n_tr:].astype(np.int32)
        ngram_cols.append(name)

    name = "tg_" + "__".join(NGRAM_CATS)
    joined = pd.concat([
        tr[NGRAM_CATS[0]].astype(str) + "_" + tr[NGRAM_CATS[1]].astype(str) + "_" + tr[NGRAM_CATS[2]].astype(str),
        te[NGRAM_CATS[0]].astype(str) + "_" + te[NGRAM_CATS[1]].astype(str) + "_" + te[NGRAM_CATS[2]].astype(str),
    ], ignore_index=True)
    codes, _ = pd.factorize(joined)
    tr[name] = codes[:n_tr].astype(np.int32)
    te[name] = codes[n_tr:].astype(np.int32)
    ngram_cols.append(name)
    return tr, te, ngram_cols


def encode_categoricals(tr: pd.DataFrame, te: pd.DataFrame) -> None:
    """Label-encode raw categorical string columns in-place using train vocab."""
    for c in CATEGORICAL:
        cats = sorted(tr[c].astype(str).unique())
        m = {v: i for i, v in enumerate(cats)}
        tr[c] = tr[c].map(m).astype("int32")
        te[c] = te[c].map(m).astype("int32")


def run(tag: str = "lgbm_v2_te") -> dict:
    t0 = time.time()
    print(f"\n=== LGBM TE run: tag={tag} ===")
    tr_df, te_df = load_raw()
    tr_df = add_features(tr_df)
    te_df = add_features(te_df)

    tr_df, te_df, ngram_cols = make_ngrams(tr_df, te_df)
    print(f"ngram cols ({len(ngram_cols)}): {ngram_cols}")

    target_map = {c: i for i, c in enumerate(CLASSES)}
    y = tr_df[TARGET].map(target_map).to_numpy()

    encode_categoricals(tr_df, te_df)

    base_num = NUMERIC + NEW_NUMERIC + ngram_cols
    base_features = base_num + CATEGORICAL
    X = tr_df[base_features].copy()
    X_test = te_df[base_features].copy()
    print(f"base features: {len(base_features)}")

    # Columns to target-encode: low-cardinality discretes
    TE_COLS = CATEGORICAL + ngram_cols + ["formula_score", "formula_pred"]
    print(f"TE cols ({len(TE_COLS)}): {TE_COLS}")

    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_pred = np.zeros((len(X_test), 3), dtype=np.float32)
    importance_gain: pd.Series | None = None

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\n-- fold {fold+1}/{N_FOLDS} --")
        Xtr = X.iloc[tr_idx].copy()
        Xva = X.iloc[va_idx].copy()
        ytr = y[tr_idx]
        yva = y[va_idx]
        Xte = X_test.copy()

        t_te = time.time()
        encoder = TargetEncoder(
            target_type="multiclass",
            smooth="auto",
            cv=TE_CV,
            random_state=SEED,
        )
        te_tr = encoder.fit_transform(Xtr[TE_COLS], ytr)
        te_va = encoder.transform(Xva[TE_COLS])
        te_te = encoder.transform(Xte[TE_COLS])
        n_out = te_tr.shape[1]
        te_names = [f"TE_{i}" for i in range(n_out)]
        Xtr[te_names] = te_tr
        Xva[te_names] = te_va
        Xte[te_names] = te_te
        print(f"  TE done in {time.time()-t_te:.1f}s; te_cols={n_out}; total cols={Xtr.shape[1]}")

        dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=CATEGORICAL)
        dva = lgb.Dataset(Xva, label=yva, categorical_feature=CATEGORICAL, reference=dtr)
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
        test_pred += booster.predict(Xte, num_iteration=booster.best_iteration) / N_FOLDS

        if importance_gain is None:
            importance_gain = pd.Series(0.0, index=Xtr.columns)
        importance_gain += booster.feature_importance(importance_type="gain") / N_FOLDS

        fold_pred = np.argmax(oof[va_idx], axis=1)
        print(f"fold {fold+1} balanced_acc (argmax): {balanced_accuracy_score(yva, fold_pred):.5f} "
              f"best_iter={booster.best_iteration}")

    argmax_pred = np.argmax(oof, axis=1)
    ba_argmax = balanced_accuracy_score(y, argmax_pred)
    print("\n=== OOF results (argmax) ===")
    print(f"balanced_acc: {ba_argmax:.5f}")
    print(classification_report(y, argmax_pred, target_names=CLASSES, digits=4))
    print("Confusion:")
    print(confusion_matrix(y, argmax_pred))

    print("\n=== OOF results (tuned weights) ===")
    weights, ba_tuned = tune_weights(oof, y)
    tuned_pred = apply_weights(oof, weights)
    print(f"weights={weights.round(4).tolist()}  balanced_acc: {ba_tuned:.5f}")
    print(classification_report(y, tuned_pred, target_names=CLASSES, digits=4))
    print("Confusion:")
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
        "te_cols": TE_COLS,
        "elapsed_sec": round(elapsed, 1),
    }
    print(f"\nsummary: {json.dumps(summary, indent=2)}")
    Path(SUBMISSIONS / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    np.save(SUBMISSIONS / f"oof_{tag}.npy", oof)
    np.save(SUBMISSIONS / f"test_pred_{tag}.npy", test_pred)

    imp_df = importance_gain.sort_values(ascending=False).to_frame("gain")
    imp_df.to_csv(SUBMISSIONS / f"importance_{tag}.csv")
    print("\nTop 25 features by gain:")
    print(imp_df.head(25))
    return summary


if __name__ == "__main__":
    run(tag="lgbm_v2_te")
