"""Blend OOF probabilities from multiple models, then re-tune class weights.

Decoupled approach: try a handful of hand-picked blend ratios and tune class
weights once per blend. Avoids the nested optimization that made the previous
softmax/Nelder-Mead version intractable on 630k rows.

Run:
    uv run python -m src.ensemble
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.data import load_raw
from src.paths import CLASSES, SUBMISSIONS, TARGET
from src.threshold import apply_weights, tune_weights

MODELS = [
    "lgbm_v0_unweighted",
    "catboost_v0",
    "xgb_v0",
]


def load_oof(tag: str) -> tuple[np.ndarray, np.ndarray]:
    oof = np.load(SUBMISSIONS / f"oof_{tag}.npy")
    test = np.load(SUBMISSIONS / f"test_pred_{tag}.npy")
    return oof, test


def blend(oofs: list[np.ndarray], weights: list[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return sum(wi * o for wi, o in zip(w, oofs))


def main() -> None:
    tr_df, te_df = load_raw()
    target_map = {c: i for i, c in enumerate(CLASSES)}
    y = tr_df[TARGET].map(target_map).to_numpy()

    available, oofs, tests = [], [], []
    for m in MODELS:
        try:
            oof, test = load_oof(m)
            available.append(m)
            oofs.append(oof)
            tests.append(test)
            cw, score = tune_weights(oof, y)
            print(f"{m}: solo tuned balanced_acc={score:.5f}  weights={cw.round(3).tolist()}")
        except FileNotFoundError:
            print(f"{m}: skipped (no OOF saved yet)")

    if len(oofs) < 2:
        print("Need at least 2 models for ensembling.")
        return

    # Hand-picked blend ratios over (lgbm, catboost, xgb)
    candidates = [
        ("equal",       [1/3, 1/3, 1/3]),
        ("lgbm_xgb",    [0.50, 0.00, 0.50]),
        ("lgbm_heavy",  [0.50, 0.20, 0.30]),
        ("xgb_heavy",   [0.30, 0.20, 0.50]),
        ("lgbm_xgb_h",  [0.45, 0.10, 0.45]),
        ("balanced",    [0.40, 0.25, 0.35]),
        ("no_cb",       [0.55, 0.00, 0.45]),
        ("cb_light",    [0.45, 0.15, 0.40]),
    ]

    best_name, best_w, best_cw, best_score, best_blend = None, None, None, -1.0, None
    print("\n=== Blend search ===")
    for name, w in candidates:
        if len(w) != len(oofs):
            w = w[: len(oofs)]
        b = blend(oofs, w)
        cw, score = tune_weights(b, y)
        print(f"  {name:12s}  blend={[round(x,2) for x in w]}  cw={cw.round(3).tolist()}  ba={score:.5f}")
        if score > best_score:
            best_name, best_w, best_cw, best_score, best_blend = name, w, cw, score, b

    print(f"\nbest blend: {best_name}  weights={[round(x,3) for x in best_w]}  ba={best_score:.5f}")
    print(f"class_weights={best_cw.round(4).tolist()}")

    pred_oof = apply_weights(best_blend, best_cw)
    print("\n=== Blend OOF report ===")
    print(classification_report(y, pred_oof, target_names=CLASSES, digits=4))
    print("Confusion:")
    print(confusion_matrix(y, pred_oof))

    # Apply same blend ratio + class weights to test predictions
    bw = np.asarray(best_w, dtype=float)
    bw = bw / bw.sum()
    blend_test = sum(wi * t for wi, t in zip(bw, tests))
    pred_test = apply_weights(blend_test, best_cw)

    sub_path = SUBMISSIONS / "sub_blend.csv"
    pd.DataFrame({"id": te_df["id"].values, "Irrigation_Need": np.array(CLASSES)[pred_test]}) \
        .to_csv(sub_path, index=False)
    print(f"\nwrote {sub_path}")

    summary = {
        "models": available,
        "best_blend_name": best_name,
        "blend_weights": bw.tolist(),
        "class_weights": best_cw.tolist(),
        "ba_blend_tuned": float(best_score),
    }
    (SUBMISSIONS / "summary_blend.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
