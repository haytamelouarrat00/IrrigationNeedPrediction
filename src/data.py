"""Data loading + consistent label encoding for train/test."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import NEW_NUMERIC, add_features
from src.paths import (
    CATEGORICAL,
    CLASSES,
    NUMERIC,
    TARGET,
    TEST_CSV,
    TRAIN_CSV,
)


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(TRAIN_CSV)
    te = pd.read_csv(TEST_CSV)
    return tr, te


def encode(
    tr: pd.DataFrame,
    te: pd.DataFrame,
    use_features: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, dict[str, int]]]:
    """Label-encode categoricals using categories present in train (test is a strict subset, verified in EDA).

    Parameters
    ----------
    use_features : bool
        If True, augment numeric features via src.features.add_features before encoding.

    Returns
    -------
    X_tr, X_te : feature frames in the same column order
    y          : integer-encoded target with CLASSES order (Low=0, Medium=1, High=2)
    cat_maps   : per-column category -> int mapping for reproducibility
    """
    if use_features:
        tr = add_features(tr)
        te = add_features(te)
        numeric_cols = NUMERIC + NEW_NUMERIC
    else:
        numeric_cols = NUMERIC

    X_tr = tr[numeric_cols + CATEGORICAL].copy()
    X_te = te[numeric_cols + CATEGORICAL].copy()

    cat_maps: dict[str, dict[str, int]] = {}
    for c in CATEGORICAL:
        cats = sorted(X_tr[c].astype(str).unique())
        m = {v: i for i, v in enumerate(cats)}
        cat_maps[c] = m
        X_tr[c] = X_tr[c].map(m).astype("int32")
        X_te[c] = X_te[c].map(m).astype("int32")

    target_map = {c: i for i, c in enumerate(CLASSES)}
    y = tr[TARGET].map(target_map).to_numpy()
    return X_tr, X_te, y, cat_maps
