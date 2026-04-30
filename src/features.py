"""Feature engineering on the raw (pre-encoding) DataFrame.

Hypotheses motivating each feature, grounded in the EDA:
- Heat / evaporative demand drives irrigation: Temperature, Sunlight, Wind, Humidity.
- Water status: Soil_Moisture (low → High), Rainfall_mm (low → High), Previous_Irrigation_mm.
- Crop_Growth_Stage in {Flowering, Vegetative} marks the active-water-demand window
  (>30x ratio in High-class rate vs Sowing/Harvest).
- Mulching_Used=No correlates with high water loss (Yes class has 7x lower High rate).
- pH deviation from 6.5 (neutral) was weak in EDA; included for completeness, will be
  pruned if its importance is negligible.

All features are numeric so they slot into NUMERIC; categorical set is unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ACTIVE_GROWTH = {"Flowering", "Vegetative"}

NEW_NUMERIC = [
    "et_proxy",
    "vpd_proxy",
    "heat_wind",
    "rain_per_area",
    "rain_to_soil",
    "rain_log",
    "active_growth",
    "active_dryness",
    "no_mulch",
    "no_mulch_dryness",
    "prev_irr_per_area",
    "ph_deviation",
    "temp_minus_humid",
    "stress_score",
    # --- Reverse-engineered labeling-rule features (from public 0.98 notebook) ---
    "soil_lt_25",
    "wind_gt_10",
    "temp_gt_30",
    "rain_lt_300",
    "stage_low",
    "mulch_yes",
    "high_score",
    "low_score",
    "formula_score",
    "formula_pred",
]

LOW_STAGES = {"Harvest", "Sowing"}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # --- Heat / evaporative demand ---
    out["et_proxy"] = out["Temperature_C"] * out["Sunlight_Hours"] / (out["Humidity"] + 1.0)
    out["vpd_proxy"] = (100.0 - out["Humidity"]) * out["Temperature_C"] / 100.0
    out["heat_wind"] = out["Temperature_C"] * out["Wind_Speed_kmh"]
    out["temp_minus_humid"] = out["Temperature_C"] - out["Humidity"]

    # --- Water balance / deficit ---
    out["rain_per_area"] = out["Rainfall_mm"] / (out["Field_Area_hectare"] + 1.0)
    out["rain_to_soil"] = out["Rainfall_mm"] / (out["Soil_Moisture"] + 1.0)
    out["rain_log"] = np.log1p(out["Rainfall_mm"])
    out["prev_irr_per_area"] = out["Previous_Irrigation_mm"] / (out["Field_Area_hectare"] + 1.0)

    # --- Active-growth interactions ---
    active = out["Crop_Growth_Stage"].isin(ACTIVE_GROWTH).astype(np.float32)
    out["active_growth"] = active
    # Dryness magnitude only when crop is actively demanding water
    dryness = (50.0 - out["Soil_Moisture"]).clip(lower=0.0)
    out["active_dryness"] = active * dryness

    # --- Mulch interactions ---
    no_mulch = (out["Mulching_Used"] == "No").astype(np.float32)
    out["no_mulch"] = no_mulch
    out["no_mulch_dryness"] = no_mulch * dryness

    # --- Soil quality ---
    out["ph_deviation"] = (out["Soil_pH"] - 6.5).abs()

    # --- Composite stress score (z-scored sum of the strong-signal features) ---
    # Constants are train-derived means/stds (constant across train/test, so safe).
    # Higher score → drier/hotter/windier/less rain → expect more irrigation.
    out["stress_score"] = (
        + (out["Temperature_C"] - 27.0) / 8.62
        + (out["Wind_Speed_kmh"] - 10.4) / 5.69
        - (out["Soil_Moisture"] - 37.3) / 16.38
        - (out["Rainfall_mm"] - 1462.0) / 613.0
    )

    # --- Reverse-engineered labeling rules (public 0.98 notebook) ---
    # The public notebook reverse-engineered the synthetic labels as a rule:
    # high_score - low_score, bucketed → class. Adding the raw flags lets the
    # tree match the exact rule, and `formula_pred` gives a near-direct target.
    out["soil_lt_25"]  = (out["Soil_Moisture"]  < 25).astype(np.int8)
    out["wind_gt_10"]  = (out["Wind_Speed_kmh"] > 10).astype(np.int8)
    out["temp_gt_30"]  = (out["Temperature_C"]  > 30).astype(np.int8)
    out["rain_lt_300"] = (out["Rainfall_mm"]    < 300).astype(np.int8)
    out["stage_low"]   = out["Crop_Growth_Stage"].isin(LOW_STAGES).astype(np.int8)
    out["mulch_yes"]   = (out["Mulching_Used"] == "Yes").astype(np.int8)

    out["high_score"] = (
        out["soil_lt_25"]  * 2
        + out["rain_lt_300"] * 2
        + out["temp_gt_30"]  * 1
        + out["wind_gt_10"]  * 1
    ).astype(np.int8)
    out["low_score"]  = (out["stage_low"] * 2 + out["mulch_yes"] * 1).astype(np.int8)
    out["formula_score"] = (out["high_score"] - out["low_score"]).astype(np.int8)

    pred = np.zeros(len(out), dtype=np.int8)
    pred[(out["formula_score"] > 0) & (out["formula_score"] <= 3)] = 1
    pred[out["formula_score"] > 3] = 2
    out["formula_pred"] = pred

    return out
