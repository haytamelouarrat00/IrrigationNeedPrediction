"""Render the top-15 feature-importance chart for the README."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "submissions" / "importance_lgbm_v1_features.csv"
OUT = ROOT / "assets" / "feature_importance.png"

df = pd.read_csv(CSV).sort_values("gain", ascending=False).head(15)
df = df.iloc[::-1]

fig, ax = plt.subplots(figsize=(9, 6), dpi=140)
bars = ax.barh(df["feature"], df["gain"], color="#2563eb", edgecolor="none")

top = df.iloc[-1]
top_idx = list(df["feature"]).index(top["feature"])
bars[top_idx].set_color("#dc2626")

ax.set_xscale("log")
ax.set_xlabel("LightGBM gain (log scale)")
ax.set_title("Top 15 features by gain — LightGBM v1", loc="left", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", linestyle="--", alpha=0.3)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
