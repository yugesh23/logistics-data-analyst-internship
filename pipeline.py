import json
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 5000

# ---------------------------------------------------------------
# 1. SIMULATED RAW DATA (structure modelled on public shipping datasets)
# ---------------------------------------------------------------
order_ts = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 180 * 24 * 60, N), unit="m")
distance = np.round(rng.gamma(4, 6, N) + 1, 1)
weight = np.round(rng.gamma(3, 4, N) + 0.5, 1)
promised_days = rng.choice([1, 2, 3, 4], N, p=[0.2, 0.4, 0.3, 0.1])
actual_days = np.clip(np.round(distance / 25 + rng.normal(1, 0.8, N)), 0, None)
cost = np.round(30 + distance * 4.5 + weight * 3 + rng.normal(0, 20, N), 2)

df = pd.DataFrame({
    "order_id": [f"ORD{100000 + i}" for i in range(N)],
    "order_ts": order_ts,
    "ship_mode": rng.choice(["Standard Class", "Second Class", "First Class", "Same Day"], N, p=[0.6, 0.2, 0.15, 0.05]),
    "zone": rng.choice(["North", "South", "East", "West", "Central"], N),
    "distance_km": distance,
    "weight_kg": weight,
    "promised_days": promised_days,
    "actual_days": actual_days,
    "delivery_cost": cost,
    "qty_ordered": rng.integers(1, 30, N),
})
df["qty_delivered"] = df["qty_ordered"]

# ---- inject realistic data-quality problems ----
# duplicates
dups = df.sample(120, random_state=1)
df = pd.concat([df, dups], ignore_index=True)
# inconsistent categories
df["ship_mode"] = df["ship_mode"].astype(object)
idx = df.sample(400, random_state=2).index
df.loc[idx[:150], "ship_mode"] = df.loc[idx[:150], "ship_mode"].str.upper()
df.loc[idx[150:300], "ship_mode"] = df.loc[idx[150:300], "ship_mode"].str.lower() + " "
df.loc[idx[300:], "ship_mode"] = "Std Class"
df["zone"] = df["zone"].astype(object)
df.loc[df.sample(200, random_state=3).index, "zone"] = "north "
# missing values
for col, n in [("distance_km", 300), ("weight_kg", 250), ("delivery_cost", 180), ("zone", 150), ("actual_days", 100)]:
    df.loc[df.sample(n, random_state=hash(col) % 1000).index, col] = np.nan
# impossible values
df.loc[df.sample(25, random_state=5).index, "weight_kg"] = -3
df.loc[df.sample(20, random_state=6).index, "actual_days"] = -1
# extreme outliers
df.loc[df.sample(40, random_state=7).index, "distance_km"] = rng.uniform(900, 2500, 40)
df.loc[df.sample(30, random_state=8).index, "delivery_cost"] = rng.uniform(8000, 20000, 30)
# text-typed numbers
df["distance_km"] = df["distance_km"].astype(object)
df.loc[df.sample(60, random_state=9).index, "distance_km"] = "unknown"
# partial delivery
df.loc[df.sample(150, random_state=10).index, "qty_delivered"] = (df["qty_ordered"] * 0.7).round()

df.to_csv("raw_logistics_orders.csv", index=False)

# ---------------------------------------------------------------
# 2. CLEANING PIPELINE
# ---------------------------------------------------------------
stats = {}
raw = pd.read_csv("raw_logistics_orders.csv", parse_dates=["order_ts"])
stats["raw_rows"] = len(raw)
stats["raw_cols"] = raw.shape[1]
stats["missing_before"] = {k: int(v) for k, v in raw.isna().sum().items() if v > 0}
stats["dtype_before_distance"] = str(raw["distance_km"].dtype)

df = raw.copy()

# (a) duplicates
stats["duplicates"] = int(df.duplicated(subset="order_id").sum())
df = df.drop_duplicates(subset="order_id", keep="first")

# (b) data types: coerce text to numeric
before_na = df["distance_km"].isna().sum()
df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
stats["coerced_text_to_nan"] = int(df["distance_km"].isna().sum() - before_na)

# (c) categorical standardisation
stats["ship_mode_before"] = sorted(df["ship_mode"].dropna().unique().tolist())
df["ship_mode"] = (df["ship_mode"].str.strip().str.title()
                   .replace({"Std Class": "Standard Class"}))
stats["ship_mode_after"] = sorted(df["ship_mode"].dropna().unique().tolist())
stats["zone_before"] = sorted(df["zone"].dropna().unique().tolist())
df["zone"] = df["zone"].str.strip().str.title()
stats["zone_after"] = sorted(df["zone"].dropna().unique().tolist())

# (d) impossible values -> NaN
stats["neg_weight"] = int((df["weight_kg"] <= 0).sum())
stats["neg_days"] = int((df["actual_days"] < 0).sum())
df.loc[df["weight_kg"] <= 0, "weight_kg"] = np.nan
df.loc[df["actual_days"] < 0, "actual_days"] = np.nan

# (e) outliers: IQR method, flagged then capped (winsorised)
def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

outliers = {}
for col in ["distance_km", "weight_kg", "delivery_cost"]:
    lo, hi = iqr_bounds(df[col].dropna())
    mask = (df[col] < lo) | (df[col] > hi)
    outliers[col] = {"count": int(mask.sum()), "lower": round(float(lo), 2), "upper": round(float(hi), 2),
                     "max_before": round(float(df[col].max()), 2)}
    df[f"{col}_outlier"] = mask.astype(int)
    df[col] = df[col].clip(lower=max(lo, 0), upper=hi)
    outliers[col]["max_after"] = round(float(df[col].max()), 2)
stats["outliers"] = outliers

# (f) missing values
stats["missing_after_coerce"] = {k: int(v) for k, v in df.isna().sum().items() if v > 0}
df["zone"] = df["zone"].fillna("Unknown")                              # categorical -> label
df["actual_days"] = df["actual_days"].fillna(df["actual_days"].median())
for col in ["distance_km", "weight_kg", "delivery_cost"]:
    df[col] = df[col].fillna(df.groupby("ship_mode")[col].transform("median"))
    df[col] = df[col].fillna(df[col].median())
stats["missing_final"] = int(df.isna().sum().sum())

# (g) feature engineering
df["is_late"] = (df["actual_days"] > df["promised_days"]).astype(int)
df["fill_rate"] = (df["qty_delivered"] / df["qty_ordered"]).round(3)
df["cost_per_km"] = (df["delivery_cost"] / df["distance_km"]).round(2)
df["order_month"] = df["order_ts"].dt.to_period("M").astype(str)

# (h) normalisation
num = ["distance_km", "weight_kg", "delivery_cost"]
desc_before = df[num].describe().loc[["mean", "std", "min", "max"]].round(2)
for c in num:
    df[f"{c}_minmax"] = ((df[c] - df[c].min()) / (df[c].max() - df[c].min())).round(4)
    df[f"{c}_z"] = ((df[c] - df[c].mean()) / df[c].std()).round(4)
stats["desc_before_norm"] = desc_before.to_dict()
stats["minmax_range"] = {c: [float(df[f"{c}_minmax"].min()), float(df[f"{c}_minmax"].max())] for c in num}
stats["z_mean_std"] = {c: [round(float(df[f"{c}_z"].mean()), 3), round(float(df[f"{c}_z"].std()), 3)] for c in num}

# (i) final
stats["clean_rows"] = len(df)
stats["clean_cols"] = df.shape[1]
stats["late_rate"] = round(float(df["is_late"].mean() * 100), 1)
stats["avg_fill_rate"] = round(float(df["fill_rate"].mean() * 100), 1)

# KPI sensitivity: mean cost before vs after cleaning
raw_cost = pd.to_numeric(raw["delivery_cost"], errors="coerce")
stats["mean_cost_raw"] = round(float(raw_cost.mean()), 1)
stats["mean_cost_clean"] = round(float(df["delivery_cost"].mean()), 1)
raw_dist = pd.to_numeric(raw["distance_km"], errors="coerce")
stats["mean_dist_raw"] = round(float(raw_dist.mean()), 1)
stats["mean_dist_clean"] = round(float(df["distance_km"].mean()), 1)

df.to_csv("clean_logistics_orders.csv", index=False)
json.dump(stats, open("stats.json", "w"), indent=2, default=str)
print(json.dumps(stats, indent=2, default=str))
