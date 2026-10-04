import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("charts", exist_ok=True)
rng = np.random.default_rng(7)

# ----------------------------------------------------------------------
# 1. SIMULATED DATASET (Jan-Jun 2026, structure follows the Week 2 clean schema)
# ----------------------------------------------------------------------
N = 8000
days = pd.date_range("2026-01-01", "2026-06-30")
w = np.ones(len(days))
w[days.day >= 27] *= 1.4                                   # month-end peaks
w[days.dayofweek >= 5] *= 0.8                              # quieter weekends
w[(days >= "2026-03-14") & (days <= "2026-03-24")] *= 2.0  # festival-season surge
w /= w.sum()
day_idx = rng.choice(len(days), N, p=w)

hour_w = np.array([1, 1, 1, 1, 1, 2, 4, 7, 10, 13, 15, 14, 11, 9, 8, 8, 9, 11, 14, 13, 10, 6, 3, 2], float)
hour_w /= hour_w.sum()
hours = rng.choice(24, N, p=hour_w)
order_ts = days[day_idx] + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(rng.integers(0, 60, N), unit="m")

zone = rng.choice(["North", "South", "East", "West", "Central"], N, p=[0.2, 0.2, 0.2, 0.15, 0.25])
zone_dist = {"North": 22, "South": 26, "East": 30, "West": 38, "Central": 14}
zone_delay = {"North": 0.0, "South": 0.0, "East": 0.1, "West": 0.25, "Central": 0.0}
distance = np.array([rng.gamma(4, zone_dist[z] / 4) + 1 for z in zone]).round(1)
weight = (rng.gamma(3, 4, N) + 0.5).round(1)

ship_mode = rng.choice(["Standard Class", "Second Class", "First Class", "Same Day"], N, p=[0.55, 0.2, 0.2, 0.05])
promised = pd.Series(ship_mode).map({"Standard Class": 3, "Second Class": 3, "First Class": 2, "Same Day": 1}).values
base = pd.Series(ship_mode).map({"Standard Class": 2.0, "Second Class": 1.6, "First Class": 1.0, "Same Day": 0.3}).values
premium = pd.Series(ship_mode).map({"Standard Class": 0, "Second Class": 25, "First Class": 70, "Same Day": 140}).values

df = pd.DataFrame({"order_ts": order_ts, "zone": zone, "ship_mode": ship_mode, "distance_km": distance,
                   "weight_kg": weight, "promised_days": promised})
df["order_date"] = df["order_ts"].dt.normalize()
daily = df.groupby("order_date").size()
ratio = df["order_date"].map(daily) / daily.median()       # load relative to a normal day
df["load_ratio"] = ratio.round(2)

congestion = 0.6 * np.clip(ratio - 1.2, 0, None)
df["actual_days"] = np.clip(distance / 55 + base + pd.Series(zone).map(zone_delay).values + congestion
                            + rng.normal(0, 0.6, N), 0.2, None).round(1)
df["delivery_cost"] = np.clip(40 + 4.2 * distance + 3.0 * weight + premium + 20 * np.clip(ratio - 1.3, 0, None)
                              + rng.normal(0, 18, N), 25, None).round(2)
df["qty_ordered"] = rng.integers(1, 31, N)
short = rng.random(N) < (0.04 + 0.07 * (ratio > 1.5))
df["qty_delivered"] = np.where(short, (df["qty_ordered"] * 0.7).round(), df["qty_ordered"]).astype(int)

df["is_late"] = (df["actual_days"] > df["promised_days"]).astype(int)
df["fill_rate"] = df["qty_delivered"] / df["qty_ordered"]
df["otif"] = ((df["is_late"] == 0) & (df["qty_delivered"] >= df["qty_ordered"])).astype(int)
df["cost_per_km"] = df["delivery_cost"] / df["distance_km"]
df["hour"] = df["order_ts"].dt.hour
df["month"] = df["order_ts"].dt.month
df["month_name"] = df["order_ts"].dt.strftime("%b")
df["weekday"] = df["order_ts"].dt.dayofweek
df.drop(columns=["order_date"]).to_csv("logistics_orders_week3.csv", index=False)

# ----------------------------------------------------------------------
# 2. EDA STATISTICS
# ----------------------------------------------------------------------
S = {"rows": N, "cols": df.shape[1] - 1}
num = ["distance_km", "weight_kg", "delivery_cost", "actual_days", "cost_per_km"]
desc = {}
for c in num:
    s = df[c]
    desc[c] = {"mean": round(s.mean(), 2), "median": round(s.median(), 2), "std": round(s.std(), 2),
               "min": round(s.min(), 2), "max": round(s.max(), 2), "skew": round(s.skew(), 2)}
S["desc"] = desc
corr = df[["distance_km", "weight_kg", "delivery_cost", "actual_days", "load_ratio", "is_late", "fill_rate"]].corr().round(2)
S["corr"] = corr.to_dict()
S["late_rate"] = round(df["is_late"].mean() * 100, 1)
S["otif"] = round(df["otif"].mean() * 100, 1)
S["full_rate"] = round((df["qty_delivered"] >= df["qty_ordered"]).mean() * 100, 1)
S["late_by_zone"] = (df.groupby("zone")["is_late"].mean() * 100).round(1).sort_values(ascending=False).to_dict()
S["late_by_mode"] = (df.groupby("ship_mode")["is_late"].mean() * 100).round(1).sort_values(ascending=False).to_dict()
S["cost_by_mode"] = df.groupby("ship_mode")["delivery_cost"].mean().round(1).sort_values().to_dict()
S["cpk_by_mode"] = df.groupby("ship_mode")["cost_per_km"].median().round(2).to_dict()
S["dist_by_zone"] = df.groupby("zone")["distance_km"].mean().round(1).to_dict()
S["late_by_month"] = (df.groupby("month")["is_late"].mean() * 100).round(1).to_dict()
S["otif_by_month"] = (df.groupby("month")["otif"].mean() * 100).round(1).to_dict()
S["orders_by_month"] = df.groupby("month").size().to_dict()
byhour = df.groupby("hour").size()
S["peak_hours_share"] = round(byhour.loc[[9, 10, 11, 18, 19]].sum() / N * 100, 1)
S["late_by_hour_peak"] = round(df[df["hour"].isin([9, 10, 11, 18, 19])]["is_late"].mean() * 100, 1)
S["late_by_hour_off"] = round(df[~df["hour"].isin([9, 10, 11, 18, 19])]["is_late"].mean() * 100, 1)
S["daily_median"] = int(daily.median())
S["daily_max"] = int(daily.max())
S["daily_max_date"] = str(daily.idxmax().date())
S["late_peak_days"] = round(df[df["load_ratio"] > 1.5]["is_late"].mean() * 100, 1)
S["late_normal_days"] = round(df[df["load_ratio"] <= 1.5]["is_late"].mean() * 100, 1)
S["full_peak_days"] = round((df[df["load_ratio"] > 1.5]["fill_rate"] >= 1).mean() * 100, 1)
S["full_normal_days"] = round((df[df["load_ratio"] <= 1.5]["fill_rate"] >= 1).mean() * 100, 1)
S["peak_day_share"] = round((df["load_ratio"] > 1.5).mean() * 100, 1)
slope, intercept = np.polyfit(df["distance_km"], df["delivery_cost"], 1)
S["cost_slope"] = round(slope, 2)
S["cost_intercept"] = round(intercept, 1)
S["r_dist_cost"] = round(df["distance_km"].corr(df["delivery_cost"]), 2)
S["r2_dist_cost"] = round(df["distance_km"].corr(df["delivery_cost"]) ** 2, 2)
S["avg_cost"] = round(df["delivery_cost"].mean(), 1)
zm = df.pivot_table(index="zone", columns="ship_mode", values="is_late", aggfunc="mean") * 100
S["heat_zone_mode"] = zm.round(1).to_dict()
worst = zm.stack().idxmax()
S["worst_cell"] = [worst[0], worst[1], round(float(zm.stack().max()), 1)]
S["west_standard"] = round(float(zm.loc["West", "Standard Class"]), 1)
S["central_standard"] = round(float(zm.loc["Central", "Standard Class"]), 1)
json.dump(S, open("stats3.json", "w"), indent=2, default=str)

# ----------------------------------------------------------------------
# 3. VISUALISATIONS
# ----------------------------------------------------------------------
ORANGE, BLUE, GREY, RED = "#C8641E", "#2F6B9A", "#8A8A8A", "#B03A2E"
sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({"axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
                     "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight"})

# Chart 1: daily order volume trend
fig, ax = plt.subplots(figsize=(8, 3.6))
daily_s = daily.sort_index()
ax.plot(daily_s.index, daily_s.values, color=GREY, lw=0.8, alpha=0.7, label="Daily orders")
ax.plot(daily_s.index, daily_s.rolling(7, center=True).mean(), color=ORANGE, lw=2.2, label="7-day average")
ax.axhline(daily.median(), color=BLUE, ls="--", lw=1, label=f"Median day ({int(daily.median())})")
ax.set_title("Daily order volume, Jan to Jun 2026")
ax.set_ylabel("Orders per day"); ax.set_xlabel("")
ax.legend(frameon=False, loc="upper left")
fig.savefig("charts/01_daily_volume.png"); plt.close(fig)

# Chart 2: distributions of cost and distance
fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
sns.histplot(df["delivery_cost"], bins=40, kde=True, color=ORANGE, ax=axes[0])
axes[0].axvline(df["delivery_cost"].mean(), color=BLUE, ls="--", label=f"Mean {df['delivery_cost'].mean():.0f}")
axes[0].axvline(df["delivery_cost"].median(), color=RED, ls=":", label=f"Median {df['delivery_cost'].median():.0f}")
axes[0].set_title("Delivery cost (INR)"); axes[0].set_xlabel("INR"); axes[0].legend(frameon=False)
sns.histplot(df["distance_km"], bins=40, kde=True, color=BLUE, ax=axes[1])
axes[1].axvline(df["distance_km"].mean(), color=ORANGE, ls="--", label=f"Mean {df['distance_km'].mean():.1f}")
axes[1].axvline(df["distance_km"].median(), color=RED, ls=":", label=f"Median {df['distance_km'].median():.1f}")
axes[1].set_title("Delivery distance (km)"); axes[1].set_xlabel("km"); axes[1].set_ylabel(""); axes[1].legend(frameon=False)
fig.tight_layout(); fig.savefig("charts/02_distributions.png"); plt.close(fig)

# Chart 3: cost by shipping mode (boxplot)
order = ["Standard Class", "Second Class", "First Class", "Same Day"]
fig, ax = plt.subplots(figsize=(8, 3.6))
sns.boxplot(data=df, x="ship_mode", y="delivery_cost", order=order, color=ORANGE, fliersize=2, ax=ax)
ax.set_title("Delivery cost by shipping mode"); ax.set_xlabel(""); ax.set_ylabel("Cost (INR)")
fig.savefig("charts/03_cost_by_mode.png"); plt.close(fig)

# Chart 4: distance vs cost scatter + fit
fig, ax = plt.subplots(figsize=(8, 3.8))
samp = df.sample(2500, random_state=1)
sns.scatterplot(data=samp, x="distance_km", y="delivery_cost", hue="ship_mode", hue_order=order,
                palette=[GREY, BLUE, ORANGE, RED], s=14, alpha=0.55, linewidth=0, ax=ax)
xs = np.linspace(df["distance_km"].min(), df["distance_km"].max(), 50)
ax.plot(xs, slope * xs + intercept, color="black", lw=1.6, label=f"Fit: {slope:.1f} INR per km")
ax.set_title("Cost rises with distance; shipping mode shifts the whole line")
ax.set_xlabel("Distance (km)"); ax.set_ylabel("Cost (INR)"); ax.legend(frameon=False, fontsize=8, title="")
fig.savefig("charts/04_distance_vs_cost.png"); plt.close(fig)

# Chart 5: correlation heatmap
fig, ax = plt.subplots(figsize=(6.4, 4.8))
labels = {"distance_km": "Distance", "weight_kg": "Weight", "delivery_cost": "Cost", "actual_days": "Transit days",
          "load_ratio": "Daily load", "is_late": "Late", "fill_rate": "Fill rate"}
cm = corr.rename(index=labels, columns=labels)
sns.heatmap(cm, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, linewidths=0.5,
            cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Correlation between key variables")
fig.savefig("charts/05_correlation.png"); plt.close(fig)

# Chart 6: late rate by zone and by hour
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
lz = (df.groupby("zone")["is_late"].mean() * 100).sort_values(ascending=False)
colors = [RED if z == lz.index[0] else GREY for z in lz.index]
axes[0].bar(lz.index, lz.values, color=colors)
axes[0].axhline(S["late_rate"], color=BLUE, ls="--", lw=1, label=f"Overall {S['late_rate']}%")
axes[0].set_title("Late rate by zone"); axes[0].set_ylabel("% orders late"); axes[0].legend(frameon=False)
lh = df.groupby("hour")["is_late"].mean() * 100
axes[1].plot(lh.index, lh.values, color=ORANGE, marker="o", ms=3)
axes[1].axhline(S["late_rate"], color=BLUE, ls="--", lw=1)
axes[1].set_title("Late rate by order hour"); axes[1].set_xlabel("Hour of day (order placed)")
fig.tight_layout(); fig.savefig("charts/06_late_zone_hour.png"); plt.close(fig)

# Chart 7: zone x shipping mode late-rate heatmap
fig, ax = plt.subplots(figsize=(7, 3.8))
zm2 = zm[order]
sns.heatmap(zm2, annot=True, fmt=".0f", cmap="OrRd", linewidths=0.5, cbar_kws={"label": "% late"}, ax=ax)
ax.set_title("Late rate (%) by zone and shipping mode"); ax.set_xlabel(""); ax.set_ylabel("")
fig.savefig("charts/07_heatmap_zone_mode.png"); plt.close(fig)

# Chart 8: monthly OTIF
fig, ax = plt.subplots(figsize=(8, 3.4))
mn = df.groupby("month_name", sort=False).agg(orders=("is_late", "size"), otif=("otif", "mean"))
mn = mn.reindex(["Jan", "Feb", "Mar", "Apr", "May", "Jun"])
ax.bar(mn.index, mn["orders"], color="#E8C3A4", label="Orders")
ax.set_ylabel("Orders")
ax2 = ax.twinx()
ax2.plot(mn.index, mn["otif"] * 100, color=BLUE, marker="o", lw=2, label="OTIF %")
ax2.set_ylabel("OTIF (%)"); ax2.grid(False)
ax2.set_ylim(max(0, mn["otif"].min() * 100 - 10), 100)
ax.set_title("Monthly orders and OTIF rate")
fig.savefig("charts/08_monthly_otif.png"); plt.close(fig)

print(json.dumps(S, indent=1, default=str))
