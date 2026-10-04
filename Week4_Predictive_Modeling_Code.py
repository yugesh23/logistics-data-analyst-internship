"""
Week 4: Predictive Modeling and Optimization in Logistics Systems
Prepared for the Yuva Intern - Logistics Data Analyst Internship.

This script continues the Week 3 simulated logistics case study. It predicts
actual delivery time (days) from information available at/near dispatch time,
compares regression models, validates the best model, and performs two simple
model-based optimization scenarios.

Input:
    logistics_orders_week3.csv
Outputs:
    week4_model_results.json
    week4_test_predictions.csv
    charts_week4/*.png
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_validate, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "logistics_orders_week3.csv"
CHART_DIR = BASE / "charts_week4"
CHART_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Load and prepare data
# -----------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["order_ts"])
df = df.sort_values("order_ts").reset_index(drop=True)

# Features chosen to avoid target leakage. actual_days, is_late, OTIF, fill_rate,
# delivery_cost and qty_delivered are excluded because they are observed after
# or during fulfilment rather than known when predicting ETA.
features = [
    "distance_km", "weight_kg", "promised_days", "load_ratio",
    "hour", "weekday", "month", "zone", "ship_mode"
]
target = "actual_days"
X = df[features].copy()
y = df[target].copy()

numeric_features = [
    "distance_km", "weight_kg", "promised_days", "load_ratio",
    "hour", "weekday", "month"
]
categorical_features = ["zone", "ship_mode"]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features),
])

# Time-based split: first 80% trains, latest 20% tests. This better imitates
# deployment than a random split because future orders are not used to predict past orders.
split_idx = int(len(df) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
meta_test = df.iloc[split_idx:].copy()

# -----------------------------------------------------------------------------
# 2. Baseline and candidate models
# -----------------------------------------------------------------------------
# Baseline: predict the training-set mean delivery time for every future order.
baseline_pred = np.repeat(y_train.mean(), len(y_test))

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=250, max_depth=12, min_samples_leaf=3,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=180, learning_rate=0.05, max_depth=3,
        random_state=RANDOM_STATE
    ),
}


def metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
    }

results = {
    "Baseline Mean": metrics(y_test, baseline_pred)
}
fitted = {}

for name, estimator in models.items():
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", estimator),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results[name] = metrics(y_test, pred)
    fitted[name] = (pipe, pred)

# -----------------------------------------------------------------------------
# 3. Time-series cross-validation and hyperparameter tuning
# -----------------------------------------------------------------------------
# Evaluate candidate models on sequential folds from the training period only.
tscv = TimeSeriesSplit(n_splits=5)
cv_results = {}
for name, estimator in models.items():
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", estimator),
    ])
    scores = cross_validate(
        pipe, X_train, y_train, cv=tscv,
        scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
        n_jobs=-1,
    )
    cv_results[name] = {
        "CV_MAE_mean": float(-scores["test_mae"].mean()),
        "CV_MAE_std": float(scores["test_mae"].std()),
        "CV_R2_mean": float(scores["test_r2"].mean()),
        "CV_R2_std": float(scores["test_r2"].std()),
    }

# Tune Gradient Boosting because it captures nonlinear effects while remaining
# comparatively interpretable and stable on a medium-sized tabular dataset.
gb_pipe = Pipeline([
    ("prep", preprocessor),
    ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
])
param_grid = {
    "model__n_estimators": [100, 180],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth": [2, 3],
}
grid = GridSearchCV(
    gb_pipe, param_grid=param_grid, cv=tscv,
    scoring="neg_mean_absolute_error", n_jobs=-1, refit=True
)
grid.fit(X_train, y_train)
tuned_gb = grid.best_estimator_
tuned_gb_pred = tuned_gb.predict(X_test)
results["Tuned Gradient Boosting"] = metrics(y_test, tuned_gb_pred)

# Select the final model by lowest holdout MAE. Hyperparameter tuning is kept
# as a documented comparison; if the tuned nonlinear model does not beat the
# simpler alternative, prefer the simpler model.
candidate_final = {name: obj[0] for name, obj in fitted.items()}
candidate_final["Tuned Gradient Boosting"] = tuned_gb
best_model_name = min(
    [k for k in results if k != "Baseline Mean"],
    key=lambda k: results[k]["MAE"]
)
best_model = candidate_final[best_model_name]
best_pred = best_model.predict(X_test)

# -----------------------------------------------------------------------------
# 4. Late-delivery risk derived from predicted ETA
# -----------------------------------------------------------------------------
# Convert ETA prediction into an operational flag. A predicted ETA above the
# promised time means the order is flagged as at risk before dispatch.
predicted_late = (best_pred > X_test["promised_days"].to_numpy()).astype(int)
actual_late = (y_test.to_numpy() > X_test["promised_days"].to_numpy()).astype(int)

# Simple classification-style diagnostics without fitting a second model.
tp = int(((predicted_late == 1) & (actual_late == 1)).sum())
fp = int(((predicted_late == 1) & (actual_late == 0)).sum())
fn = int(((predicted_late == 0) & (actual_late == 1)).sum())
tn = int(((predicted_late == 0) & (actual_late == 0)).sum())
precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
risk_accuracy = (tp + tn) / len(actual_late)

# -----------------------------------------------------------------------------
# 5. Feature importance
# -----------------------------------------------------------------------------
# Permutation importance is calculated on the original feature columns, making
# the result easier to explain than one-hot-level tree importances.
perm = permutation_importance(
    best_model, X_test, y_test, n_repeats=8,
    random_state=RANDOM_STATE, scoring="neg_mean_absolute_error", n_jobs=-1
)
importance = (
    pd.DataFrame({"feature": features, "importance": perm.importances_mean})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

# -----------------------------------------------------------------------------
# 6. Model-based optimization scenarios
# -----------------------------------------------------------------------------
# These are decision-support simulations, not causal guarantees. They answer:
# "What would the model predict if an operational input changed?"

# Scenario A: Peak-capacity relief. On very busy test-period days, assume extra
# vehicles/drivers bring effective load pressure down to 1.20 times a normal day.
X_capacity = X_test.copy()
peak_mask = X_capacity["load_ratio"] > 1.5
X_capacity.loc[peak_mask, "load_ratio"] = 1.20
pred_capacity = best_model.predict(X_capacity)

base_late_pred = (best_pred > X_test["promised_days"].to_numpy()).mean()
cap_late_pred = (pred_capacity > X_test["promised_days"].to_numpy()).mean()
peak_base_late = (best_pred[peak_mask.to_numpy()] > X_test.loc[peak_mask, "promised_days"].to_numpy()).mean() if peak_mask.any() else 0
peak_cap_late = (pred_capacity[peak_mask.to_numpy()] > X_test.loc[peak_mask, "promised_days"].to_numpy()).mean() if peak_mask.any() else 0

# Scenario B: Service-policy screening. Same Day orders are allowed only when
# distance <= 20 km and predicted ETA <= 1 day; all others are flagged for
# repricing, reassignment to First Class, or a revised promise.
same_day_mask = X_test["ship_mode"].eq("Same Day")
same_day_n = int(same_day_mask.sum())
same_day_pred_late_rate = float((best_pred[same_day_mask.to_numpy()] > 1).mean()) if same_day_n else 0.0
same_day_eligible = same_day_mask & (X_test["distance_km"] <= 20) & (best_pred <= 1.0)
eligible_n = int(same_day_eligible.sum())
eligible_pred_late_rate = float((best_pred[same_day_eligible.to_numpy()] > 1).mean()) if eligible_n else 0.0

# Scenario C: Priority queue. Rank future orders by predicted lateness margin so
# dispatchers can allocate scarce expedite capacity to the highest-risk orders.
priority = meta_test[["order_ts", "zone", "ship_mode", "distance_km", "promised_days", "actual_days"]].copy()
priority["predicted_days"] = best_pred
priority["risk_margin_days"] = priority["predicted_days"] - priority["promised_days"]
priority["predicted_late"] = (priority["risk_margin_days"] > 0).astype(int)
priority = priority.sort_values("risk_margin_days", ascending=False)

# -----------------------------------------------------------------------------
# 7. Save predictions and charts
# -----------------------------------------------------------------------------
out = meta_test.copy()
out["predicted_days"] = best_pred
out["error_days"] = out["actual_days"] - out["predicted_days"]
out["predicted_late"] = predicted_late
out.to_csv(BASE / "week4_test_predictions.csv", index=False)

# Chart 1: model comparison
model_order = list(results.keys())
mae_vals = [results[m]["MAE"] for m in model_order]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.barh(model_order, mae_vals)
ax.set_xlabel("MAE (days) - lower is better")
ax.set_title("Delivery-time model comparison on the future holdout set")
for i, v in enumerate(mae_vals):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
fig.tight_layout()
fig.savefig(CHART_DIR / "01_model_comparison.png", dpi=170)
plt.close(fig)

# Chart 2: actual vs predicted
fig, ax = plt.subplots(figsize=(6.3, 5.0))
ax.scatter(y_test, best_pred, s=12, alpha=0.35)
lo = min(y_test.min(), best_pred.min())
hi = max(y_test.max(), best_pred.max())
ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
ax.set_xlabel("Actual delivery time (days)")
ax.set_ylabel("Predicted delivery time (days)")
ax.set_title("Actual vs predicted delivery time")
fig.tight_layout()
fig.savefig(CHART_DIR / "02_actual_vs_predicted.png", dpi=170)
plt.close(fig)

# Chart 3: residual distribution
resid = y_test.to_numpy() - best_pred
fig, ax = plt.subplots(figsize=(7.4, 4.1))
ax.hist(resid, bins=35, edgecolor="white")
ax.axvline(0, linestyle="--", linewidth=1)
ax.set_xlabel("Residual = actual - predicted (days)")
ax.set_ylabel("Orders")
ax.set_title("Prediction error distribution")
fig.tight_layout()
fig.savefig(CHART_DIR / "03_residuals.png", dpi=170)
plt.close(fig)

# Chart 4: permutation importance
plot_imp = importance.head(9).sort_values("importance")
fig, ax = plt.subplots(figsize=(7.4, 4.5))
ax.barh(plot_imp["feature"], plot_imp["importance"])
ax.set_xlabel("Increase in MAE when feature is shuffled")
ax.set_title("Feature importance for delivery-time prediction")
fig.tight_layout()
fig.savefig(CHART_DIR / "04_feature_importance.png", dpi=170)
plt.close(fig)

# Chart 5: capacity scenario
labels = ["Current inputs", "Peak-capacity scenario"]
vals = [base_late_pred * 100, cap_late_pred * 100]
fig, ax = plt.subplots(figsize=(6.6, 4.0))
ax.bar(labels, vals)
ax.set_ylabel("Predicted late orders (%)")
ax.set_title("Model-based capacity scenario")
for i, v in enumerate(vals):
    ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
fig.tight_layout()
fig.savefig(CHART_DIR / "05_capacity_scenario.png", dpi=170)
plt.close(fig)

# -----------------------------------------------------------------------------
# 8. Persist a concise results object for the report
# -----------------------------------------------------------------------------
report = {
    "dataset_rows": int(len(df)),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "train_period": [str(df.iloc[0]["order_ts"]), str(df.iloc[split_idx - 1]["order_ts"])],
    "test_period": [str(df.iloc[split_idx]["order_ts"]), str(df.iloc[-1]["order_ts"])],
    "features": features,
    "model_results": results,
    "cv_results": cv_results,
    "selected_model": best_model_name,
    "best_params_for_tuned_gradient_boosting": grid.best_params_,
    "tuned_gradient_boosting_cv_mae": float(-grid.best_score_),
    "risk_flag": {
        "accuracy": float(risk_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "actual_late_rate": float(actual_late.mean()),
        "predicted_late_rate": float(predicted_late.mean()),
    },
    "feature_importance": importance.to_dict(orient="records"),
    "optimization": {
        "peak_orders_in_test": int(peak_mask.sum()),
        "all_test_predicted_late_before": float(base_late_pred),
        "all_test_predicted_late_after_capacity": float(cap_late_pred),
        "peak_predicted_late_before": float(peak_base_late),
        "peak_predicted_late_after_capacity": float(peak_cap_late),
        "same_day_orders": same_day_n,
        "same_day_predicted_late_rate": same_day_pred_late_rate,
        "same_day_eligible_orders": eligible_n,
        "same_day_eligible_predicted_late_rate": eligible_pred_late_rate,
        "top_priority_examples": priority.head(10).to_dict(orient="records"),
    },
}

with open(BASE / "week4_model_results.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, default=str)

print("\nMODEL RESULTS")
print(pd.DataFrame(results).T.round(4))
print("\nSELECTED MODEL")
print(best_model_name)
print("\nTUNED GRADIENT BOOSTING PARAMETERS")
print(grid.best_params_)
print("\nTIME-SERIES CV")
print(pd.DataFrame(cv_results).T.round(4))
print("\nRISK FLAG")
print({k: round(v, 4) if isinstance(v, float) else v for k, v in report["risk_flag"].items()})
print("\nTOP FEATURE IMPORTANCE")
print(importance.head(9).round(4).to_string(index=False))
print("\nOPTIMIZATION SCENARIOS")
print(json.dumps(report["optimization"], indent=2, default=str))
