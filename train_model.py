"""
train_model.py
--------------
Trains a House Price Prediction model on the Housing Prices Dataset
(mannatpruthi/house-prediction on Kaggle).

Dataset columns: area, bedrooms, bathrooms, stories, mainroad, guestroom,
                 basement, hotwaterheating, airconditioning, parking,
                 prefarea, furnishingstatus, price

Saves:
  model/house_price_model.pkl     – serialised sklearn Pipeline
  model/feature_columns.json      – ordered feature list used by the model
  model/metrics.json              – RMSE, MAE, R2 on validation split
  report_images/correlation_heatmap.png
  report_images/actual_vs_predicted.png
  report_images/feature_importance.png
  report_images/saleprice_distribution.png
  report_images/grlivarea_vs_saleprice.png

Usage:
  python train_model.py
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH     = os.path.join(BASE_DIR, "data", "train.csv")
MODEL_DIR     = os.path.join(BASE_DIR, "model")
IMAGES_DIR    = os.path.join(BASE_DIR, "report_images")
MODEL_PATH    = os.path.join(MODEL_DIR, "house_price_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.json")
METRICS_PATH  = os.path.join(MODEL_DIR, "metrics.json")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
print("Loading data ...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

TARGET = "price"

# ── Preprocessing ─────────────────────────────────────────────────────────────
# Encode binary yes/no columns
BINARY_COLS = ["mainroad", "guestroom", "basement", "hotwaterheating",
               "airconditioning", "prefarea"]
for col in BINARY_COLS:
    df[col] = df[col].map({"yes": 1, "no": 0})

# Encode furnishingstatus: furnished=2, semi-furnished=1, unfurnished=0
df["furnishingstatus"] = df["furnishingstatus"].map(
    {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
)

# Impute any remaining missing values with median
for col in df.columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

print(f"  Missing values after preprocessing: {df.isnull().sum().sum()}")

# ── Features ──────────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "area", "bedrooms", "bathrooms", "stories",
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "parking", "prefarea", "furnishingstatus"
]

X = df[NUMERIC_FEATURES]
y = df[TARGET]
feature_columns = list(X.columns)

# ── Train / Validation Split ───────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Train size: {len(X_train)}  |  Val size: {len(X_val)}")

# ── Model Selection ───────────────────────────────────────────────────────────
def build_pipeline(estimator):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  estimator)
    ])

candidates = {
    "LinearRegression": build_pipeline(LinearRegression()),
    "Ridge":            build_pipeline(Ridge(alpha=10)),
    "Lasso":            build_pipeline(Lasso(alpha=50, max_iter=10000)),
}

best_name, best_pipe, best_rmse = None, None, float("inf")

print("\nModel comparison:")
results = {}
for name, pipe in candidates.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    rmse  = np.sqrt(mean_squared_error(y_val, preds))
    mae   = mean_absolute_error(y_val, preds)
    r2    = r2_score(y_val, preds)
    results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    print(f"  {name:20s}  RMSE={rmse:,.0f}  MAE={mae:,.0f}  R2={r2:.4f}")
    if rmse < best_rmse:
        best_rmse, best_name, best_pipe = rmse, name, pipe

print(f"\nBest model: {best_name}  (RMSE={best_rmse:,.0f})")

# ── Save Model Artefacts ──────────────────────────────────────────────────────
joblib.dump(best_pipe, MODEL_PATH)
print(f"Model saved -> {MODEL_PATH}")

with open(FEATURES_PATH, "w") as f:
    json.dump(feature_columns, f, indent=2)
print(f"Feature columns saved -> {FEATURES_PATH}")

metrics = {
    "best_model": best_name,
    "RMSE":  round(results[best_name]["RMSE"], 2),
    "MAE":   round(results[best_name]["MAE"],  2),
    "R2":    round(results[best_name]["R2"],   4),
    "all_models": {k: {m: round(v, 2) for m, v in mv.items()}
                   for k, mv in results.items()}
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved -> {METRICS_PATH}")

# ── Report Charts ──────────────────────────────────────────────────────────────
best_val_preds = best_pipe.predict(X_val)
print("\nGenerating charts ...")

# 1. Correlation Heatmap
fig, ax = plt.subplots(figsize=(12, 9))
corr = df[NUMERIC_FEATURES + [TARGET]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax, linewidths=0.5, annot_kws={"size": 8})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "correlation_heatmap.png"), dpi=120)
plt.close()

# 2. Actual vs Predicted
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_val, best_val_preds, alpha=0.5, color="#3b82d4",
           edgecolors="white", linewidths=0.3)
mn = min(y_val.min(), best_val_preds.min())
mx = max(y_val.max(), best_val_preds.max())
ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual Price", fontsize=11)
ax.set_ylabel("Predicted Price", fontsize=11)
ax.set_title(f"Actual vs Predicted - {best_name}", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "actual_vs_predicted.png"), dpi=120)
plt.close()

# 3. Feature Importance (coefficients)
coefs    = best_pipe.named_steps["model"].coef_
feat_imp = pd.Series(np.abs(coefs), index=feature_columns).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 6))
feat_imp.plot(kind="barh", ax=ax, color="#7c5cd8", edgecolor="white")
ax.set_title("Feature Importance (|Coefficient|)", fontsize=13, fontweight="bold")
ax.set_xlabel("|Coefficient|", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "feature_importance.png"), dpi=120)
plt.close()

# 4. Price Distribution
fig, ax = plt.subplots(figsize=(9, 5))
df[TARGET].hist(bins=40, ax=ax, color="#3b82d4", edgecolor="white")
ax.set_title("Price Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Price", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "saleprice_distribution.png"), dpi=120)
plt.close()

# 5. Area vs Price scatter
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(df["area"], df[TARGET], alpha=0.4, color="#3b82d4",
           edgecolors="white", linewidths=0.3)
ax.set_xlabel("Area (sq ft)", fontsize=11)
ax.set_ylabel("Price", fontsize=11)
ax.set_title("Area vs Price", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "grlivarea_vs_saleprice.png"), dpi=120)
plt.close()

print("All charts saved to report_images/")
print("\nTraining complete!")
print(f"   Best Model : {best_name}")
print(f"   RMSE       : {metrics['RMSE']:,.2f}")
print(f"   MAE        : {metrics['MAE']:,.2f}")
print(f"   R2         : {metrics['R2']}")
