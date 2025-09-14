import json
import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

import shap
import matplotlib.pyplot as plt

from src.feature_engineering import (
    TARGET_COL,
    build_features,
    select_feature_columns,
)


ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)
MODEL_PATH = Path("ozone_model.pkl")
FEATURES_PATH = ARTIFACTS_DIR / "feature_columns.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# Use a reduced feature set that avoids lags/rolling/cumulative features so
# predictions on single rows (e.g., manual inputs) are meaningful.
USE_MIN_FEATURE_SET = False
MIN_BASE_FEATURES = [
    # contemporaneous meteorology
    "tmax", "tmin", "tavg", "wspd", "pres",
    # ocean indices
    "cuti", "beuti",
    # land-sea interaction
    "land_sea_temp_diff",
    # temporal encodings
    "month_sin", "month_cos", "year", "month", "dayofweek",
    # interactions
    "tmax_x_cuti", "wspd_x_land_sea_temp_diff",
]


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = build_features(df)
    # Sort by date to preserve time ordering
    df = df.sort_values("date").reset_index(drop=True)
    return df


def train_test_split_by_year(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["year"].between(2020, 2022)]
    test = df[df["year"] == 2023]
    return train, test


def build_preprocessor(feature_cols):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return preprocessor


def train_and_tune_models(X_train, y_train, preprocessor) -> Tuple[Pipeline, Dict[str, float]]:
    models = []

    rf = Pipeline(
        steps=[
            ("pre", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                    oob_score=False,
                ),
            ),
        ]
    )
    rf_param_grid = {
        "model__n_estimators": [300, 600],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
    }
    models.append(("RandomForest", rf, rf_param_grid))

    if XGBRegressor is not None:
        xgb = Pipeline(
            steps=[
                ("pre", preprocessor),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        random_state=42,
                        n_estimators=500,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                ),
            ]
        )
        xgb_param_grid = {
            "model__max_depth": [3, 6, 9],
            "model__learning_rate": [0.03, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        }
        models.append(("XGBoost", xgb, xgb_param_grid))

    best_model = None
    best_name = None
    best_score = -np.inf
    best_estimator = None

    tscv = TimeSeriesSplit(n_splits=5)

    for name, pipe, grid in models:
        search = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            cv=tscv,
            scoring="r2",
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)
        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_model = search
            best_estimator = search.best_estimator_
            best_name = name

    return best_estimator, {"cv_best_r2": float(best_score), "best_model": best_name}


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    return {"r2": float(r2), "rmse": rmse, "mae": mae}


def save_permutation_importance(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, feature_cols):
    from sklearn.inspection import permutation_importance

    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    importances = result.importances_mean
    indices = np.argsort(importances)[::-1][:10]
    top_features = [feature_cols[i] for i in indices]
    top_values = importances[indices]

    plt.figure(figsize=(8, 5))
    plt.barh(range(len(top_features))[::-1], top_values[::-1])
    plt.yticks(range(len(top_features))[::-1], top_features[::-1])
    plt.title("Top 10 Permutation Importances")
    plt.tight_layout()
    out_path = ARTIFACTS_DIR / "permutation_importance_top10.png"
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_shap_summary(model: Pipeline, X_train: pd.DataFrame, feature_cols):
    # Get the underlying model after preprocessing
    fitted_pre = model.named_steps["pre"]
    transformed = fitted_pre.transform(X_train)

    if hasattr(model.named_steps["model"], "feature_names_in_"):
        transformed_cols = [f"f{i}" for i in range(transformed.shape[1])]
    else:
        transformed_cols = [f"f{i}" for i in range(transformed.shape[1])]

    # SHAP for tree models
    base_model = model.named_steps["model"]
    try:
        explainer = shap.Explainer(base_model)
        # Use a sample for speed
        sample_idx = np.random.choice(transformed.shape[0], size=min(2000, transformed.shape[0]), replace=False)
        shap_values = explainer(transformed[sample_idx])
        plt.figure()
        shap.plots.beeswarm(shap_values, max_display=10, show=False)
        out_path = ARTIFACTS_DIR / "shap_beeswarm_top10.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
    except Exception as e:
        print(f"SHAP computation failed: {e}")


def main():
    data_path = Path("final_cal.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {data_path.resolve()}")

    df = load_and_prepare(str(data_path))

    feature_cols = select_feature_columns(df, TARGET_COL)
    if USE_MIN_FEATURE_SET:
        # Keep only features that are likely available at inference without history
        feature_cols = [c for c in feature_cols if c in MIN_BASE_FEATURES]

    train_df, test_df = train_test_split_by_year(df)

    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COL].copy()

    pre = build_preprocessor(feature_cols)

    best_model, cv_info = train_and_tune_models(X_train, y_train, pre)

    metrics = evaluate(best_model, X_test, y_test)
    all_metrics = {**cv_info, **metrics}

    # Save model and feature columns
    joblib.dump({"model": best_model, "feature_cols": feature_cols}, MODEL_PATH)
    with open(FEATURES_PATH, "w") as f:
        json.dump({"feature_cols": feature_cols}, f, indent=2)
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("Saved best model to", MODEL_PATH.resolve())
    print("Metrics:", all_metrics)

    # Save importances and SHAP
    save_permutation_importance(best_model, X_test, y_test, feature_cols)
    save_shap_summary(best_model, X_train, feature_cols)


if __name__ == "__main__":
    main()
