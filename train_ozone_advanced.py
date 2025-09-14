import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

from src.feature_engineering import (
    TARGET_COL,
    build_features,
    select_feature_columns,
)

# Paths
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)
MODEL_PATH = Path("ozone_model_advanced.pkl")
FEATURES_PATH = ARTIFACTS_DIR / "feature_columns_advanced.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics_advanced.json"
DIAG_DIR = ARTIFACTS_DIR / "diagnostics"
DIAG_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
# Keep nearly all correlated predictors to retain rich CUTI/BEUTI feature family
USE_CORR_FILTER = False
CORR_THRESHOLD = 0.995


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = build_features(df)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def train_test_split_by_year(df: pd.DataFrame, train_years: Tuple[int, int] = (2020, 2022), test_year: int = 2023) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["year"].between(train_years[0], train_years[1])]
    test = df[df["year"] == test_year]
    return train, test


def high_corr_filter(df: pd.DataFrame, cols: List[str], thr: float = CORR_THRESHOLD) -> List[str]:
    X = df[cols].astype(float)
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [column for column in upper.columns if any(upper[column] > thr)]
    kept = [c for c in cols if c not in drop_cols]
    return kept


def build_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            ("var", VarianceThreshold(threshold=0.0)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, feature_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return pre


def build_base_models() -> Dict[str, Tuple[BaseEstimator, Dict[str, List]]]:
    models: Dict[str, Tuple[BaseEstimator, Dict[str, List]]] = {}

    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=400, n_jobs=-1)
    rf_grid = {
        "model__n_estimators": [300, 600],
        "model__max_depth": [None, 12, 24],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }
    models["rf"] = (rf, rf_grid)

    if XGBRegressor is not None:
        xgb = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_estimators=600,
            n_jobs=-1,
            tree_method="hist",
        )
        xgb_grid = {
            "model__max_depth": [3, 6, 9],
            "model__learning_rate": [0.03, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
            "model__min_child_weight": [1, 3],
        }
        models["xgb"] = (xgb, xgb_grid)

    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-4, max_iter=500, random_state=RANDOM_STATE)
    mlp_grid = {
        "model__hidden_layer_sizes": [(128, 64), (256, 128)],
        "model__alpha": [1e-4, 1e-3],
        "model__learning_rate_init": [1e-3, 3e-3],
    }
    models["mlp"] = (mlp, mlp_grid)

    ridge = Ridge(random_state=RANDOM_STATE)
    ridge_grid = {"model__alpha": [0.1, 1.0, 10.0]}
    models["ridge"] = (ridge, ridge_grid)

    return models


def build_pipe_and_search(pre: ColumnTransformer, model: BaseEstimator, param_grid: Dict[str, List], tscv: TimeSeriesSplit) -> HalvingGridSearchCV:
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    search = HalvingGridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=tscv,
        factor=2,
        scoring="r2",
        n_jobs=-1,
        verbose=2,
        random_state=RANDOM_STATE,
    )
    return search


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    return {"r2": float(r2), "rmse": rmse, "mae": mae}


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, y_pred - y_true, s=8, alpha=0.6)
    ax.axhline(0, color="k", linestyle="--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Residual (Pred - Actual)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def permutation_importance_plot(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, feature_cols: List[str], out_path: Path):
    try:
        from sklearn.inspection import permutation_importance
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
        importances = result.importances_mean
        indices = np.argsort(importances)[::-1][:15]
        top_features = [feature_cols[i] for i in indices]
        top_values = importances[indices]
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(top_features))[::-1], top_values[::-1])
        plt.yticks(range(len(top_features))[::-1], top_features[::-1])
        plt.title("Top Permutation Importances (Advanced)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
    except Exception:
        pass


def stack_models(pre: ColumnTransformer, best_models: Dict[str, Pipeline]) -> Pipeline:
    estimators = []
    for name, pipe in best_models.items():
        estimators.append((name, pipe))
    meta = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    stack = StackingRegressor(estimators=estimators, final_estimator=meta, n_jobs=-1, passthrough=False)
    full = Pipeline(steps=[("pre", pre), ("stack", stack)])
    return full


def main(data_path: str = "final_cal.csv", experimental_path: Optional[str] = None):
    print(f"[{time.strftime('%H:%M:%S')}] Loading and building features from {data_path} ...", flush=True)
    df = load_and_prepare(data_path)
    print(f"[{time.strftime('%H:%M:%S')}] Data prepared with {len(df)} rows.", flush=True)
    feature_cols = select_feature_columns(df, TARGET_COL)
    print(f"[{time.strftime('%H:%M:%S')}] Initial numeric feature candidates: {len(feature_cols)}", flush=True)

    if USE_CORR_FILTER:
        feature_cols = high_corr_filter(df, feature_cols, thr=CORR_THRESHOLD)

    train_df, test_df = train_test_split_by_year(df)
    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COL].copy()
    print(f"[{time.strftime('%H:%M:%S')}] Train shape: {X_train.shape}, Test shape: {X_test.shape}", flush=True)

    pre = build_preprocessor(feature_cols)
    tscv = TimeSeriesSplit(n_splits=5)

    models = build_base_models()
    best_models: Dict[str, Pipeline] = {}
    cv_summary: Dict[str, float] = {}

    for name, (est, grid) in models.items():
        print(f"[{time.strftime('%H:%M:%S')}] Starting model: {name} with {len(grid)} hyperparams.", flush=True)
        search = build_pipe_and_search(pre, est, grid, tscv)
        search.fit(X_train, y_train)
        best_models[name] = search.best_estimator_
        cv_summary[f"cv_best_r2_{name}"] = float(search.best_score_)
        print(f"[{time.strftime('%H:%M:%S')}] Finished {name}: CV best R2 = {search.best_score_:.4f}", flush=True)

    # Evaluate individual best models
    test_metrics: Dict[str, Dict[str, float]] = {}
    for name, model in best_models.items():
        test_metrics[name] = evaluate(model, X_test, y_test)
        preds = model.predict(X_test)
        plot_residuals(y_test, preds, DIAG_DIR / f"residuals_{name}.png", f"Residuals - {name}")

    # Build stacking ensemble
    print(f"[{time.strftime('%H:%M:%S')}] Fitting stacking ensemble ...", flush=True)
    stack_pipe = stack_models(pre, best_models)
    stack_pipe.fit(X_train, y_train)
    stack_metrics = evaluate(stack_pipe, X_test, y_test)
    print(f"[{time.strftime('%H:%M:%S')}] Stacking done. Test R2 = {stack_metrics['r2']:.4f}", flush=True)

    # Save model + features
    joblib.dump({"model": stack_pipe, "feature_cols": feature_cols}, MODEL_PATH)
    with open(FEATURES_PATH, "w") as f:
        json.dump({"feature_cols": feature_cols}, f, indent=2)

    # Save metrics
    all_metrics = {"stack": stack_metrics, "individual": test_metrics, **cv_summary}
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Importance plot for stacking (uses permutation on full pipe)
    permutation_importance_plot(stack_pipe, X_test, y_test, feature_cols, ARTIFACTS_DIR / "perm_importance_advanced.png")

    # Optional experimental dataset comparison
    if experimental_path is not None and Path(experimental_path).exists():
        df_exp = load_and_prepare(experimental_path)
        X_exp = df_exp[feature_cols].reindex(columns=feature_cols, fill_value=np.nan)
        y_exp = df_exp[TARGET_COL] if TARGET_COL in df_exp.columns else None
        preds_exp = stack_pipe.predict(X_exp)
        exp_metrics = {}
        if y_exp is not None:
            exp_metrics = evaluate(stack_pipe, X_exp, y_exp)
        with open(ARTIFACTS_DIR / "experimental_eval.json", "w") as f:
            json.dump({"metrics": exp_metrics, "n_rows": int(len(X_exp))}, f, indent=2)

    print("Saved advanced model to", MODEL_PATH.resolve())
    print("Stack test metrics:", stack_metrics)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="final_cal.csv")
    p.add_argument("--experimental", type=str, default=None)
    args = p.parse_args()
    main(data_path=args.data, experimental_path=args.experimental)
