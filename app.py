import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.feature_engineering import build_features, select_feature_columns, TARGET_COL

MODEL_PATH = Path("ozone_model.pkl")
FEATURES_JSON = Path("artifacts/feature_columns.json")
# Advanced ensemble artifacts (preferred if available)
ADV_MODEL_PATH = Path("ozone_model_advanced.pkl")
ADV_FEATURES_JSON = Path("artifacts/feature_columns_advanced.json")

st.set_page_config(page_title="Coastal Ozone Prediction App", layout="wide")
st.title("Coastal Ozone Prediction App")

@st.cache_resource
def load_model():
    # Prefer advanced ensemble if present
    if ADV_MODEL_PATH.exists():
        st.session_state["model_label"] = "advanced"
        return joblib.load(ADV_MODEL_PATH)
    # Fallback to baseline
    if MODEL_PATH.exists():
        st.session_state["model_label"] = "baseline"
        return joblib.load(MODEL_PATH)
    st.warning("No trained model found. Run `python train_ozone_model.py` or `python train_ozone_advanced.py` first.")
    return None

@st.cache_resource
def load_feature_cols():
    # Prefer advanced feature list if present
    path = ADV_FEATURES_JSON if ADV_FEATURES_JSON.exists() else FEATURES_JSON
    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)
            return data.get("feature_cols", [])
    return []

bundle = load_model()
feature_cols = load_feature_cols()
if "model_label" in st.session_state:
    st.info(f"Using {st.session_state['model_label']} model")

st.sidebar.header("Input Mode")
mode = st.sidebar.radio("Choose input type:", ["Upload CSV", "Manual Inputs"]) 

UNHEALTHY_THRESHOLD = 70.0  # ppb


def preprocess_new_data(df: pd.DataFrame) -> pd.DataFrame:
    df_proc = build_features(df)
    # Ensure model feature columns exist; add missing as NaN for the pipeline imputer
    for c in feature_cols:
        if c not in df_proc.columns:
            df_proc[c] = np.nan
    # Keep only model features
    df_proc = df_proc[feature_cols]
    return df_proc


def predict_dataframe(df_in: pd.DataFrame) -> np.ndarray:
    if bundle is None:
        return np.array([])
    model = bundle["model"]
    X = preprocess_new_data(df_in)
    preds = model.predict(X)
    return preds

if mode == "Upload CSV":
    st.subheader("Upload daily data for prediction")
    file = st.file_uploader("Upload CSV with columns like tmax, tmin, wspd, CUTI, BEUTI, land_sea_temp_diff, year, month, dayofweek, etc.", type=["csv"])

    if file is not None:
        df_new = pd.read_csv(file)
        preds = predict_dataframe(df_new)
        if preds.size > 0:
            out_df = df_new.copy()
            out_df["pred_ozone_ppb"] = preds
            # If actual available, compute diff/metrics inline
            if TARGET_COL in out_df.columns:
                out_df["abs_error"] = (out_df[TARGET_COL] - out_df["pred_ozone_ppb"]).abs()
            st.write("Predictions:")
            st.dataframe(out_df)

            # Trend plot if date available
            if "date" in out_df.columns:
                try:
                    out_df["date"] = pd.to_datetime(out_df["date"], errors="coerce")
                    plot_df = out_df.sort_values("date")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(plot_df["date"], plot_df["pred_ozone_ppb"], label="Predicted")
                    if TARGET_COL in plot_df.columns:
                        ax.plot(plot_df["date"], plot_df[TARGET_COL], label="Actual")
                    ax.axhline(UNHEALTHY_THRESHOLD, color="r", linestyle="--", label="70 ppb threshold")
                    ax.set_ylabel("Ozone (ppb)")
                    ax.legend()
                    st.pyplot(fig)
                except Exception:
                    pass

            # Warning badges
            if (out_df["pred_ozone_ppb"] > UNHEALTHY_THRESHOLD).any():
                st.warning("Unhealthy ozone predicted (> 70 ppb) for some days.")

            # Feature importance (if available)
            st.subheader("Feature Importance")
            try:
                model = bundle["model"].named_steps["model"]
                if hasattr(model, "feature_importances_") and len(feature_cols) == len(model.feature_importances_):
                    importances = pd.Series(model.feature_importances_, index=feature_cols)
                    topk = importances.sort_values(ascending=False).head(10)
                    st.bar_chart(topk)
                else:
                    st.info("Feature importances unavailable for current model. See artifacts for permutation importance/SHAP plots.")
            except Exception:
                st.info("Feature importances unavailable for current model. See artifacts for permutation importance/SHAP plots.")

elif mode == "Manual Inputs":
    st.subheader("Enter features manually for a single-day prediction")
    # Provide common inputs; others can be left blank via defaults
    col1, col2, col3 = st.columns(3)
    with col1:
        tmax = st.number_input("tmax (°C)", value=25.0)
        tmin = st.number_input("tmin (°C)", value=15.0)
        tavg = st.number_input("tavg (°C)", value=20.0)
        wspd = st.number_input("wspd (m/s)", value=3.0)
    with col2:
        pres = st.number_input("pres (hPa)", value=1015.0)
        cuti = st.number_input("CUTI", value=0.5)
        beuti = st.number_input("BEUTI", value=0.3)
        land_sea_temp_diff = st.number_input("land_sea_temp_diff (°C)", value=2.0)
    with col3:
        year = st.number_input("year", value=2023, step=1)
        month = st.slider("month", 1, 12, 7)
        day = st.slider("day", 1, 31, 15)
        dayofweek = st.slider("dayofweek (0=Mon)", 0, 6, 2)

    if st.button("Predict"):
        input_row = {
            "tmax": tmax,
            "tmin": tmin,
            "tavg": tavg,
            "wspd": wspd,
            "pres": pres,
            "cuti": cuti,
            "beuti": beuti,
            "land_sea_temp_diff": land_sea_temp_diff,
            "year": int(year),
            "month": int(month),
            "day": int(day),
            "dayofweek": int(dayofweek),
        }
        df_manual = pd.DataFrame([input_row])
        preds = predict_dataframe(df_manual)
        if preds.size > 0:
            pred_val = float(preds[0])
            st.metric("Predicted Ozone (ppb)", f"{pred_val:.1f}")
            if pred_val > UNHEALTHY_THRESHOLD:
                st.warning("Unhealthy ozone predicted (> 70 ppb).")

st.caption("Note: Lags/rolling features are computed from available context. For single-row manual input, these may be NaN and are handled by the model's imputer.")
