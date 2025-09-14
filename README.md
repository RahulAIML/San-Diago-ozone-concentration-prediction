# Coastal Ozone Prediction

Predict daily maximum ground-level ozone concentration in coastal San Diego using meteorology, ocean indices (CUTI/BEUTI), and land–sea interaction features. Includes a Streamlit app for interactive predictions.

## Project Structure

- `final_cal.csv` — training dataset (provided by you)
- `src/feature_engineering.py` — reusable feature engineering (lags, rolling stats, interactions, temporal encodings, leakage guards)
- `train_ozone_model.py` — baseline training (RF + optional XGBoost with tuning, evaluation, SHAP/permutation importance)
- `train_ozone_advanced.py` — advanced training (multiple base learners, halving grid search, stacking ensemble, diagnostics)
- `app.py` — Streamlit app to load model, accept CSV/manual inputs, and predict
- `artifacts/` — metrics and plots (created on training)
- `ozone_model.pkl` / `ozone_model_advanced.pkl` — saved model bundles

## Setup

1. Create/activate a Python 3.10+ environment (or use the included `.venv`).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Baseline Training (time-based split)

1. Ensure `final_cal.csv` is in the project root.
2. Run:
   ```bash
   python train_ozone_model.py
   ```
3. Outputs:
   - `ozone_model.pkl`
   - `artifacts/metrics.json`
   - `artifacts/feature_columns.json`
   - `artifacts/permutation_importance_top10.png`
   - `artifacts/shap_beeswarm_top10.png` (if SHAP succeeds)

Split: Train = 2020–2022, Test = 2023.

### Current Baseline Metrics (2023 test)

- R²: 0.2816
- RMSE: 8.4333
- MAE: 6.5073

Source: `artifacts/metrics.json`

## Advanced Training (stacking ensemble + more features)

The advanced script trains multiple base models (RandomForest, XGBoost, Ridge, MLP) with `TimeSeriesSplit` + `HalvingGridSearchCV`, then stacks them. It retains a rich set of CUTI/BEUTI features and adds diagnostics.

Run:
```bash
python train_ozone_advanced.py
```

Outputs:
- `ozone_model_advanced.pkl`
- `artifacts/feature_columns_advanced.json`
- `artifacts/metrics_advanced.json`
- `artifacts/diagnostics/` (residual plots)
- `artifacts/perm_importance_advanced.png`

Optional A/B evaluation (experimental dataset):
```bash
python train_ozone_advanced.py --experimental path/to/experimental.csv
```
Writes `artifacts/experimental_eval.json`.

## Run the App

```bash
streamlit run app.py
```

Notes:
- The app prefers `ozone_model_advanced.pkl` if present, otherwise uses `ozone_model.pkl`.
- App shows which model is loaded (advanced/baseline).
- Upload a CSV or enter features manually. If `date` is provided, a time-series chart is rendered. A warning is shown if predicted ozone > 70 ppb.

## Implementation Notes

- Feature engineering adds:
  - Temporal encodings: `month_sin`, `month_cos`, plus standard time splits.
  - Interactions: e.g., `tmax_x_cuti`, `wspd_x_land_sea_temp_diff`.
  - Ocean indices: CUTI/BEUTI families retained broadly in advanced training.
- Leakage guard: columns containing "ozone" without explicit lags are excluded from modeling features.
- Missing values are imputed via a median imputer in the pipeline.
- Importance: permutation importance is saved; SHAP is attempted for tree models.

## Contributing / Reproducibility

- Python: 3.10+
- Install: `pip install -r requirements.txt`
- Train: `python train_ozone_model.py` or `python train_ozone_advanced.py`
- App: `streamlit run app.py`
