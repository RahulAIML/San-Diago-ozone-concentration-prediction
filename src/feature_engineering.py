import pandas as pd
import numpy as np
from typing import List, Tuple

TARGET_COL = "ozone_ppb"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )
    return df


def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        # Try common combinations
        if set(["year", "month", "day"]).issubset(df.columns):
            df["date"] = pd.to_datetime(
                df[["year", "month", "day"]].rename(columns={"day": "day"}),
                errors="coerce",
            )
        elif set(["year", "month", "dayofmonth"]).issubset(df.columns):
            df["date"] = pd.to_datetime(
                df[["year", "month", "dayofmonth"]].rename(columns={"dayofmonth": "day"}),
                errors="coerce",
            )
        else:
            # As a last resort try to parse an index or fallback to range index
            try:
                df["date"] = pd.to_datetime(df.index)
            except Exception:
                df["date"] = pd.date_range(start="2000-01-01", periods=len(df), freq="D")
    return df


def add_lag_features(
    df: pd.DataFrame,
    lag_specs: List[Tuple[str, int]] = ((TARGET_COL, 1), ("cuti", 1), ("cuti", 3)),
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("date")
    for col, lag in lag_specs:
        if col in df.columns:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    roll_specs: List[Tuple[str, int, str]] = (
        (TARGET_COL, 7, "mean"),
        ("cuti", 7, "mean"),
    ),
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("date")
    for col, window, how in roll_specs:
        if col in df.columns:
            if how == "mean":
                df[f"{col}_roll{window}_mean"] = df[col].rolling(window=window, min_periods=1).mean()
            elif how == "std":
                df[f"{col}_roll{window}_std"] = df[col].rolling(window=window, min_periods=1).std()
    return df


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "tmax" in df.columns and "cuti" in df.columns:
        df["tmax_x_cuti"] = df["tmax"] * df["cuti"]
    if "wspd" in df.columns and "land_sea_temp_diff" in df.columns:
        df["wspd_x_land_sea_temp_diff"] = df["wspd"] * df["land_sea_temp_diff"]
    return df


def add_temporal_encoding(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" not in df.columns:
        df = ensure_datetime(df)
    # Month sine/cosine
    month = df["date"].dt.month.fillna(1).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    # Keep original split helper columns if present
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year
    if "month" not in df.columns:
        df["month"] = df["date"].dt.month
    if "dayofweek" not in df.columns:
        df["dayofweek"] = df["date"].dt.dayofweek
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the full feature set required by the models.

    The function is idempotent and safe to call on new/unseen data. If prior-history
    rows are not available (for lags/rolling), those fields will be NaN and should be
    handled by downstream imputers in the ML pipeline.
    """
    df = clean_columns(df)
    df = ensure_datetime(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_interactions(df)
    df = add_temporal_encoding(df)
    return df


def select_feature_columns(df: pd.DataFrame, target_col: str = TARGET_COL) -> List[str]:
    # numeric columns except target and obvious identifiers
    # additionally, drop columns that are entirely NaN to avoid imputer warnings
    ignore = {target_col, "date"}
    cols: List[str] = []
    for c in df.columns:
        if c in ignore:
            continue
        # Guard against target leakage: drop any current-day target-derived columns
        # e.g., columns containing 'ozone' but not explicit lags such as 'lag1', 'lag7'
        cname = str(c).lower()
        if ("ozone" in cname) and ("lag" not in cname):
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        # keep only columns with at least one observed value
        if not df[c].notna().any():
            continue
        cols.append(c)
    return cols
