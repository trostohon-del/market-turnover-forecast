"""
Reusable three-phase linear forecasting pipeline inspired by `3p_linear_model.ipynb`.

The implementation focuses on offline usage with local CSV datasets while preserving
the main ideas of the 3-phase approach:
    1. Baseline statistical forecast (Holt–Winters) per group.
    2. Extensive feature engineering with seasonal and lagged signals.
    3. Gradient boosted decision trees (XGBoost) with time-series cross-validation.

The utilities here are intentionally dependency-light so the notebooks can be executed
locally without access to the original infrastructure (ClickHouse, GCS, internal utils).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import logging
import warnings

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from statsmodels.tsa.holtwinters import ExponentialSmoothing

import xgboost as xgb


logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


@dataclass
class ForecastConfig:
    time_col: str
    target_col: str
    group_cols: List[str]
    freq: str
    forecast_horizon: int
    seasonal_periods: int = 12
    min_history: int = 18
    additional_regressors: Optional[List[str]] = None
    lags: Sequence[int] = (1, 2, 3, 6, 12)
    rolling_windows: Sequence[int] = (3, 6, 12)
    seasonal: str = "add"
    trend: Optional[str] = "add"
    random_search_iterations: int = 20
    random_state: int = 46
    n_splits: int = 3
    n_estimators: int = 600
    max_lag_lookback: Optional[int] = None
    fill_future_regressors: bool = True
    future_fill_method: str = "ffill"
    allow_mean_fallback: bool = True
    target_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
    target_inverse_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def __post_init__(self) -> None:
        if not self.group_cols:
            raise ValueError("`group_cols` must contain at least one column.")
        if self.forecast_horizon <= 0:
            raise ValueError("`forecast_horizon` must be a positive integer.")
        if self.min_history < 5:
            raise ValueError("`min_history` should be at least 5 observations.")
        if isinstance(self.additional_regressors, list):
            self.additional_regressors = list(dict.fromkeys(self.additional_regressors))
        if isinstance(self.lags, Sequence) and not self.lags:
            raise ValueError("`lags` sequence cannot be empty.")
        if isinstance(self.rolling_windows, Sequence) and not self.rolling_windows:
            raise ValueError("`rolling_windows` sequence cannot be empty.")


@dataclass
class GroupForecast:
    group_key: Tuple
    predictions: pd.DataFrame
    best_params: Dict[str, float]
    best_score: float
    train_rows: int
    skipped_reason: Optional[str] = None


def _ensure_datetime(df: pd.DataFrame, column: str) -> pd.Series:
    series = pd.to_datetime(df[column])
    if series.isna().any():
        raise ValueError(f"Column `{column}` contains invalid datetime values.")
    return series


def _generate_future_index(last_timestamp: pd.Timestamp, freq: str, horizon: int) -> pd.DatetimeIndex:
    offset = to_offset(freq)
    return pd.date_range(start=last_timestamp + offset, periods=horizon, freq=freq)


def _add_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df["year"] = df[time_col].dt.year
    df["month_number"] = df[time_col].dt.month
    df["quarter"] = df[time_col].dt.quarter
    df["weekofyear"] = df[time_col].dt.isocalendar().week.astype(int)
    df["dayofyear"] = df[time_col].dt.dayofyear

    df["sin_month"] = np.sin(2 * np.pi * df["month_number"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month_number"] / 12)

    df["sin_quarter"] = np.sin(2 * np.pi * df["quarter"] / 4)
    df["cos_quarter"] = np.cos(2 * np.pi * df["quarter"] / 4)

    df["sin_year"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["cos_year"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    return df


def _add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: Sequence[int],
    rolling_windows: Sequence[int],
) -> pd.DataFrame:
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    for window in rolling_windows:
        shifted = df[target_col].shift(1)
        df[f"rolling_mean_{window}"] = shifted.rolling(window=window, min_periods=1).mean()
        df[f"rolling_std_{window}"] = shifted.rolling(window=window, min_periods=1).std()
    return df


def _holt_winters_baseline(
    train_series: pd.Series,
    future_steps: int,
    seasonal: str,
    trend: Optional[str],
    seasonal_periods: int,
    allow_mean_fallback: bool,
) -> Tuple[pd.Series, pd.Series]:
    """Returns fitted values for train indices and forecasts for future horizon."""
    if train_series.isna().any():
        train_series = train_series.dropna()

    if len(train_series) < 2:
        baseline = pd.Series(np.full(len(train_series) + future_steps, train_series.mean()), index=None)
        return baseline.iloc[: len(train_series)], baseline.iloc[len(train_series) :]

    seasonal_periods = max(2, min(seasonal_periods, len(train_series)))
    try:
        model = ExponentialSmoothing(
            train_series,
            trend=trend if seasonal_periods >= 2 else None,
            seasonal=seasonal if seasonal_periods >= 2 else None,
            seasonal_periods=seasonal_periods if seasonal_periods >= 2 else None,
        )
        fit = model.fit(optimized=True)
        fitted = fit.fittedvalues
        forecast = fit.forecast(steps=future_steps)
        return fitted, forecast
    except Exception as exc:  # pragma: no cover - defensive
        if not allow_mean_fallback:
            raise
        warnings.warn(
            f"Holt-Winters failed ({exc}). Falling back to historical mean.",
            RuntimeWarning,
        )
        mean_value = train_series.mean()
        fitted = pd.Series(np.full(len(train_series), mean_value), index=train_series.index)
        forecast = pd.Series(np.full(future_steps, mean_value))
        return fitted, forecast


def _prepare_regressors(df: pd.DataFrame, columns: Iterable[str], fill_method: str) -> pd.DataFrame:
    if not columns:
        return df
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Regressor columns not found: {missing_cols}")

    if fill_method == "ffill":
        df[list(columns)] = df[list(columns)].ffill()
    elif fill_method == "bfill":
        df[list(columns)] = df[list(columns)].bfill()
    else:
        raise ValueError(f"Unsupported fill method `{fill_method}`.")
    return df


def _prepare_feature_columns(
    df: pd.DataFrame,
    config: ForecastConfig,
    baseline_col: str,
) -> List[str]:
    feature_cols = [
        "year",
        "month_number",
        "quarter",
        "weekofyear",
        "dayofyear",
        "sin_month",
        "cos_month",
        "sin_quarter",
        "cos_quarter",
        "sin_year",
        "cos_year",
        baseline_col,
    ]

    feature_cols.extend([f"lag_{lag}" for lag in config.lags])
    for window in config.rolling_windows:
        feature_cols.append(f"rolling_mean_{window}")
        feature_cols.append(f"rolling_std_{window}")

    if config.additional_regressors:
        feature_cols.extend(config.additional_regressors)

    feature_cols = list(dict.fromkeys(feature_cols))

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing engineered feature columns: {missing}")

    return feature_cols


def _time_series_split_length(n_samples: int, desired_splits: int) -> int:
    if n_samples < 10:
        return 2
    return min(desired_splits, max(2, n_samples // 8))


def _sample_param_space(random_state: np.random.RandomState, iterations: int) -> List[Dict[str, float]]:
    if iterations <= 0:
        return []
    params: List[Dict[str, float]] = []
    for _ in range(iterations):
        params.append(
            {
                "max_depth": int(random_state.choice([3, 4, 5, 6])),
                "learning_rate": float(random_state.uniform(0.02, 0.2)),
                "subsample": float(random_state.uniform(0.6, 1.0)),
                "colsample_bytree": float(random_state.uniform(0.6, 1.0)),
                "min_child_weight": float(random_state.uniform(1.0, 6.0)),
                "gamma": float(random_state.uniform(0.0, 0.5)),
                "reg_alpha": float(random_state.uniform(0.0, 0.3)),
                "reg_lambda": float(random_state.uniform(0.5, 2.0)),
            }
        )
    return params


def _train_group_model(
    df: pd.DataFrame,
    config: ForecastConfig,
    group_key: Tuple,
) -> GroupForecast:
    time_col = config.time_col
    target_col = config.target_col

    df = df.sort_values(time_col).reset_index(drop=True)
    df[time_col] = _ensure_datetime(df, time_col)

    train_mask = df[target_col].notna()
    train_df = df[train_mask]

    if len(train_df) < config.min_history:
        reason = f"insufficient history ({len(train_df)} rows)"
        logger.debug("Group %s skipped: %s", group_key, reason)
        return GroupForecast(group_key, pd.DataFrame(), {}, np.nan, len(train_df), reason)

    # Create future rows when the dataset does not provide them explicitly.
    future_rows = df[~train_mask]
    if future_rows.empty:
        last_timestamp = train_df[time_col].max()
        future_index = _generate_future_index(last_timestamp, config.freq, config.forecast_horizon)
        future_template = pd.DataFrame({time_col: future_index})
        for col, value in zip(config.group_cols, group_key):
            future_template[col] = value
        df = pd.concat([df, future_template], ignore_index=True, sort=False)
        df[time_col] = _ensure_datetime(df, time_col)
        train_mask = df[target_col].notna()
        future_rows = df[~train_mask]

    df = _add_time_features(df, time_col)
    df = _add_lag_features(df, target_col, config.lags, config.rolling_windows)

    if config.additional_regressors:
        df = _prepare_regressors(df, config.additional_regressors, config.future_fill_method)

    # Baseline forecast (Phase 1)
    fitted_vals, forecast_vals = _holt_winters_baseline(
        train_df.set_index(time_col)[target_col],
        future_steps=len(df) - len(train_df),
        seasonal=config.seasonal,
        trend=config.trend,
        seasonal_periods=config.seasonal_periods,
        allow_mean_fallback=config.allow_mean_fallback,
    )
    baseline_col = f"{target_col}_holtwinters"
    df.loc[train_mask, baseline_col] = fitted_vals.values
    df.loc[~train_mask, baseline_col] = forecast_vals.values

    feature_cols = _prepare_feature_columns(df, config, baseline_col)

    # Fill engineered features.
    df[feature_cols] = df[feature_cols].ffill()
    df[feature_cols] = df[feature_cols].bfill()
    df[feature_cols] = df[feature_cols].fillna(0)

    train_df = df[train_mask].dropna(subset=feature_cols)
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    transform = config.target_transform or (lambda arr: arr)
    inverse_transform = config.target_inverse_transform or (lambda arr: arr)

    def _transform_target(values: pd.Series) -> np.ndarray:
        transformed = transform(values.to_numpy())
        return np.asarray(transformed, dtype=float)

    def _inverse(values: np.ndarray) -> np.ndarray:
        restored = inverse_transform(values)
        return np.asarray(restored, dtype=float)

    random_state = np.random.RandomState(config.random_state)

    n_splits = _time_series_split_length(len(X_train), config.n_splits)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    params_grid = _sample_param_space(random_state, config.random_search_iterations)
    base_params = {
        "objective": "reg:squarederror",
        "n_estimators": config.n_estimators,
        "tree_method": "hist",
        "random_state": config.random_state,
        "eval_metric": "mae",
    }

    best_score = np.inf
    best_params: Dict[str, float] = {}
    best_iteration = base_params["n_estimators"]

    if not params_grid or config.random_search_iterations <= 0:
        best_params = {}
        best_score = np.nan
        best_iteration = config.n_estimators
    elif len(X_train) < n_splits + 1:
        best_params = params_grid[0]
        best_score = np.nan
        best_iteration = config.n_estimators
    else:
        for params in params_grid:
            scores: List[float] = []
            iterations: List[int] = []
            for train_index, val_index in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
                y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

                model = xgb.XGBRegressor(**base_params, **params, early_stopping_rounds=30)
                model.fit(
                    X_tr,
                    _transform_target(y_tr),
                    eval_set=[(X_val, _transform_target(y_val))],
                    verbose=False,
                )
                pred_val = _inverse(model.predict(X_val))
                scores.append(mean_absolute_error(y_val, pred_val))
                iterations.append(getattr(model, "best_iteration", base_params["n_estimators"]))

            mean_score = float(np.mean(scores))
            mean_iter = int(np.mean(iterations))

            if mean_score < best_score:
                best_score = mean_score
                best_params = params
                best_iteration = mean_iter

    final_params = {**base_params, **best_params}
    if not isinstance(best_iteration, int) or best_iteration <= 0:
        best_iteration = config.n_estimators
    final_params["n_estimators"] = max(50, best_iteration)
    final_params.pop("random_state", None)

    final_model = xgb.XGBRegressor(**final_params, random_state=config.random_state)
    final_model.fit(X_train, _transform_target(y_train), verbose=False)

    future_df = df[~train_mask].copy()
    if future_df.empty:
        predictions = pd.DataFrame(columns=[*config.group_cols, time_col, target_col, "prediction", baseline_col])
        return GroupForecast(group_key, predictions, best_params, best_score, len(train_df))

    preds_future = _inverse(final_model.predict(future_df[feature_cols]))
    preds_future = np.clip(preds_future, 0.0, None)
    future_df["prediction"] = preds_future

    predictions = future_df[[*config.group_cols, time_col, baseline_col]].copy()
    predictions[target_col] = np.nan
    predictions["prediction"] = future_df["prediction"].values

    return GroupForecast(group_key, predictions, best_params, best_score, len(train_df))


def run_three_phase_forecast(
    df: pd.DataFrame,
    config: ForecastConfig,
) -> Tuple[pd.DataFrame, List[GroupForecast]]:
    """
    Execute the three-phase forecasting pipeline for each unique group.

    Parameters
    ----------
    df:
        Source dataframe containing historical observations and (optionally) placeholder
        rows for the forecast horizon with missing target values.
    config:
        Configuration of the forecast with time column, target column and other settings.

    Returns
    -------
    predictions_df:
        DataFrame with columns: `group_cols`, `time_col`, `prediction` and baseline values.
    group_summaries:
        List with metadata for each processed group (tuning score, params, skipped reason).
    """
    results: List[pd.DataFrame] = []
    summaries: List[GroupForecast] = []

    sort_cols = [*config.group_cols, config.time_col]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    for group_key, group_df in df.groupby(config.group_cols, sort=False):
        summary = _train_group_model(group_df, config, group_key)
        summaries.append(summary)
        if summary.skipped_reason:
            logger.warning("Group %s skipped: %s", group_key, summary.skipped_reason)
            continue
        if not summary.predictions.empty:
            results.append(summary.predictions)

    if results:
        predictions_df = pd.concat(results, ignore_index=True)
    else:
        predictions_df = pd.DataFrame(columns=[*config.group_cols, config.time_col, config.target_col, "prediction", f"{config.target_col}_holtwinters"])

    return predictions_df, summaries


__all__ = [
    "ForecastConfig",
    "GroupForecast",
    "run_three_phase_forecast",
]
