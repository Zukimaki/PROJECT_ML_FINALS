"""Reusable data loading, splitting, cross-validation and modelling code."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as C

# Three compared models
MODEL_SPECS = {
    "Linear Regression": (
        LinearRegression(),
        {},
    ),
    "Random Forest Regressor": (
        RandomForestRegressor(random_state=C.RANDOM_STATE, n_jobs=1),
        {"model__n_estimators": [100, 300],
         "model__max_depth": [3, 5, None],
         "model__min_samples_leaf": [1, 3]},
    ),
    "Gradient Boosting Regressor": (
        GradientBoostingRegressor(random_state=C.RANDOM_STATE),
        {"model__n_estimators": [100, 300],
         "model__learning_rate": [0.03, 0.1],
         "model__max_depth": [2, 3]},
    ),
}


def load_clean() -> pd.DataFrame:
    return pd.read_csv(C.PROCESSED_CSV)


def usable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a known previous-year count (2015 has none and is never imputed)."""
    return df.dropna(subset=C.FEATURES + [C.TARGET]).reset_index(drop=True)


def chronological_split(df: pd.DataFrame):
    train = df[df["year"] <= C.LAST_TRAIN_YEAR].reset_index(drop=True)
    test = df[df["year"] > C.LAST_TRAIN_YEAR].reset_index(drop=True)
    return train, test


def year_folds(df: pd.DataFrame, n_folds: int = 5):
    """Expanding-window folds: validate on one year using only earlier years."""
    years = sorted(df["year"].unique())
    val_years = years[-n_folds:]
    folds = []
    for vy in val_years:
        tr = np.flatnonzero((df["year"] < vy).to_numpy())
        va = np.flatnonzero((df["year"] == vy).to_numpy())
        folds.append((tr, va))
    return folds


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def tune_models(train: pd.DataFrame):
    """Grid-search each algorithm on the training years only (same folds, same pipeline)."""
    X, y = train[C.FEATURES], train[C.TARGET]
    folds = year_folds(train)
    results, fitted = [], {}
    for name, (est, grid) in MODEL_SPECS.items():
        gs = GridSearchCV(make_pipeline(est), grid, cv=folds,
                          scoring="neg_mean_absolute_error", refit=True, n_jobs=1)
        gs.fit(X, y)
        per_fold = []
        for tr, va in folds:
            m = make_pipeline(est).set_params(**gs.best_params_).fit(X.iloc[tr], y.iloc[tr])
            per_fold.append(regression_metrics(y.iloc[va], m.predict(X.iloc[va])))
        pf = pd.DataFrame(per_fold)
        row = {"model": name, "best_params": {k.replace("model__", ""): v
                                              for k, v in gs.best_params_.items()}}
        for k in ["MAE", "RMSE", "R2", "MAPE_%"]:
            row[f"CV_{k}_mean"], row[f"CV_{k}_std"] = pf[k].mean(), pf[k].std(ddof=1)
        results.append(row)
        fitted[name] = gs.best_estimator_
    return pd.DataFrame(results), fitted


def naive_baseline(df: pd.DataFrame) -> np.ndarray:
    """Predict this year's crashes as last year's crashes (reference only, not a model)."""
    return df["prev_crashes"].to_numpy(dtype=float)
