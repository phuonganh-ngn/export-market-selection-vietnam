from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, average_precision_score
from sklearn.base import clone
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

from common import CFG
from temporal_validation import NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURE_GROUPS


MODEL_NAMES = ["RandomForest", "XGBoost", "LightGBM"]


def inverse_log1p_predictions(pred_log: np.ndarray) -> np.ndarray:
    """Stable, nonnegative inverse transform for predictions trained on log1p USD."""
    return np.maximum(0.0, np.expm1(np.clip(np.asarray(pred_log, dtype=float), 0.0, 35.0)))


def validation_mae_scale(actual_usd: np.ndarray, predicted_log: np.ndarray) -> float:
    """Choose a multiplicative USD calibration minimizing validation MAE.

    For nonnegative base predictions x, minimizing sum |y - a*x| over a >= 0
    is solved by the x-weighted median of y/x. This avoids mean Duan smearing,
    which is unstable under strongly heteroskedastic export-value residuals.
    """
    actual = np.asarray(actual_usd, dtype=float)
    base = inverse_log1p_predictions(predicted_log)
    mask = np.isfinite(actual) & np.isfinite(base) & (base > 1e-12)
    if not mask.any() or float(base[mask].sum()) <= 0:
        return 1.0
    ratios = np.maximum(0.0, actual[mask] / base[mask])
    weights = base[mask]
    order = np.argsort(ratios, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    return float(ratios[order[np.searchsorted(cumulative, cumulative[-1] / 2.0, side="left")]])


def feature_columns(drop_groups: list[str] | None = None) -> tuple[list[str], list[str]]:
    dropped = set()
    for group in drop_groups or []:
        dropped.update(FEATURE_GROUPS.get(group, []))
    nums = [c for c in NUMERIC_FEATURES if c not in dropped]
    cats = list(CATEGORICAL_FEATURES)
    return nums, cats


def make_preprocessor(train: pd.DataFrame, drop_groups: list[str] | None = None, scale: bool = True):
    nums, cats = feature_columns(drop_groups)
    nums = [c for c in nums if c in train.columns]
    cats = [c for c in cats if c in train.columns]
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                        ("scale", StandardScaler(with_mean=False) if scale else "passthrough")])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                             ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32))])
    return ColumnTransformer([("numeric", numeric, nums), ("category", categorical, cats)], remainder="drop", sparse_threshold=0.3)


def model_configs(kind: str, classification: bool = False, n_trials: int | None = None) -> list[dict]:
    n = min(n_trials or CFG["study"]["tuning_trials"], 20)
    threads = CFG["compute"]["model_threads"]
    if kind == "RandomForest":
        trees = int(CFG["compute"]["random_forest_trees"])
        grid = ([{"n_estimators": trees, "max_depth": 12, "min_samples_leaf": 5},
                 {"n_estimators": trees, "max_depth": 18, "min_samples_leaf": 12},
                 {"n_estimators": trees, "max_depth": None, "min_samples_leaf": 20}])
        if classification:
            grid = [dict(c, class_weight="balanced_subsample") for c in grid]
        return grid[:n]
    if kind == "XGBoost":
        rounds = int(CFG["compute"]["boosting_rounds"])
        if classification:
            return [{"n_estimators": rounds, "max_depth": d, "learning_rate": 0.05, "subsample": 0.85,
                     "colsample_bytree": 0.85, "min_child_weight": 5, "reg_lambda": 5,
                     "scale_pos_weight": 1.0, "n_jobs": threads, "tree_method": "hist", "eval_metric": "aucpr"} for d in [4, 6, 8]][:n]
        return [{"n_estimators": rounds, "max_depth": d, "learning_rate": 0.05, "subsample": 0.85,
                 "colsample_bytree": 0.85, "min_child_weight": 5, "reg_lambda": 5, "n_jobs": threads,
                 "tree_method": "hist", "objective": "reg:squarederror"} for d in [4, 6, 8]][:n]
    if kind == "LightGBM":
        rounds = int(CFG["compute"]["boosting_rounds"])
        if classification:
            return [{"n_estimators": rounds, "num_leaves": leaves, "learning_rate": 0.05, "max_depth": -1,
                     "min_child_samples": 40, "class_weight": "balanced", "n_jobs": threads, "verbosity": -1} for leaves in [15, 31, 63]][:n]
        return [{"n_estimators": rounds, "num_leaves": leaves, "learning_rate": 0.05, "max_depth": -1,
                 "min_child_samples": 40, "n_jobs": threads, "verbosity": -1} for leaves in [15, 31, 63]][:n]
    raise ValueError(kind)


def build_estimator(kind: str, params: dict, seed: int, classification: bool = False):
    cfg = dict(params)
    if kind == "RandomForest":
        cls = RandomForestClassifier if classification else RandomForestRegressor
        return cls(random_state=seed, n_jobs=CFG["compute"]["model_threads"], **cfg)
    if kind == "XGBoost":
        cls = XGBClassifier if classification else XGBRegressor
        return cls(random_state=seed, **cfg)
    cls = LGBMClassifier if classification else LGBMRegressor
    return cls(random_state=seed, **cfg)


def pick_inner_split(train: pd.DataFrame, lag: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(train.outcome_year.unique())
    candidates = [int(y) for y in years[-2:] if int(y) - lag >= int(years[0])]
    if not candidates:
        return train.iloc[:-max(1, len(train)//5)], train.iloc[-max(1, len(train)//5):]
    val_year = candidates[-1]
    inner_train = train.loc[train.outcome_year.le(val_year - lag)]
    inner_val = train.loc[train.outcome_year.eq(val_year)]
    if len(inner_train) < max(100, int(0.2 * len(train))) or inner_val.empty:
        val_year = int(years[-1])
        inner_train = train.loc[train.outcome_year.lt(val_year)]
        inner_val = train.loc[train.outcome_year.eq(val_year)]
    return inner_train, inner_val


def tune_regressor(kind: str, train: pd.DataFrame, lag: int, drop_groups: list[str] | None = None, trials: int | None = None) -> tuple[dict, float]:
    itrain, ival = pick_inner_split(train, lag)
    nums, cats = feature_columns(drop_groups)
    nums = [c for c in nums if c in train.columns]
    cats = [c for c in cats if c in train.columns]
    prep = make_preprocessor(itrain, drop_groups)
    xtr = prep.fit_transform(itrain[nums + cats]).astype(np.float32)
    xva = prep.transform(ival[nums + cats]).astype(np.float32)
    ytr = np.log1p(itrain.export_value_usd.clip(lower=0).to_numpy())
    best, best_score = None, np.inf
    for params in model_configs(kind, n_trials=trials):
        model = build_estimator(kind, params, seed=11)
        model.fit(xtr, ytr)
        pred_log = model.predict(xva)
        base = inverse_log1p_predictions(pred_log)
        scale = validation_mae_scale(ival.export_value_usd.to_numpy(float), pred_log)
        pred = base * scale
        score = mean_absolute_error(ival.export_value_usd, pred)
        if score < best_score:
            best, best_score = params, score
    return dict(best or model_configs(kind)[0]), float(best_score)


def tune_classifier(kind: str, train: pd.DataFrame, target_col: str, lag: int, trials: int | None = None) -> tuple[dict, float]:
    itrain, ival = pick_inner_split(train, lag)
    nums, cats = feature_columns()
    nums, cats = [c for c in nums if c in train.columns], [c for c in cats if c in train.columns]
    prep = make_preprocessor(itrain)
    xtr = prep.fit_transform(itrain[nums + cats]).astype(np.float32)
    xva = prep.transform(ival[nums + cats]).astype(np.float32)
    ytr = itrain[target_col].astype(int).to_numpy()
    yva = ival[target_col].astype(int).to_numpy()
    if np.unique(ytr).size < 2 or np.unique(yva).size < 2:
        return model_configs(kind, classification=True, n_trials=1)[0], np.nan
    if kind == "XGBoost":
        ratio = max(1.0, float((ytr == 0).sum() / max(1, (ytr == 1).sum())))
    else:
        ratio = 1.0
    best, best_score = None, -np.inf
    for params0 in model_configs(kind, classification=True, n_trials=trials):
        params = dict(params0)
        if kind == "XGBoost":
            params["scale_pos_weight"] = ratio
        model = build_estimator(kind, params, seed=11, classification=True)
        model.fit(xtr, ytr)
        proba = model.predict_proba(xva)[:, 1]
        score = average_precision_score(yva, proba)
        if score > best_score:
            best, best_score = params, score
    return dict(best or model_configs(kind, classification=True)[0]), float(best_score)


@dataclass
class FittedModel:
    preprocessor: object
    estimator: object
    numeric_features: list[str]
    categorical_features: list[str]
    usd_scale_factor: float = 1.0
    feature_groups: dict | None = None

    def predict(self, frame: pd.DataFrame, log_target: bool = True) -> np.ndarray:
        x = self.preprocessor.transform(frame[self.numeric_features + self.categorical_features]).astype(np.float32)
        pred = np.asarray(self.estimator.predict(x), dtype=float)
        if log_target:
            pred = inverse_log1p_predictions(pred) * self.usd_scale_factor
        return pred


def fit_regressor(kind: str, train: pd.DataFrame, params: dict, seed: int, lag: int,
                  drop_groups: list[str] | None = None) -> FittedModel:
    nums, cats = feature_columns(drop_groups)
    nums, cats = [c for c in nums if c in train.columns], [c for c in cats if c in train.columns]
    itrain, ival = pick_inner_split(train, lag)
    prep_val = make_preprocessor(itrain, drop_groups)
    xtr_val = prep_val.fit_transform(itrain[nums + cats]).astype(np.float32)
    xva = prep_val.transform(ival[nums + cats]).astype(np.float32)
    estimator_val = build_estimator(kind, params, seed=seed)
    estimator_val.fit(xtr_val, np.log1p(itrain.export_value_usd.clip(lower=0).to_numpy()))
    val_pred = estimator_val.predict(xva)
    y_val = np.log1p(ival.export_value_usd.clip(lower=0).to_numpy())
    scale = validation_mae_scale(ival.export_value_usd.to_numpy(float), val_pred) if len(ival) else 1.0
    preprocessor = make_preprocessor(train, drop_groups)
    x = preprocessor.fit_transform(train[nums + cats]).astype(np.float32)
    estimator = build_estimator(kind, params, seed=seed)
    estimator.fit(x, np.log1p(train.export_value_usd.clip(lower=0).to_numpy()))
    return FittedModel(preprocessor, estimator, nums, cats, scale, FEATURE_GROUPS)


def fit_classifier(kind: str, train: pd.DataFrame, target_col: str, params: dict, seed: int) -> FittedModel:
    nums, cats = feature_columns()
    nums, cats = [c for c in nums if c in train.columns], [c for c in cats if c in train.columns]
    prep = make_preprocessor(train)
    x = prep.fit_transform(train[nums + cats]).astype(np.float32)
    y = train[target_col].astype(int).to_numpy()
    adjusted = dict(params)
    if kind == "XGBoost":
        adjusted["scale_pos_weight"] = max(1.0, float((y == 0).sum() / max(1, (y == 1).sum())))
    estimator = build_estimator(kind, adjusted, seed=seed, classification=True)
    estimator.fit(x, y)
    return FittedModel(prep, estimator, nums, cats, 1.0, FEATURE_GROUPS)


def logistic_model(train: pd.DataFrame, target_col: str, test: pd.DataFrame | None = None):
    nums, cats = feature_columns()
    nums, cats = [c for c in nums if c in train.columns], [c for c in cats if c in train.columns]
    prep = make_preprocessor(train)
    xtr = prep.fit_transform(train[nums + cats])
    model = LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced", solver="lbfgs", tol=1e-4)
    model.fit(xtr, train[target_col].astype(int))
    if test is None:
        return model, prep, nums, cats
    xte = prep.transform(test[nums + cats])
    return model.predict_proba(xte)[:, 1], model, prep
