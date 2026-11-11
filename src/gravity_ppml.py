from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import sparse

from common import CFG
from temporal_validation import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def _design(train: pd.DataFrame, test: pd.DataFrame | None = None):
    excluded = set(CFG["study"].get("excluded_features") or [])
    nums = [c for c in NUMERIC_FEATURES if c in train.columns and c not in excluded]
    cats = [c for c in CATEGORICAL_FEATURES if c in train.columns]
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    xnum = imputer.fit_transform(train[nums]).astype(float)
    scaler = StandardScaler(with_mean=False).fit(xnum)
    xnum = scaler.transform(xnum)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=float).fit(train[cats])
    xcat = enc.transform(train[cats])
    x = sparse.hstack([sparse.csr_matrix(xnum), xcat], format="csr")
    if test is None:
        return x, (nums, cats, imputer, scaler, enc)
    xt_num = scaler.transform(imputer.transform(test[nums])).astype(float)
    xt_cat = enc.transform(test[cats])
    xt = sparse.hstack([sparse.csr_matrix(xt_num), xt_cat], format="csr")
    return x, xt, (nums, cats, imputer, scaler, enc)


def fit_ppml(train: pd.DataFrame, test: pd.DataFrame, maxiter: int | None = None):
    xtr, xte, objects = _design(train, test)
    y = train.export_value_usd.to_numpy(dtype=float) / 1e6
    if (y < 0).any() or not np.isfinite(y).all():
        raise ValueError("PPML requires finite nonnegative outcomes")
    # Log-link Poisson pseudo-maximum likelihood with a small L2 penalty to stabilize
    # the many destination/product indicators and avoid separated zero-flow cells.
    # The config value is the single source of truth; the fallback equals the value used for the
    # archived models (0.1), not 0.01, which did not converge (reports/RESEARCH_INTEGRITY.md).
    alpha=float(CFG["compute"].get("gravity_ridge_alpha",0.1))
    maxiter = int(maxiter or CFG["compute"].get("gravity_max_iter", 2000))
    result=PoissonRegressor(alpha=alpha,max_iter=maxiter,tol=1e-4,fit_intercept=True)
    result.fit(xtr,y)
    mu=np.asarray(result.predict(xte),dtype=float)
    pred = mu * 1e6
    pred = np.maximum(0.0, pred)
    mu_train=np.maximum(result.predict(xtr),1e-300)
    deviance_terms=np.where(y>0,y*np.log(np.maximum(y,1e-300)/mu_train),0)-(y-mu_train)
    diagnostics = {"converged": bool(result.n_iter_ < maxiter), "iterations": int(result.n_iter_),
                   "deviance": float(2*np.sum(deviance_terms)), "n_parameters": int(xtr.shape[1]+1), "n_train": len(train),
                   "ridge_alpha":alpha,"message": "Log-link Poisson pseudo-maximum likelihood; response scaled to USD millions, L2 regularization stabilizes destination/product effects"}
    return pred, result, diagnostics, objects
