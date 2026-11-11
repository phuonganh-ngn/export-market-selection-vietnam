from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import feature_columns

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
DEVICE = torch.device("cpu")
MAX_EPOCHS = 100
PATIENCE = 12
BATCH_SIZE = 1024


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    def __init__(self, d_in: int, width: int, depth: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        d = d_in
        for _ in range(depth):
            layers.extend([nn.Linear(d, width), nn.ReLU(), nn.Dropout(dropout)])
            d = width
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


MLP_CONFIGS = [
    {"width": 64, "depth": 2, "dropout": 0.10, "learning_rate": 0.001, "batch_size": BATCH_SIZE},
    {"width": 128, "depth": 3, "dropout": 0.20, "learning_rate": 0.0003, "batch_size": BATCH_SIZE},
]
TABM_CONFIGS = [
    {"width": 64, "depth": 2, "dropout": 0.10, "learning_rate": 0.001, "batch_size": BATCH_SIZE, "k": 8},
    {"width": 128, "depth": 3, "dropout": 0.20, "learning_rate": 0.0003, "batch_size": BATCH_SIZE, "k": 8},
]


@dataclass
class FoldEncoder:
    model_name: str
    feature_set: str
    num_cols: list[str]
    cat_cols: list[str]
    transformer: object
    category_maps: dict[str, dict[str, int]]
    num_feature_count: int

    @classmethod
    def fit(cls, train: pd.DataFrame, model_name: str, feature_set: str = "F4_full") -> "FoldEncoder":
        num_cols, cat_cols = feature_columns(feature_set)
        absent = [c for c in num_cols + cat_cols if c not in train.columns]
        if absent:
            raise ValueError(f"Missing model input columns: {absent}")
        if model_name == "MLP":
            num = Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("scale", StandardScaler()),
            ])
            cat = Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
            ])
            transformer = ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)], remainder="drop")
            transformer.fit(cls._frame(train, num_cols, cat_cols))
            n_num = transformer.named_transformers_["num"].transform(train[num_cols]).shape[1]
            return cls(model_name, feature_set, num_cols, cat_cols, transformer, {}, int(n_num))
        if model_name == "TabM":
            num_pipe = Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("scale", StandardScaler()),
            ])
            x_num = num_pipe.fit_transform(train[num_cols]).astype(np.float32, copy=False)
            maps = {}
            for c in cat_cols:
                levels = sorted(train[c].fillna("__MISSING__").astype(str).unique().tolist())
                maps[c] = {value: idx + 1 for idx, value in enumerate(levels)}
            return cls(model_name, feature_set, num_cols, cat_cols, num_pipe, maps, int(x_num.shape[1]))
        raise ValueError(model_name)

    @staticmethod
    def _frame(frame: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
        x = frame[num_cols + cat_cols].copy()
        for c in cat_cols:
            x[c] = x[c].fillna("__MISSING__").astype(str)
        return x

    def transform(self, frame: pd.DataFrame):
        if self.model_name == "MLP":
            x = self.transformer.transform(self._frame(frame, self.num_cols, self.cat_cols))
            if sparse.issparse(x):
                x = x.toarray()
            return np.asarray(x, dtype=np.float32), None
        x_num = self.transformer.transform(frame[self.num_cols]).astype(np.float32, copy=False)
        cats = np.empty((len(frame), len(self.cat_cols)), dtype=np.int64)
        for j, col in enumerate(self.cat_cols):
            mapping = self.category_maps[col]
            cats[:, j] = frame[col].fillna("__MISSING__").astype(str).map(mapping).fillna(0).to_numpy(np.int64)
        return x_num, cats

    def get_state(self) -> dict:
        return {"model_name": self.model_name, "feature_set": self.feature_set, "num_cols": self.num_cols,
                "cat_cols": self.cat_cols, "transformer": self.transformer,
                "category_maps": self.category_maps, "num_feature_count": self.num_feature_count}


def create_model(model_name: str, encoder: FoldEncoder, config: dict):
    if model_name == "MLP":
        d_in = len(encoder.transformer.get_feature_names_out())
        return MLP(d_in, config["width"], config["depth"], config["dropout"])
    if model_name == "TabM":
        import tabm
        cardinalities = [len(encoder.category_maps[c]) + 1 for c in encoder.cat_cols]
        return tabm.TabM.make(
            n_num_features=encoder.num_feature_count,
            cat_cardinalities=cardinalities,
            d_out=1,
            n_blocks=config["depth"],
            d_block=config["width"],
            dropout=config["dropout"],
            activation="ReLU",
            k=config["k"],
            arch_type="tabm",
            start_scaling_init="random-signs",
        )
    raise ValueError(model_name)


def _tensor_dataset(x, x_cat, y):
    if x_cat is None:
        return TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32).reshape(-1, 1)))
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(x_cat), torch.from_numpy(y.astype(np.float32).reshape(-1, 1)))


def _forward(model, model_name: str, batch):
    if model_name == "MLP":
        x, y = batch
        return model(x), y
    x_num, x_cat, y = batch
    return model(x_num, x_cat), y


@torch.no_grad()
def predict_raw(model, model_name: str, x, x_cat=None, batch_size: int = BATCH_SIZE) -> np.ndarray:
    model.eval()
    outs = []
    if model_name == "MLP":
        tensor = torch.from_numpy(x)
        for start in range(0, len(tensor), batch_size):
            outs.append(model(tensor[start:start + batch_size].to(DEVICE)).detach().cpu().numpy().reshape(-1))
    else:
        tx, tc = torch.from_numpy(x), torch.from_numpy(x_cat)
        for start in range(0, len(tx), batch_size):
            output = model(tx[start:start + batch_size].to(DEVICE), tc[start:start + batch_size].to(DEVICE))
            # Official TabM semantics: average member probabilities for classification
            # (applied by caller after sigmoid), and average member outputs for regression.
            outs.append(output.squeeze(-1).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def _validation_metric(raw, y, task: str) -> tuple[float, float]:
    if task == "classification":
        from sklearn.metrics import average_precision_score
        if np.unique(y).size < 2:
            return float("inf"), np.nan
        if raw.ndim == 2:
            prob = 1.0 / (1.0 + np.exp(-np.clip(raw, -40, 40)))
            prob = prob.mean(axis=1)
        else:
            prob = 1.0 / (1.0 + np.exp(-np.clip(raw, -40, 40)))
        ap = float(average_precision_score(y, prob))
        return -ap, ap
    from ml_models import validation_mae_scale, inverse_log1p_predictions
    if raw.ndim == 2:
        raw = raw.mean(axis=1)
    factor = validation_mae_scale(np.expm1(y), raw)
    pred = inverse_log1p_predictions(raw) * factor
    mae = float(np.mean(np.abs(np.expm1(y) - pred)))
    return mae, factor


def fit_with_validation(model_name: str, task: str, encoder: FoldEncoder, xtr, cat_tr, ytr, xval, cat_val, yval,
                        config: dict, seed: int, max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE) -> dict:
    """Tune one configuration on a chronological inner validation fold."""
    seed_everything(seed)
    model = create_model(model_name, encoder, config).to(DEVICE)
    class_weight = 1.0
    if task == "classification":
        pos = max(1.0, float(np.sum(ytr)))
        neg = max(1.0, float(len(ytr) - pos))
        class_weight = min(50.0, neg / pos)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    if task == "classification":
        pos_weight = torch.tensor(float(class_weight), dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.MSELoss()
    ds = _tensor_dataset(xtr, cat_tr, ytr)
    loader = DataLoader(ds, batch_size=config["batch_size"], shuffle=True, num_workers=0, drop_last=False)
    best_state = None
    best_value = float("inf")
    best_epoch = 1
    best_aux = np.nan
    stale = 0
    start_time = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in loader:
            batch = tuple(t.to(DEVICE) for t in batch)
            out, yb = _forward(model, model_name, batch)
            if model_name == "TabM":
                out = out.squeeze(-1)
                target = yb.expand_as(out)
            else:
                out = out.reshape(-1)
                target = yb.reshape(-1)
            loss = criterion(out, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        raw = predict_raw(model, model_name, xval, cat_val)
        value, aux = _validation_metric(raw, yval, task)
        improved = best_state is None or value < best_value - max(1e-10, abs(best_value) * 1e-6)
        if improved:
            best_value, best_aux, best_epoch = value, aux, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    elapsed = time.perf_counter() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)
    if task == "classification":
        validation_score = float(-best_value)
        calibration = np.nan
    else:
        validation_score = float(best_value)
        calibration = float(best_aux)
    return {"model": model, "best_epoch": best_epoch, "validation_score": validation_score,
            "calibration_factor": calibration, "class_weight": float(class_weight),
            "epochs_run": epoch, "runtime_seconds": elapsed,
            "parameter_count": int(sum(p.numel() for p in model.parameters()))}


def fit_fixed_epochs(model_name: str, task: str, encoder: FoldEncoder, xtr, cat_tr, ytr,
                     config: dict, seed: int, epochs: int) -> dict:
    seed_everything(seed)
    model = create_model(model_name, encoder, config).to(DEVICE)
    pos_weight_value = 1.0
    if task == "classification":
        pos = max(1.0, float(np.sum(ytr)))
        neg = max(1.0, float(len(ytr) - pos))
        pos_weight_value = min(50.0, neg / pos)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32))
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    loader = DataLoader(_tensor_dataset(xtr, cat_tr, ytr), batch_size=config["batch_size"], shuffle=True,
                        num_workers=0, drop_last=False)
    start_time = time.perf_counter()
    model.train()
    for _ in range(max(1, int(epochs))):
        for batch in loader:
            batch = tuple(t.to(DEVICE) for t in batch)
            out, yb = _forward(model, model_name, batch)
            if model_name == "TabM":
                out = out.squeeze(-1)
                target = yb.expand_as(out)
            else:
                out = out.reshape(-1)
                target = yb.reshape(-1)
            loss = criterion(out, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    elapsed = time.perf_counter() - start_time
    return {"model": model, "epochs_run": max(1, int(epochs)), "runtime_seconds": elapsed,
            "class_weight": pos_weight_value, "parameter_count": int(sum(p.numel() for p in model.parameters()))}


def predict_task(model, model_name: str, task: str, x, cat=None, calibration_factor: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    raw = predict_raw(model, model_name, x, cat)
    if task == "classification":
        members = 1.0 / (1.0 + np.exp(-np.clip(raw, -40, 40)))
        # Per official README, average probabilities rather than averaging logits.
        prob = members.mean(axis=1) if members.ndim == 2 else members
        return prob, raw
    log_prediction = raw.mean(axis=1) if raw.ndim == 2 else raw
    from ml_models import inverse_log1p_predictions
    usd = inverse_log1p_predictions(log_prediction) * float(calibration_factor)
    return usd, log_prediction
