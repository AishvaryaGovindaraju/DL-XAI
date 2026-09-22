"""
models.py
=========
The predictors: two deep architectures and two shallow baselines.

Every model exposes the same three things, which is all the rest of the
pipeline ever asks for:

    predict_proba(X)  -> P(positive), shape (n,)
    embed(X)          -> penultimate representation, shape (n, d)
    has_dropout       -> whether MC-Dropout sampling is meaningful

The representation returned by `embed` is what the explanation-quality head
in selective.py reads. That is the reason the deep models are not
interchangeable with the baselines here: a logistic regression has no
learned representation to condition on.
"""

import numpy as np
import torch
import torch.nn as nn

import config


# ---------------------------------------------------------------------
# Deep model 1: MLP with dropout (MC-Dropout capable)
# ---------------------------------------------------------------------

class MLP(nn.Module):
    """Fully connected net; dropout is kept for uncertainty, not just regularisation."""

    has_dropout = True
    name = "mlp"

    def __init__(self, n_features: int,
                 hidden=config.MLP_HIDDEN, dropout: float = config.MLP_DROPOUT):
        super().__init__()
        layers, width = [], n_features
        for units in hidden:
            layers += [nn.Linear(width, units), nn.BatchNorm1d(units),
                       nn.ReLU(), nn.Dropout(dropout)]
            width = units
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)
        self.embedding_dim = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.body(x))).squeeze(-1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ---------------------------------------------------------------------
# Deep model 2: FT-Transformer
# ---------------------------------------------------------------------

class FeatureTokenizer(nn.Module):
    """Give every feature its own learned token: value * weight + bias."""

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.normal_(self.weight, std=d_token ** -0.5)
        nn.init.normal_(self.bias, std=d_token ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight + self.bias      # (n, f, d)


class FTTransformer(nn.Module):
    """
    Transformer over feature tokens (Gorishniy et al., 2021), compact
    re-implementation. Included because Diabetes-130 is categorical-heavy
    and attention over feature tokens is the current standard deep model
    for that shape of data - it is also what makes "why deep learning"
    answerable beyond an MLP.
    """

    has_dropout = True
    name = "ft_transformer"

    def __init__(self, n_features: int,
                 d_token: int = config.FTT_D_TOKEN,
                 n_blocks: int = config.FTT_N_BLOCKS,
                 n_heads: int = config.FTT_N_HEADS,
                 dropout: float = config.FTT_DROPOUT):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        block = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 2,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(block, num_layers=n_blocks)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)
        self.embedding_dim = d_token

    def _represent(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.norm(encoded[:, 0])                        # CLS token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self._represent(x))).squeeze(-1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self._represent(x)


# ---------------------------------------------------------------------
# Uniform wrapper for torch models
# ---------------------------------------------------------------------

class TorchPredictor:
    """Numpy-facing wrapper: hides batching, eval mode and tensors."""

    is_deep = True

    def __init__(self, module: nn.Module, batch_size: int = 4096):
        self.module = module
        self.batch_size = batch_size
        self.name = getattr(module, "name", module.__class__.__name__)
        self.has_dropout = getattr(module, "has_dropout", False)

    def _batched(self, X: np.ndarray, fn) -> np.ndarray:
        self.module.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                chunk = torch.as_tensor(np.asarray(X[start:start + self.batch_size],
                                                   dtype=np.float32))
                out.append(fn(chunk).cpu().numpy())
        return np.concatenate(out) if out else np.empty(0)

    def predict_proba(self, X) -> np.ndarray:
        return self._batched(np.asarray(X, dtype=np.float32), self.module.forward)

    def embed(self, X) -> np.ndarray:
        return self._batched(np.asarray(X, dtype=np.float32), self.module.embed)


# ---------------------------------------------------------------------
# Shallow baselines - present to answer "is the deep model necessary?"
# ---------------------------------------------------------------------

class SklearnPredictor:
    is_deep = False
    has_dropout = False

    def __init__(self, estimator, name: str):
        self.estimator = estimator
        self.name = name

    def fit(self, X, y):
        self.estimator.fit(np.asarray(X, dtype=np.float32), y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.estimator.predict_proba(np.asarray(X, dtype=np.float32))[:, 1]

    def embed(self, X) -> np.ndarray:
        raise NotImplementedError("shallow baselines have no learned representation")


def build_baselines(seed: int = config.RANDOM_STATE) -> dict:
    """Logistic regression and gradient boosting, both class-balanced."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier

    baselines = {
        "logreg": SklearnPredictor(
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=seed), "logreg"),
        "hgb": SklearnPredictor(
            HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                           early_stopping=True,
                                           random_state=seed), "hgb"),
    }
    try:                                   # preferred GBDT if installed
        from xgboost import XGBClassifier
        baselines["xgboost"] = SklearnPredictor(
            XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", random_state=seed,
                          n_jobs=4), "xgboost")
    except ImportError:
        pass
    return baselines


def build_deep_model(architecture: str, n_features: int, seed: int = None) -> nn.Module:
    """Create one deep model by name; seeding here makes runs reproducible."""
    if seed is not None:
        config.set_seeds(seed)
    if architecture == "mlp":
        return MLP(n_features)
    if architecture == "ft_transformer":
        return FTTransformer(n_features)
    raise ValueError(f"unknown architecture: {architecture}")


DEEP_ARCHITECTURES = ["mlp", "ft_transformer"]


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ============================================================
# CHECKLIST
# - MLP: 256-128-64 with BatchNorm and dropout, sigmoid output
# - FTTransformer: per-feature tokens + CLS transformer encoder, the
#   modern tabular deep architecture, needed for the categorical-heavy
#   Diabetes-130 data
# - Both expose embed() - the penultimate representation the explanation
#   quality head in selective.py conditions on
# - TorchPredictor wraps either model behind a numpy predict_proba/embed
#   interface so explainers never touch tensors
# - build_baselines(): logistic regression + gradient boosting (XGBoost if
#   installed, HistGradientBoosting otherwise) to test whether the deep
#   model is actually needed
# ============================================================
