"""Time-to-compromise (TTC) regression.

No public TTC dataset exists, so the target is generated from a documented
formula plus noise and the report says so. For a host with

    patch_lag      days since its last patch
    betweenness    betweenness centrality in the topology (how many attack paths cross it)
    alert_rate     alerts per hour seen on it
    credential     0 user, 1 local admin, 2 domain admin credentials cached on it

the hours until an attacker compromises it are

    ttc = 96 / ((1 + 0.03 * patch_lag) * (1 + 6 * betweenness) * (1 + 0.4 * alert_rate) * 2 ** credential)

times log-normal noise (sigma 0.2). Two distractor features (uptime_days,
ram_gb) have no effect, so the Lasso comparison can show coefficients going to
exactly zero. The formula is multiplicative, so a linear model on raw features
underfits and gradient boosting should win; fitting log(ttc) makes it nearly linear.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TTC_FEATURES = ["patch_lag", "betweenness", "alert_rate", "credential", "uptime_days", "ram_gb"]
INFORMATIVE = ["patch_lag", "betweenness", "alert_rate", "credential"]


def ttc_formula(patch_lag, betweenness, alert_rate, credential):
    return 96.0 / ((1 + 0.03 * np.asarray(patch_lag)) * (1 + 6 * np.asarray(betweenness))
                   * (1 + 0.4 * np.asarray(alert_rate)) * 2.0 ** np.asarray(credential))


def generate_ttc_dataset(n: int = 2000, graph: Optional[nx.DiGraph] = None, seed: int = 0
                         ) -> Tuple[pd.DataFrame, pd.Series]:
    """Synthetic hosts with TTC targets. Betweenness is sampled from a real topology's nodes."""
    from ..search.graph_builder import generate_topology

    rng = np.random.default_rng(seed)
    graph = graph if graph is not None else generate_topology(n_workstations=30, n_servers=8, seed=seed)
    centrality = np.array(list(nx.betweenness_centrality(graph).values()))
    X = pd.DataFrame({
        "patch_lag": rng.gamma(2.0, 20.0, n),
        "betweenness": rng.choice(centrality, n),
        "alert_rate": rng.exponential(1.5, n),
        "credential": rng.choice([0, 1, 2], n, p=[0.6, 0.3, 0.1]),
        "uptime_days": rng.uniform(1, 365, n),
        "ram_gb": rng.choice([8, 16, 32, 64], n),
    })
    noise = rng.lognormal(0.0, 0.2, n)
    y = pd.Series(ttc_formula(X.patch_lag, X.betweenness, X.alert_rate, X.credential) * noise, name="ttc_hours")
    return X, y


def regressor_zoo(seed: int = 0) -> Dict[str, Pipeline]:
    alphas = np.logspace(-3, 2, 30)
    return {
        "ridge": Pipeline([("scale", StandardScaler()), ("model", RidgeCV(alphas=alphas))]),
        "lasso": Pipeline([("scale", StandardScaler()),
                           ("model", LassoCV(alphas=alphas, cv=5, random_state=seed, max_iter=20000))]),
        "gradient_boosting": Pipeline([("scale", StandardScaler()),
                                       ("model", GradientBoostingRegressor(random_state=seed, n_estimators=300,
                                                                           max_depth=3, learning_rate=0.05))]),
    }


def compare_regressors(X: pd.DataFrame, y: pd.Series, seed: int = 0, log_target: bool = False
                       ) -> Tuple[List[Dict[str, object]], Dict[str, Pipeline]]:
    """Test RMSE, MAE and R^2 for Ridge, Lasso and gradient boosting, plus the coefficients Lasso zeroes."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed)
    target = np.log if log_target else (lambda v: v)
    inverse = np.exp if log_target else (lambda v: v)
    rows, fitted = [], {}
    for name, pipe in regressor_zoo(seed).items():
        pipe.fit(X_tr, target(y_tr))
        pred = inverse(pipe.predict(X_te))
        row = {
            "model": name,
            "target": "log(ttc)" if log_target else "ttc",
            "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
            "mae": float(mean_absolute_error(y_te, pred)),
            "r2": float(r2_score(y_te, pred)),
        }
        m = pipe.named_steps["model"]
        if hasattr(m, "coef_"):
            row["coefficients"] = {f: float(c) for f, c in zip(X.columns, m.coef_)}
            row["alpha"] = float(m.alpha_)
        if name == "lasso":
            row["zeroed"] = [f for f, c in zip(X.columns, m.coef_) if abs(c) < 1e-10]
        rows.append(row)
        fitted[name] = pipe
    return rows, fitted


def lasso_coefficient_path(X: pd.DataFrame, y: pd.Series, n_alphas: int = 60) -> Dict[str, object]:
    """Standardized Lasso coefficients across alphas, largest alpha first.

    Reading it from left to right shows the order in which features enter the
    model; the distractors enter last (or never).
    """
    from sklearn.linear_model import lasso_path

    Xs = StandardScaler().fit_transform(X)
    yc = np.asarray(y) - np.mean(y)
    alpha_max = np.abs(Xs.T @ yc).max() / len(yc)  # smallest alpha at which every coefficient is zero
    grid = alpha_max * np.logspace(0, -3, n_alphas)  # explicit grid: n_alphas is deprecated in sklearn 1.9
    alphas, coefs, _ = lasso_path(Xs, yc, alphas=grid)
    entry = {}
    for f, row in zip(X.columns, coefs):
        nonzero = np.flatnonzero(np.abs(row) > 1e-10)
        entry[f] = float(alphas[nonzero[0]]) if len(nonzero) else None
    return {"alphas": alphas.tolist(), "coefficients": {f: row.tolist() for f, row in zip(X.columns, coefs)},
            "entry_alpha": entry}


class TimeToCompromiseRegressor:
    """Gradient boosting on log(ttc), the model the closed loop uses for urgency."""

    def __init__(self, seed: int = 0):
        self.pipeline = regressor_zoo(seed)["gradient_boosting"]
        self.log_target = True
        self.columns: Optional[List[str]] = None

    def fit(self, X, y) -> "TimeToCompromiseRegressor":
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float)
        self.log_target = bool(np.all(y > 0))
        self.columns = list(X.columns)
        self.pipeline.fit(X, np.log(y) if self.log_target else y)
        return self

    def predict(self, X) -> np.ndarray:
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X, dtype=float), columns=self.columns)
        pred = self.pipeline.predict(X)
        return np.exp(pred) if self.log_target else pred

    def predict_host(self, features: Dict[str, float]) -> float:
        return float(self.predict(pd.DataFrame([features], columns=self.columns))[0])


def host_features(graph: nx.DiGraph, host: str, patch_lag: float, alert_rate: float,
                  centrality: Optional[Dict[str, float]] = None, uptime_days: float = 30.0,
                  ram_gb: float = 16.0) -> Dict[str, float]:
    """TTC features for one host of a topology (credential level from its cached admin credentials)."""
    centrality = centrality if centrality is not None else nx.betweenness_centrality(graph)
    attrs = graph.nodes[host]
    credential = 2 if attrs.get("type") == "domain_controller" else 1 if attrs.get("admin_credentials") else 0
    return {"patch_lag": patch_lag, "betweenness": centrality.get(host, 0.0), "alert_rate": alert_rate,
            "credential": credential, "uptime_days": uptime_days, "ram_gb": ram_gb}


__all__: Sequence[str] = ["INFORMATIVE", "TTC_FEATURES", "TimeToCompromiseRegressor", "compare_regressors",
                          "generate_ttc_dataset", "host_features", "lasso_coefficient_path", "regressor_zoo", "ttc_formula"]
