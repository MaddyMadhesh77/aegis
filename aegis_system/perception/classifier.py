"""Attack classification: a model zoo, model selection, and rule extraction.

Every model is a pipeline: a ColumnTransformer (one-hot categoricals, standardize
numerics) followed by the estimator. GridSearchCV with stratified 5-fold
cross-validation picks each model's hyperparameters on macro-F1, so rare
classes (u2r has 52 training rows) count as much as common ones.

Model selection follows the one-standard-error rule: among the models whose
mean CV score is within one standard error of the best, take the one with the
fewest learned parameters. That is how "prefer the simpler model" (Occam's
razor) becomes a concrete, checkable choice.

extract_rules() turns a fitted decision tree into production rules for the
Phase 5 production system, with thresholds mapped back to raw feature units.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve, validation_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ..core.logic import Atom, Const
from ..core.types import Alert
from ..knowledge.production_system import Rule, threshold_rule

SEED = 0


# --------------------------------------------------------------------------- pipelines


def _as_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    arr = np.asarray(X, dtype=float)
    return pd.DataFrame(arr, columns=[f"x{i}" for i in range(arr.shape[1])])


def make_preprocessor(X: pd.DataFrame, categorical: Optional[Sequence[str]] = None,
                      scale: bool = True) -> ColumnTransformer:
    """One-hot the categoricals; standardize the numerics, or pass them through (scale=False, for trees)."""
    if categorical is None:
        categorical = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) in ("category", "string", "str")]
    numeric = [c for c in X.columns if c not in categorical]
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(categorical)),
         ("num", StandardScaler() if scale else "passthrough", numeric)],
        verbose_feature_names_out=False,
    )


def make_pipeline(model, X: pd.DataFrame, categorical: Optional[Sequence[str]] = None) -> Pipeline:
    """Trees split on raw values (scaling changes nothing for them and keeps thresholds readable)."""
    scale = not isinstance(model, (DecisionTreeClassifier, RandomForestClassifier))
    return Pipeline([("pre", make_preprocessor(X, categorical, scale)), ("model", model)])


@dataclass
class ModelSpec:
    name: str
    estimator: object
    grid: Dict[str, list]
    max_train: Optional[int] = None  # SVC scales badly: fit it on a stratified subsample


def model_zoo(seed: int = SEED, fast: bool = False) -> List[ModelSpec]:
    """The four models compared in the report. fast=True shrinks grids for tests."""
    if fast:
        return [
            ModelSpec("decision_tree", DecisionTreeClassifier(random_state=seed), {"model__max_depth": [4, 8]}),
            ModelSpec("random_forest", RandomForestClassifier(n_estimators=30, random_state=seed, n_jobs=-1),
                      {"model__max_depth": [8]}),
            ModelSpec("svc_rbf", calibrated_svc(seed), {"model__estimator__C": [1.0]}, max_train=1500),
            ModelSpec("mlp", MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=seed),
                      {"model__alpha": [1e-3]}),
        ]
    return [
        ModelSpec("decision_tree", DecisionTreeClassifier(random_state=seed),
                  {"model__max_depth": [4, 8, 12, 20], "model__min_samples_leaf": [1, 5]}),
        ModelSpec("random_forest", RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
                  {"model__max_depth": [12, None], "model__min_samples_leaf": [1, 3]}),
        ModelSpec("svc_rbf", calibrated_svc(seed), {"model__estimator__C": [1.0, 10.0]}, max_train=5000),
        ModelSpec("mlp", MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, early_stopping=True,
                                       random_state=seed),
                  {"model__hidden_layer_sizes": [(64,), (64, 32)], "model__alpha": [1e-4, 1e-3]}),
    ]


def calibrated_svc(seed: int = SEED) -> CalibratedClassifierCV:
    """RBF SVC with calibrated probabilities (sklearn 1.9 replaces SVC(probability=True) with this)."""
    return CalibratedClassifierCV(SVC(kernel="rbf", random_state=seed), ensemble=False)


def model_complexity(pipeline: Pipeline) -> int:
    """Number of learned parameters, the yardstick for "simplest" in the 1-SE rule."""
    m = pipeline.named_steps["model"]
    if isinstance(m, CalibratedClassifierCV):
        m = m.calibrated_classifiers_[0].estimator
    if isinstance(m, DecisionTreeClassifier):
        return int(m.tree_.node_count)
    if isinstance(m, RandomForestClassifier):
        return int(sum(t.tree_.node_count for t in m.estimators_))
    if isinstance(m, SVC):
        return int(m.support_vectors_.size + m.dual_coef_.size)
    if isinstance(m, MLPClassifier):
        return int(sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_))
    raise TypeError(f"No complexity measure for {type(m).__name__}")


# --------------------------------------------------------------------------- comparison


def _subsample(X: pd.DataFrame, y: pd.Series, n: Optional[int], seed: int):
    if n is None or n >= len(X):
        return X, y
    from .datasets import stratified_sample
    return stratified_sample(X, y, n, seed)


def macro_auc(model, X, y) -> float:
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    present = [c for c in classes if c in set(y)]
    if len(present) < 2:
        return float("nan")
    cols = [classes.index(c) for c in present]
    p = proba[:, cols]
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    mask = np.isin(np.asarray(y), present)
    return float(roc_auc_score(np.asarray(y)[mask], p[mask], multi_class="ovr", average="macro", labels=present))


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    specs: Optional[Sequence[ModelSpec]] = None,
    folds: int = 5,
    seed: int = SEED,
    categorical: Optional[Sequence[str]] = None,
    n_jobs: int = 1,
) -> Tuple[List[Dict[str, object]], Dict[str, Pipeline]]:
    """Grid-search each model; report CV macro-F1 (mean, std, standard error) and test macro-F1 / ROC-AUC."""
    X, X_test = _as_frame(X), _as_frame(X_test)
    specs = list(specs if specs is not None else model_zoo(seed))
    rows, fitted = [], {}
    for spec in specs:
        Xs, ys = _subsample(X, y, spec.max_train, seed)
        cv = StratifiedKFold(n_splits=min(folds, int(ys.value_counts().min()) if ys.value_counts().min() >= 2 else 2),
                             shuffle=True, random_state=seed)
        search = GridSearchCV(make_pipeline(clone(spec.estimator), Xs, categorical), spec.grid,
                              scoring="f1_macro", cv=cv, n_jobs=n_jobs, error_score="raise")
        start = time.perf_counter()
        search.fit(Xs, ys)
        seconds = time.perf_counter() - start
        best = search.best_estimator_
        i = search.best_index_
        mean = float(search.cv_results_["mean_test_score"][i])
        std = float(search.cv_results_["std_test_score"][i])
        pred = best.predict(X_test)
        rows.append({
            "model": spec.name,
            "params": {k.replace("model__", "").replace("estimator__", ""): v
                       for k, v in search.best_params_.items()},
            "train_rows": len(Xs),
            "cv_f1_macro": mean,
            "cv_f1_std": std,
            "cv_f1_se": float(std / np.sqrt(cv.get_n_splits())),
            "test_f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "test_accuracy": float(np.mean(pred == np.asarray(y_test))),
            "test_roc_auc_ovr": macro_auc(best, X_test, y_test),
            "complexity": model_complexity(best),
            "fit_seconds": round(seconds, 2),
        })
        fitted[spec.name] = best
    return rows, fitted


def one_standard_error_choice(rows: Sequence[Dict[str, object]]) -> str:
    """Simplest model (fewest parameters) whose CV score is within one standard error of the best."""
    best = max(rows, key=lambda r: r["cv_f1_macro"])
    cutoff = best["cv_f1_macro"] - best["cv_f1_se"]
    eligible = [r for r in rows if r["cv_f1_macro"] >= cutoff]
    return min(eligible, key=lambda r: (r["complexity"], r["model"]))["model"]


# --------------------------------------------------------------------------- bias-variance


def learning_curves(model, X, y, sizes=(0.1, 0.25, 0.5, 0.75, 1.0), folds: int = 5, seed: int = SEED,
                    scoring: str = "f1_macro"):
    """Train and validation scores against training-set size.

    With a class of only a few rows (u2r), prefer scoring="f1_weighted": macro-F1
    skips classes absent from a validation fold, which can put validation above training.
    """
    X = _as_frame(X)
    pipe = model if isinstance(model, Pipeline) else make_pipeline(model, X)
    n, train, val = learning_curve(pipe, X, y, train_sizes=list(sizes), scoring=scoring,
                                   cv=StratifiedKFold(folds, shuffle=True, random_state=seed), n_jobs=1)
    return {"sizes": n.tolist(), "train": train.mean(axis=1).tolist(), "validation": val.mean(axis=1).tolist(),
            "train_std": train.std(axis=1).tolist(), "validation_std": val.std(axis=1).tolist()}


def validation_curves(model, X, y, param: str, values: Sequence, folds: int = 5, seed: int = SEED,
                      scoring: str = "f1_macro"):
    """Train and validation scores against one hyperparameter (e.g. tree depth, SVM C)."""
    X = _as_frame(X)
    pipe = model if isinstance(model, Pipeline) else make_pipeline(model, X)
    train, val = validation_curve(pipe, X, y, param_name=f"model__{param}", param_range=list(values),
                                  scoring=scoring, cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
                                  n_jobs=1)
    return {"param": param, "values": list(values), "train": train.mean(axis=1).tolist(),
            "validation": val.mean(axis=1).tolist()}


# --------------------------------------------------------------------------- the classifier used at runtime


class AttackClassifier:
    """A fitted pipeline with the runtime conveniences the closed loop needs."""

    def __init__(self, model=None, categorical: Optional[Sequence[str]] = None, random_state: int = SEED,
                 n_estimators: int = 50):
        self.model = model if model is not None else RandomForestClassifier(n_estimators=n_estimators,
                                                                            random_state=random_state)
        self.categorical = categorical
        self.pipeline: Optional[Pipeline] = None

    def fit(self, X, y) -> "AttackClassifier":
        X = _as_frame(X)
        self.pipeline = make_pipeline(clone(self.model), X, self.categorical)
        self.pipeline.fit(X, list(y))
        return self

    def _check(self) -> Pipeline:
        if self.pipeline is None:
            raise ValueError("Model has not been fitted yet.")
        return self.pipeline

    @property
    def classes_(self) -> List[str]:
        return list(self._check().classes_)

    def predict(self, X) -> List[str]:
        return self._check().predict(_as_frame(X)).tolist()

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities, so the HMM and Bayes net get confidences rather than bare labels."""
        return self._check().predict_proba(_as_frame(X))

    def alerts(self, flows: pd.DataFrame, features: Sequence[str], benign: str = "normal",
               min_confidence: float = 0.5, host_column: str = "src") -> List[Alert]:
        """One Alert per flow whose most likely class is an attack with enough confidence."""
        proba = self.predict_proba(flows[list(features)])
        classes = self.classes_
        out = []
        for i, row in enumerate(proba):
            k = int(np.argmax(row))
            if classes[k] != benign and row[k] >= min_confidence:
                record = flows.iloc[i]
                out.append(Alert(host=str(record[host_column]), attack_class=classes[k], confidence=float(row[k]),
                                 features={f: float(record[f]) for f in features}))
        return out


# --------------------------------------------------------------------------- rule extraction


@dataclass
class TreeRule:
    conditions: List[Tuple[str, str, float]]  # (feature, "<=" | ">", threshold) in raw units
    label: str
    confidence: float
    samples: int

    def __str__(self) -> str:
        cond = " and ".join(f"{f} {op} {t:.4g}" for f, op, t in self.conditions) or "true"
        return f"if {cond} then {self.label} (confidence {self.confidence:.2f}, {self.samples} samples)"


def tree_paths(pipeline: Pipeline, min_confidence: float = 0.0, min_samples: int = 1) -> List[TreeRule]:
    """Every root-to-leaf path of the pipeline's decision tree, with thresholds in raw units.

    Tree pipelines pass numerics through unscaled (if a scaler is present, thresholds
    are mapped back with t * scale + mean). A one-hot column becomes the indicator
    feature "protocol_type=tcp".
    """
    tree = pipeline.named_steps["model"]
    if not isinstance(tree, DecisionTreeClassifier):
        raise TypeError("extract_rules needs a decision tree pipeline")
    pre: ColumnTransformer = pipeline.named_steps["pre"]
    names, unscale = _feature_mapping(pre)
    t = tree.tree_
    classes = list(tree.classes_)
    rules: List[TreeRule] = []

    def walk(node: int, path: List[Tuple[str, str, float]]) -> None:
        if t.children_left[node] == -1:  # leaf
            counts = t.value[node][0]
            k = int(np.argmax(counts))
            conf = float(counts[k] / counts.sum())
            samples = int(t.n_node_samples[node])
            if conf >= min_confidence and samples >= min_samples:
                rules.append(TreeRule(list(path), classes[k], conf, samples))
            return
        j = t.feature[node]
        thr = unscale[j](t.threshold[node])
        walk(t.children_left[node], path + [(names[j], "<=", thr)])
        walk(t.children_right[node], path + [(names[j], ">", thr)])

    walk(0, [])
    return rules


def _feature_mapping(pre: ColumnTransformer):
    names: List[str] = []
    unscale = []
    for name, transformer, cols in pre.transformers_:
        if len(cols) == 0:  # e.g. no categorical columns: that transformer is never fitted
            continue
        if name == "cat":
            for col, cats in zip(cols, transformer.categories_):
                for c in cats:
                    names.append(f"{col}={c}")
                    unscale.append(lambda v: float(v))
        elif name == "num" and isinstance(transformer, StandardScaler):
            for col, mean, scale in zip(cols, transformer.mean_, transformer.scale_):
                names.append(col)
                unscale.append(lambda v, m=mean, s=scale: float(v * s + m))
        elif name == "num":  # passthrough
            for col in cols:
                names.append(col)
                unscale.append(lambda v: float(v))
    return names, unscale


def extract_rules(pipeline: Pipeline, min_confidence: float = 0.0, min_samples: int = 1) -> List[Rule]:
    """Production rules, one per leaf: Feature(flow, f, v) conditions with threshold tests -> Classified(flow, label).

    Salience is the leaf's confidence in percent, so purer leaves win conflicts.
    """
    return [threshold_rule(f"tree_leaf_{i}", r.conditions, r.label, salience=int(round(100 * r.confidence)),
                           confidence=r.confidence)
            for i, r in enumerate(tree_paths(pipeline, min_confidence, min_samples))]


def flow_facts_for(flow_id: str, record: Dict[str, object], indicators: Sequence[str] = ()) -> List[Atom]:
    """Working-memory facts for one flow, matching what extract_rules' conditions expect.

    indicators are the one-hot names from indicator_names(pipeline); a categorical
    field gets 1.0 for its own value and 0.0 for every other category, because a
    tree condition such as "protocol_type=tcp <= 0.5" means "not TCP".
    """
    prefixes = {name.split("=", 1)[0] for name in indicators}
    facts = []
    for key, value in record.items():
        if key in prefixes:
            own = f"{key}={value}"
            for name in indicators:
                if name.startswith(key + "="):
                    facts.append(Atom("Feature", (Const(flow_id), Const(name), Const("1.0" if name == own else "0.0"))))
        else:
            facts.append(Atom("Feature", (Const(flow_id), Const(key), Const(repr(float(value))))))
    return facts


def indicator_names(pipeline: Pipeline) -> List[str]:
    """All indicator features the tree could test, so absent categories can be asserted as 0."""
    names, _ = _feature_mapping(pipeline.named_steps["pre"])
    return [n for n in names if "=" in n]
