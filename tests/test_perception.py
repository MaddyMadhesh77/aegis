"""Phase 6: datasets, model selection, rule extraction, TTC regression, and FGSM."""

import numpy as np
import pandas as pd
import pytest
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from aegis_system.core.environment import FEATURES
from aegis_system.knowledge.production_system import ProductionSystem
from aegis_system.perception.adversarial_test import (
    FGSM, adversarial_training, forward, input_gradient, loss, robustness_curves,
)
from aegis_system.perception.classifier import (
    AttackClassifier, compare_models, extract_rules, flow_facts_for, indicator_names, learning_curves,
    make_pipeline, model_complexity, model_zoo, one_standard_error_choice, tree_paths, validation_curves,
)
from aegis_system.perception.datasets import (
    ATTACK_CATEGORY, CATEGORIES, NSL_CATEGORICAL, NSL_MUTABLE, load_nsl_kdd, nsl_kdd_available,
    stratified_sample, synthetic_flows,
)
from aegis_system.perception.regressor import (
    TimeToCompromiseRegressor, compare_regressors, generate_ttc_dataset, host_features, lasso_coefficient_path,
    ttc_formula,
)
from aegis_system.search.graph_builder import example_topology

needs_nsl = pytest.mark.skipif(not nsl_kdd_available(), reason="run scripts/fetch_nsl_kdd.py")


@pytest.fixture(scope="module")
def flows():
    return synthetic_flows(ticks=600, seed=1, restart_every=30)


@pytest.fixture(scope="module")
def split(flows):
    X, y = flows[FEATURES], flows["label"]
    cut = int(0.7 * len(X))
    return X.iloc[:cut], y.iloc[:cut], X.iloc[cut:], y.iloc[cut:]


@pytest.fixture(scope="module")
def mlp_pipe(split):
    X, y, _, _ = split
    return make_pipeline(MLPClassifier((16,), max_iter=500, random_state=0), X).fit(X, y)


# --------------------------------------------------------------------------- datasets


def test_attack_mapping_uses_the_five_categories():
    assert set(ATTACK_CATEGORY.values()) == set(CATEGORIES)


@needs_nsl
def test_nsl_kdd_matches_published_counts():
    X, y = load_nsl_kdd("train")
    assert X.shape == (125_973, 41)
    assert y.value_counts().to_dict() == {"normal": 67343, "dos": 45927, "probe": 11656, "r2l": 995, "u2r": 52}
    Xt, yt = load_nsl_kdd("test")
    assert len(Xt) == 22_544 and set(yt) == set(CATEGORIES)


def test_missing_dataset_gives_a_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_nsl_kdd"):
        load_nsl_kdd(data_dir=tmp_path)


def test_stratified_sample_keeps_rare_classes():
    X = pd.DataFrame({"a": range(1000)})
    y = pd.Series(["common"] * 995 + ["rare"] * 5)
    Xs, ys = stratified_sample(X, y, 100, seed=0)
    assert set(ys) == {"common", "rare"} and 95 <= len(ys) <= 105


def test_synthetic_flows_are_seeded_and_labelled(flows):
    again = synthetic_flows(ticks=600, seed=1, restart_every=30)
    pd.testing.assert_frame_equal(flows, again)
    assert set(flows["label"]) <= set(CATEGORIES)
    assert (flows.loc[~flows.malicious, "label"] == "normal").all()
    assert {"probe", "r2l", "u2r", "dos"} <= set(flows.loc[flows.malicious, "label"])


# --------------------------------------------------------------------------- classification


def test_attack_classifier_gives_probabilities_and_alerts(flows, split):
    X, y, X_test, y_test = split
    clf = AttackClassifier().fit(X, y)
    proba = clf.predict_proba(X_test)
    assert proba.shape == (len(X_test), len(clf.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.mean(np.array(clf.predict(X_test)) == y_test.values) > 0.9
    alerts = clf.alerts(flows.iloc[len(X):], FEATURES, min_confidence=0.6)
    assert alerts and all(a.attack_class != "normal" and a.confidence >= 0.6 for a in alerts)
    assert {a.host for a in alerts} <= set(flows["src"])


def test_compare_models_reports_every_metric(split):
    rows, fitted = compare_models(*split, specs=model_zoo(fast=True), folds=3)
    assert [r["model"] for r in rows] == ["decision_tree", "random_forest", "svc_rbf", "mlp"]
    for r in rows:
        assert 0 <= r["cv_f1_macro"] <= 1 and 0 <= r["test_f1_macro"] <= 1
        assert r["complexity"] == model_complexity(fitted[r["model"]]) > 0
        assert not np.isnan(r["test_roc_auc_ovr"])
    assert rows[2]["train_rows"] <= 1501  # SVC trained on a subsample


def test_one_standard_error_rule_prefers_simpler_model():
    rows = [
        {"model": "big", "cv_f1_macro": 0.90, "cv_f1_se": 0.02, "complexity": 5000},
        {"model": "small", "cv_f1_macro": 0.89, "cv_f1_se": 0.02, "complexity": 100},
        {"model": "tiny", "cv_f1_macro": 0.80, "cv_f1_se": 0.02, "complexity": 10},
    ]
    assert one_standard_error_choice(rows) == "small"   # within 1 SE of best, fewer parameters
    rows[1]["cv_f1_macro"] = 0.87
    assert one_standard_error_choice(rows) == "big"     # now outside the band


def test_learning_and_validation_curves(split):
    X, y, _, _ = split
    lc = learning_curves(DecisionTreeClassifier(max_depth=4, random_state=0), X, y, sizes=(0.3, 1.0), folds=3)
    assert len(lc["sizes"]) == 2 and lc["sizes"][0] < lc["sizes"][1]
    vc = validation_curves(DecisionTreeClassifier(random_state=0), X, y, "max_depth", [1, 3, 8], folds=3)
    assert vc["train"][0] <= vc["train"][-1]  # deeper trees fit the training data at least as well


# --------------------------------------------------------------------------- rule extraction


def _run_rules(pipe, X):
    ps = ProductionSystem(extract_rules(pipe))
    ind = indicator_names(pipe)
    for i, rec in X.iterrows():
        for fact in flow_facts_for(f"f{i}", rec.to_dict(), ind):
            ps.assert_fact(fact)
    ps.run(10_000)
    return {f.args[0].name: f.args[1].name for f in ps.facts("Classified")}


def test_extracted_rules_reproduce_the_tree(split):
    X, y, X_test, _ = split
    tree = make_pipeline(DecisionTreeClassifier(max_depth=5, random_state=0), X).fit(X, y)
    sample = X_test.head(150)
    got = _run_rules(tree, sample)
    assert [got[f"f{i}"] for i in sample.index] == list(tree.predict(sample))
    paths = tree_paths(tree)
    assert sum(p.samples for p in paths) == len(X)
    assert all(op in ("<=", ">") for p in paths for _, op, _ in p.conditions)


@needs_nsl
def test_extracted_rules_reproduce_the_tree_on_nsl_kdd_with_categoricals():
    X, y = load_nsl_kdd(sample=3000, seed=0)
    Xt, _ = load_nsl_kdd("test", sample=600, seed=0)
    tree = make_pipeline(DecisionTreeClassifier(max_depth=6, random_state=0), X, NSL_CATEGORICAL).fit(X, y)
    sample = Xt.head(120)
    got = _run_rules(tree, sample)
    assert [got[f"f{i}"] for i in sample.index] == list(tree.predict(sample))
    assert any("=" in f for p in tree_paths(tree) for f, _, _ in p.conditions)  # one-hot conditions present


# --------------------------------------------------------------------------- regression


@pytest.fixture(scope="module")
def ttc():
    return generate_ttc_dataset(1500, seed=0)


def test_ttc_formula_direction():
    base = ttc_formula(10, 0.1, 1.0, 0)
    assert ttc_formula(60, 0.1, 1.0, 0) < base       # older patches fall faster
    assert ttc_formula(10, 0.1, 1.0, 2) == pytest.approx(base / 4)  # domain admin creds: 4x faster


def test_regressor_comparison(ttc):
    X, y = ttc
    rows = {r["model"]: r for r in compare_regressors(X, y, seed=0)[0]}
    assert rows["gradient_boosting"]["r2"] > rows["ridge"]["r2"]
    assert "ram_gb" in rows["lasso"]["zeroed"]
    assert set(rows["lasso"]["zeroed"]).isdisjoint({"patch_lag", "alert_rate", "credential"})


def test_lasso_path_enters_distractors_last(ttc):
    entry = lasso_coefficient_path(*ttc)["entry_alpha"]
    informative = min(entry[f] for f in ("patch_lag", "alert_rate", "credential"))
    assert max(entry["uptime_days"] or 0, entry["ram_gb"] or 0) < informative


def test_ttc_regressor_runtime(ttc):
    X, y = ttc
    model = TimeToCompromiseRegressor().fit(X, y)
    g = example_topology()
    dc = model.predict_host(host_features(g, "DomainController", patch_lag=60, alert_rate=3))
    ws = model.predict_host(host_features(g, "Workstation1", patch_lag=5, alert_rate=0.1))
    assert 0 < dc < ws
    assert host_features(g, "Workstation2", 1, 1)["credential"] == 1


# --------------------------------------------------------------------------- adversarial


def test_forward_matches_sklearn(split, mlp_pipe):
    Z = mlp_pipe.named_steps["pre"].transform(split[2].head(50))
    assert np.allclose(forward(mlp_pipe.named_steps["model"], Z)[2], mlp_pipe.predict_proba(split[2].head(50)))


def test_input_gradient_matches_finite_differences(split, mlp_pipe):
    mlp = mlp_pipe.named_steps["model"]
    Z = mlp_pipe.named_steps["pre"].transform(split[0].head(10))
    y = list(split[1].head(10))
    g = input_gradient(mlp, Z, y)
    h = 1e-6
    for j in range(Z.shape[1]):
        e = np.zeros_like(Z)
        e[:, j] = h
        num = (loss(mlp, Z + e, y) - loss(mlp, Z - e, y)) / (2 * h)
        assert np.allclose(g[:, j], num, atol=1e-6)


def test_fgsm_changes_only_mutable_features_and_respects_zero(split, mlp_pipe):
    mutable = ["bytes_out", "duration"]
    attack = FGSM(mlp_pipe, mutable)
    Z = attack.transform(split[2])
    Z_adv = attack.perturb(Z, split[3], eps=3.0)
    names = list(mlp_pipe.named_steps["pre"].get_feature_names_out())
    fixed = [i for i, n in enumerate(names) if n not in mutable]
    assert np.array_equal(Z_adv[:, fixed], Z[:, fixed])
    assert (Z_adv[:, names.index("bytes_out")] >= attack.lower[names.index("bytes_out")] - 1e-12).all()
    assert np.array_equal(attack.perturb(Z, split[3], 0.0), Z)


def test_fgsm_lowers_accuracy_and_adversarial_training_helps(split, mlp_pipe):
    X, y, X_test, y_test = split
    robust = adversarial_training(mlp_pipe, X, y, FEATURES, eps=1.0)
    curves = robustness_curves(mlp_pipe, robust, X_test, y_test, FEATURES, [0.0, 0.5, 1.0])
    std = [r["accuracy"] for r in curves["standard"]]
    adv = [r["accuracy"] for r in curves["adversarially_trained"]]
    assert std[-1] < std[0]
    assert adv[-1] > std[-1]


def test_fgsm_requires_mlp_and_known_features(split):
    X, y, _, _ = split
    tree = make_pipeline(DecisionTreeClassifier(), X).fit(X, y)
    with pytest.raises(TypeError):
        FGSM(tree, FEATURES)
    mlp = make_pipeline(MLPClassifier((4,), max_iter=50, random_state=0), X).fit(X, y)
    with pytest.raises(ValueError):
        FGSM(mlp, ["not_a_feature"])


def test_nsl_mutable_features_exist():
    from aegis_system.perception.datasets import NSL_NUMERIC
    assert set(NSL_MUTABLE) <= set(NSL_NUMERIC)
