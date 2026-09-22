"""Phase 3: Bayesian networks, the dynamic Bayesian network, and the kill-chain HMM."""

import itertools

import numpy as np
import pytest

from aegis_system.probabilistic.bayes_net import (
    BayesNet, Factor, attack_chain_network, exploit_probability,
)
from aegis_system.probabilistic.dbn import (
    apt_dbn, slow_and_low_demo, slow_and_low_evidence, sliding_window_detector,
)
from aegis_system.probabilistic.hmm_engine import ALERT_CLASSES, HMM, KILL_CHAIN, kill_chain_hmm

# --------------------------------------------------------------------------- factors


def test_factor_multiply_sum_out_reduce():
    f = Factor(("A",), np.array([0.4, 0.6]))
    g = Factor(("A", "B"), np.array([[0.9, 0.1], [0.2, 0.8]]))
    joint = f * g
    assert joint.vars == ("A", "B")
    assert np.allclose(joint.table, [[0.36, 0.04], [0.12, 0.48]])
    assert np.allclose(joint.sum_out("A").table, [0.48, 0.52])
    assert np.allclose(joint.reduce("B", 1).table, [0.04, 0.48])
    assert np.allclose(joint.normalize().table.sum(), 1.0)


def test_factor_multiply_aligns_shared_variables_in_any_order():
    f = Factor(("B", "A"), np.array([[1.0, 2.0], [3.0, 4.0]]))
    g = Factor(("A",), np.array([10.0, 100.0]))
    h = f * g
    assert h.vars == ("B", "A")
    assert np.allclose(h.table, [[10.0, 200.0], [30.0, 400.0]])


# --------------------------------------------------------------------------- Bayes net


def test_add_node_validates_structure_and_cpts():
    net = BayesNet()
    net.add_boolean("A", [], 0.3)
    with pytest.raises(ValueError):
        net.add_boolean("B", ["Missing"], [0.1, 0.2])  # parent must exist: no cycles possible
    with pytest.raises(ValueError):
        net.add_boolean("A", [], 0.5)  # re-adding would allow a cycle
    with pytest.raises(ValueError):
        net.add_node("C", ["A"], [[0.5, 0.6], [0.5, 0.5]])  # row does not sum to 1
    with pytest.raises(ValueError):
        net.add_node("C", ["A"], [0.5, 0.5])  # wrong shape


def test_node_with_two_parents():
    # The old implementation crashed on any node with two or more parents.
    net = attack_chain_network()
    assert net.nodes["VulnExploit"].parents == ("PortScan", "PhishingEmail")
    dist = net.variable_elimination("VulnExploit", {"PortScan": True, "PhishingEmail": True})
    assert dist[True] == pytest.approx(net.nodes["VulnExploit"].cpt[1, 1, 1])


def _all_queries(net):
    names = net.order
    for X in names:
        others = [v for v in names if v != X]
        for k in range(0, 3):
            for ev_vars in itertools.combinations(others, k):
                for values in itertools.product((False, True), repeat=k):
                    yield X, dict(zip(ev_vars, values))


def test_variable_elimination_matches_enumeration():
    net = attack_chain_network()
    for X, e in _all_queries(net):
        try:
            exact = net.enumerate_ask(X, e)
        except ZeroDivisionError:
            continue
        for order in ("min-degree", "topological"):
            ve = net.variable_elimination(X, e, order=order)
            for value in (False, True):
                assert abs(ve[value] - exact[value]) < 1e-9, (X, e, order)


def test_headline_query():
    net = attack_chain_network()
    dist = net.variable_elimination("Exfiltration", {"PortScan": True, "PrivEsc": True})
    # Given PrivEsc, Exfiltration is independent of PortScan: 0.8 * 0.75 + 0.2 * 0.01
    assert dist[True] == pytest.approx(0.602)
    assert net.variable_elimination("Exfiltration", {"PrivEsc": True})[True] == pytest.approx(dist[True])


def test_explaining_away():
    net = attack_chain_network()
    phish = net.variable_elimination("PhishingEmail", {"VulnExploit": True})[True]
    phish_given_scan = net.variable_elimination("PhishingEmail", {"VulnExploit": True, "PortScan": True})[True]
    assert phish_given_scan < phish


def test_explicit_elimination_order_and_validation():
    net = attack_chain_network()
    e = {"Exfiltration": True}
    hidden = ["PortScan", "PhishingEmail", "VulnExploit", "LateralMove"]
    order = list(reversed(hidden))
    assert net.variable_elimination("PrivEsc", e, order=order)[True] == pytest.approx(
        net.enumerate_ask("PrivEsc", e)[True])
    with pytest.raises(ValueError):
        net.variable_elimination("PrivEsc", e, order=["PortScan"])


def test_likelihood_weighting_approximates_exact():
    net = attack_chain_network()
    e = {"PortScan": True, "PrivEsc": True}
    exact = net.variable_elimination("Exfiltration", e)[True]
    approx = net.likelihood_weighting("Exfiltration", e, n=5000, seed=0)[True]
    assert approx == pytest.approx(exact, abs=0.03)


def test_non_boolean_domain():
    net = BayesNet()
    net.add_node("Severity", [], [0.5, 0.3, 0.2], domain=("low", "medium", "high"))
    net.add_node("Page", ["Severity"], [[0.95, 0.05], [0.7, 0.3], [0.1, 0.9]])
    dist = net.variable_elimination("Severity", {"Page": True})
    assert dist == pytest.approx(net.enumerate_ask("Severity", {"Page": True}))
    assert max(dist, key=dist.get) == "high"


def test_cvss_derived_prior():
    assert exploit_probability(3.9) == pytest.approx(0.99)
    assert exploit_probability(1.95) == pytest.approx(0.5)


# --------------------------------------------------------------------------- HMM


def test_hmm_accepts_dict_parameters():
    hmm = HMM(
        states=["Recon", "Exploit"],
        observations=[0, 1],
        start_probs={"Recon": 0.6, "Exploit": 0.4},
        trans_probs={"Recon": {"Recon": 0.7, "Exploit": 0.3}, "Exploit": {"Recon": 0.1, "Exploit": 0.9}},
        emit_probs={"Recon": {0: 0.8, 1: 0.2}, "Exploit": {0: 0.3, 1: 0.7}},
    )
    alpha, c = hmm.forward([0, 1, 1])
    assert np.allclose(alpha.sum(axis=1), 1.0)
    assert hmm.viterbi([0, 1, 1])[-1] == "Exploit"


def _brute_force(hmm, obs):
    """P(e) and P(X_t | e) by summing over every state sequence."""
    o = hmm.encode(obs)
    n, T = len(hmm.states), len(o)
    total = 0.0
    marg = np.zeros((T, n))
    best, best_p = None, -1.0
    for seq in itertools.product(range(n), repeat=T):
        p = hmm.pi[seq[0]] * hmm.B[seq[0], o[0]]
        for t in range(1, T):
            p *= hmm.A[seq[t - 1], seq[t]] * hmm.B[seq[t], o[t]]
        total += p
        for t, s in enumerate(seq):
            marg[t, s] += p
        if p > best_p:
            best, best_p = seq, p
    return total, marg / total, [hmm.states[i] for i in best]


def test_hmm_matches_brute_force():
    hmm = kill_chain_hmm()
    obs = ["probe", "normal", "r2l", "u2r", "normal"]
    total, marg, best = _brute_force(hmm, obs)
    assert hmm.log_likelihood(obs) == pytest.approx(np.log(total))
    assert np.allclose(hmm.smooth(obs), marg)
    assert hmm.viterbi(obs) == best


def test_filter_is_forward_marginal():
    hmm = kill_chain_hmm()
    obs = ["probe", "probe", "r2l"]
    _, marg_prefix, _ = _brute_force(hmm, obs[:2])
    assert np.allclose(hmm.filter(obs)[1], marg_prefix[1])


def test_hmm_long_sequence_does_not_underflow():
    hmm = kill_chain_hmm()
    states, obs = hmm.sample(10_000, seed=1)
    ll = hmm.log_likelihood(obs)
    assert np.isfinite(ll) and ll < 0
    gamma = hmm.smooth(obs)
    assert np.all(np.isfinite(gamma))
    assert np.allclose(gamma.sum(axis=1), 1.0)
    path = hmm.viterbi(obs)
    assert len(path) == 10_000 and path[-1] == "Exfiltration"


def test_smoothing_recovers_true_phase_most_of_the_time():
    hmm = kill_chain_hmm()
    states, obs = hmm.sample(400, seed=3)
    guess = [KILL_CHAIN[i] for i in hmm.smooth(obs).argmax(axis=1)]
    assert np.mean([g == s for g, s in zip(guess, states)]) > 0.8


def test_kill_chain_is_left_to_right():
    hmm = kill_chain_hmm()
    assert hmm.states == list(KILL_CHAIN) and hmm.observations == list(ALERT_CLASSES)
    assert np.allclose(np.tril(hmm.A, -1), 0)  # never moves back a phase


def test_baum_welch_increases_likelihood_and_keeps_structure():
    truth = kill_chain_hmm()
    data = [truth.sample(60, seed=s)[1] for s in range(20)]
    model = kill_chain_hmm()
    model.B = np.full_like(model.B, 1.0 / model.B.shape[1])  # start from uninformative emissions
    history = model.baum_welch(data, n_iter=15)
    assert all(b >= a - 1e-6 for a, b in zip(history, history[1:]))
    assert history[-1] > history[0]
    assert np.allclose(np.tril(model.A, -1), 0)
    assert np.allclose(model.A.sum(axis=1), 1.0) and np.allclose(model.B.sum(axis=1), 1.0)


def test_unknown_observation_rejected():
    with pytest.raises(ValueError):
        kill_chain_hmm().forward(["teleport"])


# --------------------------------------------------------------------------- DBN


def _brute_force_dbn(dbn, evidence_seq, var):
    """P(var_T | e_1:T) by enumerating every joint state trajectory."""
    states = list(itertools.product((0, 1), repeat=len(dbn.state_vars)))
    prior = dbn.initial_belief().reshape(-1)
    T = dbn.transition_matrix()
    k = dbn.state_vars.index(var)
    num = den = 0.0
    for traj in itertools.product(range(len(states)), repeat=len(evidence_seq)):
        p = prior[traj[0]]
        for t, s in enumerate(traj):
            if t > 0:
                p *= T[traj[t - 1], s]
            p *= dbn.sensor_likelihood(evidence_seq[t])[s]
        den += p
        if states[traj[-1]][k]:
            num += p
    return num / den


def test_dbn_exact_filter_matches_brute_force():
    dbn = apt_dbn(alert_rate=0.3)
    evidence = [{"Alert": False}, {"Alert": True, "Spike": False}, {"Spike": True}, {"Alert": True}]
    beliefs = dbn.filter(evidence)
    for var in dbn.state_vars:
        assert dbn.marginal(beliefs[-1], var) == pytest.approx(_brute_force_dbn(dbn, evidence, var))
    for b in beliefs:
        assert b.sum() == pytest.approx(1.0)


def test_dbn_transition_rows_sum_to_one():
    T = apt_dbn().transition_matrix()
    assert np.allclose(T.sum(axis=1), 1.0)


def test_particle_filter_tracks_exact_filter():
    dbn = apt_dbn(alert_rate=0.2)
    evidence = slow_and_low_evidence(ticks=80, every=5, start=10, seed=2)
    exact = [dbn.marginal(b, "Foothold") for b in dbn.filter(evidence)]
    approx = [e["Foothold"] for e in dbn.particle_filter(evidence, n=3000, seed=0)]
    assert np.mean(np.abs(np.array(exact) - np.array(approx))) < 0.05


def test_dbn_rejects_bad_models():
    dbn = apt_dbn()
    with pytest.raises(ValueError):
        dbn.add_transition("Alert", ["Foothold"], [0.1, 0.2])  # not a state variable
    with pytest.raises(ValueError):
        dbn.add_sensor("Noise", ["Unknown"], [0.1, 0.2])
    with pytest.raises(KeyError):
        dbn.sensor_likelihood({"Unmodelled": True})


def test_sliding_window_detector():
    ev = [{"Alert": a} for a in [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    fired = sliding_window_detector(ev, window=3, threshold=3)
    assert fired[2] and not fired[3] and not fired[-1]


def test_slow_and_low_apt_detected_by_dbn_but_not_by_threshold():
    demo = slow_and_low_demo(ticks=300, every=10, seed=0)
    belief = demo["belief"]
    assert belief[0] < 0.05  # before the attacker arrives at tick 20
    assert max(belief[:20]) < 0.5
    assert belief[-1] > 0.9  # the DBN accumulates the weak evidence
    assert not any(demo["detector"])  # the window detector never fires
