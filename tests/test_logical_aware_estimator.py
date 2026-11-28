"""Tests for the logical-aware (syndrome + logical-bit) estimator.

Covers: correctness of the 32-state DP and gradient, identifiability gain from labels,
recovery of the Z4Z5-burst vs Z3-burst ambiguity, readout -> label handling, and the
end-to-end effect on the bit-exact BP decoder.
"""
from __future__ import annotations

import numpy as np

from simulation.reference.noise_model import (
    LabeledShotAccumulator,
    build_d3_correlated_noise_model,
    build_d3_logical_aware_model,
    logical_bit_from_readout,
)
from simulation.reference.noise_model.logical_class_distribution import (
    LogicalClassDistributionEngine,
)
from simulation.reference.syndrome_stream.causal_stream_runner import CycleAccurateBPEngine

M = build_d3_logical_aware_model()
NAMES = [e.name for e in M.events]
BASE, BURST = 0.01, 0.08


def theta(**kw) -> np.ndarray:
    t = np.full(21, BASE)
    for k, v in kw.items():
        t[NAMES.index(k)] = v
    return t


S2 = theta(ZZ_4_5=BURST)
S3 = theta(Z_3=BURST)


def _fisher_rank(jac_rows: np.ndarray, probs: np.ndarray, tol: float = 1e-9) -> int:
    w = 1.0 / np.maximum(probs, 1e-15)
    fim = jac_rows.T @ (jac_rows * w[:, None])
    ev = np.linalg.eigvalsh(fim)
    return int(np.sum(ev > tol * ev.max()))


# ---------------------------------------------------------------- DP correctness
def test_marginal_over_logical_bit_equals_16_state_dp():
    rng = np.random.default_rng(1)
    old = build_d3_correlated_noise_model()
    for _ in range(5):
        t = rng.uniform(0.001, 0.2, 21)
        got = M.joint_sl_distribution_dp(t)
        assert abs(got.sum() - 1.0) < 1e-12
        assert np.max(np.abs(got.sum(axis=1) - old.joint_distribution_dp(t))) < 1e-12


def test_dp_matches_bruteforce_enumeration():
    rng = np.random.default_rng(2)
    t = np.zeros(21)
    active = rng.choice(21, size=14, replace=False)
    t[active] = rng.uniform(0.01, 0.3, 14)
    assert np.max(np.abs(M.joint_sl_distribution_dp(t) - M.joint_sl_bruteforce(t))) < 1e-12


def test_dp_matches_existing_32_state_oracle_engine():
    eng = LogicalClassDistributionEngine(basis="X")
    t = np.random.default_rng(3).uniform(0.005, 0.1, 21)
    # Compare in a mechanism-order independent way via per-mechanism tables.
    assert sorted(zip(eng.syndrome_signatures.tolist(), eng.logical_flips.tolist())) == sorted(
        zip(M.signatures.tolist(), M.logical_flips.tolist())
    )
    if np.array_equal(eng.syndrome_signatures, M.signatures) and np.array_equal(
        eng.logical_flips, M.logical_flips
    ):
        assert np.max(np.abs(eng.compute_joint_distribution(t) - M.joint_sl_distribution_dp(t))) < 1e-12


def test_bayes_optimum_and_posteriors_match_document():
    for t in (np.full(21, BASE), S2, S3):
        assert abs(100 * M.bayes_optimal_ler(t) - 5.71) < 0.005
    j2, j3 = M.joint_sl_distribution_dp(S2), M.joint_sl_distribution_dp(S3)
    assert abs(j2[4, 0] / j2[4].sum() - 0.82) < 0.005
    assert abs(j3[4, 0] / j3[4].sum() - 0.18) < 0.005


# ---------------------------------------------------------------- gradient
def _fd_grad(fun, x, h=1e-7):
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        g[i] = (fun(xp) - fun(xm)) / (2 * h)
    return g


def test_gradient_matches_finite_differences_labeled_and_mixed():
    rng = np.random.default_rng(4)
    t = rng.uniform(0.01, 0.15, 21)
    csl = rng.integers(1, 500, (16, 2)).astype(float)
    cs = rng.integers(0, 300, 16).astype(float)
    prior = np.full(21, 0.02)
    for c_s in (None, cs):
        f = lambda x: M.negative_log_likelihood_and_grad_sl(x, csl, c_s, prior, 0.05)[0]
        g = M.negative_log_likelihood_and_grad_sl(t, csl, c_s, prior, 0.05)[1]
        assert np.max(np.abs(g - _fd_grad(f, t))) < 1e-6


def test_unlabeled_only_loss_equals_old_histogram_loss():
    t = np.random.default_rng(5).uniform(0.01, 0.1, 21)
    hist = np.random.default_rng(6).integers(1, 400, 16).astype(float)
    old = build_d3_correlated_noise_model()
    nll_old, g_old = old.negative_log_likelihood_and_grad(t, hist)
    nll_new, g_new = M.negative_log_likelihood_and_grad_sl(t, np.zeros((16, 2)), hist)
    assert abs(nll_old - nll_new) < 1e-12
    assert np.max(np.abs(g_old - g_new)) < 1e-9


# ---------------------------------------------------------------- identifiability
def test_labels_raise_fisher_rank_to_number_of_sl_orbits():
    t = np.full(21, BASE)
    dist, jac = M.sl_distribution_and_jacobian(t)
    rows_sl = jac.reshape(32, 21)
    rows_s = jac.sum(axis=1)
    rank_s = _fisher_rank(rows_s, dist.sum(axis=1))
    rank_sl = _fisher_rank(rows_sl, dist.reshape(32))
    orbits = {
        (int(g), int(l))
        for g, l, ok in zip(M.signatures, M.logical_flips, M.observable)
        if ok
    }
    assert rank_s == 9  # documented rank of the histogram-only model
    assert rank_sl == len(orbits)
    assert rank_sl > rank_s


def test_bursts_alias_in_histogram_but_not_in_labeled_joint():
    j2, j3 = M.joint_sl_distribution_dp(S2), M.joint_sl_distribution_dp(S3)
    assert 0.5 * np.sum(np.abs(j2.sum(1) - j3.sum(1))) < 1e-12  # same P(s)
    post2, post3 = j2[:, 0] / j2.sum(1), j3[:, 0] / j3.sum(1)
    assert np.max(np.abs(post2 - post3)) > 0.5  # very different P(l|s)


# ---------------------------------------------------------------- recovery (exact data)
def test_labeled_fit_recovers_orbits_and_optimal_decisions():
    for truth, best_bit_at_0x4 in ((S2, 0), (S3, 1)):
        joint = M.joint_sl_distribution_dp(truth)
        fit = M.fit_lbfgs_labeled(joint * 1e9)
        true_orb, fit_orb = M.orbit_rates(truth), M.orbit_rates(fit["theta"])
        assert max(abs(fit_orb[k] - v) for k, v in true_orb.items()) < 1e-3
        assert fit["map_logical_decision"][4] == best_bit_at_0x4
        assert M.decision_ler(fit["map_logical_decision"], truth) - M.bayes_optimal_ler(truth) < 1e-4


def test_histogram_only_fit_cannot_serve_both_bursts():
    h2 = M.joint_sl_distribution_dp(S2).sum(1) * 1e9
    h3 = M.joint_sl_distribution_dp(S3).sum(1) * 1e9
    assert np.allclose(h2, h3, rtol=1e-9)  # the estimator literally sees the same input
    fit = M.fit_lbfgs(h2, prior_lambda=0.05, max_iter=80)
    d = M.decisions_from_joint(M.joint_sl_distribution_dp(fit["theta"]))["map_logical_decision"]
    worst = max(M.decision_ler(d, S2), M.decision_ler(d, S3))
    assert worst > 0.11  # one of the two truths pays ~12% LER


def test_finite_shot_labeled_fit_picks_correct_alias():
    for seed, (truth, bit) in enumerate(((S2, 0), (S3, 1))):
        rng = np.random.default_rng(100 + seed)
        s, l = M.sample_shots(truth, 50_000, rng)
        acc = LabeledShotAccumulator()
        acc.add_batch(s, l)
        assert acc.n_labeled == 50_000
        fit = M.fit_from_accumulator(acc)
        assert fit["map_logical_decision"][4] == bit


def test_mixed_labeled_and_unlabeled_shots_still_resolve_alias():
    rng = np.random.default_rng(7)
    s, l = M.sample_shots(S3, 60_000, rng)
    acc = LabeledShotAccumulator()
    acc.add_batch(s[:10_000], l[:10_000])  # 1/6 labeled
    acc.add_batch(s[10_000:])  # rest syndrome-only
    assert acc.n_total == 60_000 and acc.n_labeled == 10_000
    fit = M.fit_from_accumulator(acc)
    assert fit["map_logical_decision"][4] == 1


# ---------------------------------------------------------------- readout -> label
def test_logical_bit_from_readout_single_shot_semantics():
    supp = M.logical_support  # X-bar = X0 X3 X6
    def readout(qubits):
        r = np.zeros(9, dtype=np.uint8)
        r[list(qubits)] = 1  # a Z error flips the X-basis readout of that qubit
        return r
    assert logical_bit_from_readout(readout([3]), supp) == 1
    assert logical_bit_from_readout(readout([6]), supp) == 1
    assert logical_bit_from_readout(readout([4, 5]), supp) == 0
    assert logical_bit_from_readout(readout([3, 4, 5]), supp) == 1  # the logical operator
    # decoder correction on the logical support cancels the flip
    assert logical_bit_from_readout(readout([3]), supp, correction_logical=1) == 0
    assert logical_bit_from_readout(readout([3]), supp, prepared_logical=1) == 0


def test_accumulator_readout_path_matches_batch_path():
    rng = np.random.default_rng(8)
    s, l = M.sample_shots(S2, 2000, rng)
    a = LabeledShotAccumulator(logical_support=M.logical_support)
    for si, li in zip(s, l):
        r = np.zeros(9, dtype=np.uint8)
        r[3 if li else 4] = 1  # any readout whose logical parity is li
        assert a.add_readout_shot(int(si), r) == li
    b = LabeledShotAccumulator()
    b.add_batch(s, l)
    assert np.array_equal(a.counts_sl, b.counts_sl)
    assert np.array_equal(a.marginal_histogram(), np.bincount(s, minlength=16))


# ---------------------------------------------------------------- end-to-end on BP decoder
def _bp_ler(engine, q78, truth) -> float:
    d = np.array(
        [int(np.dot(engine.logical_op, engine.decode(s, q78, K=3).correction_bits) & 1) for s in range(16)]
    )
    return M.decision_ler(d, truth)


def test_end_to_end_bp_ler_labeled_fit_matches_oracle_and_beats_histogram_fit():
    eng = CycleAccurateBPEngine(code=M.code, basis="X", default_K=3)
    for truth in (S2, S3):
        joint = M.joint_sl_distribution_dp(truth)
        oracle = _bp_ler(eng, M.compute_q7_8_llrs(truth)[1], truth)
        labeled = _bp_ler(eng, M.fit_lbfgs_labeled(joint * 1e9)["llr_q7_8"], truth)
        old = _bp_ler(eng, M.fit_lbfgs(joint.sum(1) * 1e9, prior_lambda=0.05, max_iter=80)["llr_q7_8"], truth)
        assert abs(labeled - oracle) < 5e-4  # labeled estimator reaches oracle weights
        assert labeled <= old + 1e-9
    # the Z3 burst is the case the histogram fit gets wrong (11.88% vs 5.99%)
    joint3 = M.joint_sl_distribution_dp(S3)
    old3 = _bp_ler(eng, M.fit_lbfgs(joint3.sum(1) * 1e9, prior_lambda=0.05, max_iter=80)["llr_q7_8"], S3)
    new3 = _bp_ler(eng, M.fit_lbfgs_labeled(joint3 * 1e9)["llr_q7_8"], S3)
    assert old3 > 0.11 and new3 < 0.062
