"""Readout noise on the logical label (plan C5): what do wrong labels cost?  (real RTL, osd_mode=1, K=3)

A label is l = parity of the data-qubit readout on {0,3,6} (+ prep). If each data qubit is misread with
probability q_m, the label flips with probability q = 3 q_m (1-q_m)^2 + q_m^3 (about 3 q_m). Here q is the
LABEL flip probability and is swept directly.

Each labeled shot is flipped independently with probability q (the window of unlabeled shots is untouched).
Two estimators see the same noisy counts:
  naive  : fit_hybrid from test_label_budget.py (the audited one; does not know about q)
  aware  : same likelihood, but labeled counts follow P_obs(s,l) = (1-q) P(s,l) + q P(s,1-l), q given as an
           input (it must be calibrated separately, e.g. from repeated known-state readouts)
Seeds: labeled shots, window and label noise use three separate streams.

Defaults: Z4Z5 burst (the case where labels matter), N in {64,256,512,1024}, 30 repeats, ~5 minutes.
Quick:    AVID_NOISE_REPEATS=10 pytest tests/unit/test_readout_noise.py -v -s
Both bursts: AVID_NOISE_BURSTS=both

Run (repo root):  pytest tests/unit/test_readout_noise.py -v -s
Output: tests/unit/artifacts/label_noise_<burst>.csv
"""
import csv
import os
import time

import numpy as np
import pytest
from scipy.optimize import minimize

try:
    import test_label_budget as TLB
except ImportError:
    from tests.unit import test_label_budget as TLB

ENG, LG, BASE, BURSTS, joint, ler, llrs, rtl = TLB.ENG, TLB.LG, TLB.BASE, TLB.BURSTS, TLB.joint, TLB.ler, TLB.llrs, TLB.rtl
Q_LIST = [0.0, 0.01, 0.03, 0.05, 0.10, 0.50]
N_LIST = [64, 256, 512, 1024]
REPEATS = int(os.environ.get("AVID_NOISE_REPEATS", 30))
BURST_LIST = list(BURSTS) if os.environ.get("AVID_NOISE_BURSTS") == "both" else ["Z4Z5 burst"]
SEED_NOISE = 333


def flip_labels(lab, q, rng):
    """Flip each labeled shot's l with probability q (per-cell binomial)."""
    moved = rng.binomial(lab.astype(int), q)
    out = lab.astype(float) - moved
    out[:, 0] += moved[:, 1]
    out[:, 1] += moved[:, 0]
    return out


def fit_aware(labeled32, window16, q, lam=TLB.LAM):
    n = float(labeled32.sum() + window16.sum())
    z0 = LG(BASE)

    def nll(z):
        J = ENG.compute_joint_distribution(1 / (1 + np.exp(-z)))
        Jobs = (1 - q) * J + q * J[:, ::-1]
        data = (-(labeled32 * np.log(np.maximum(Jobs, 1e-300))).sum()
                - (window16 * np.log(np.maximum(J.sum(axis=1), 1e-300))).sum())
        return (data + lam * (n / TLB.PRIOR_REF) * np.sum((z - z0) ** 2)) / n

    r = minimize(nll, z0, method="L-BFGS-B", options={"maxiter": 200})
    return llrs(1 / (1 + np.exp(-r.x)))


def test_flip_labels_conserves_shots_and_flips_at_rate_q():
    rng = np.random.default_rng(1)
    lab = np.zeros((16, 2)); lab[4, 0] = 100000
    out = flip_labels(lab, 0.1, rng)
    assert out.sum() == lab.sum()
    assert abs(out[4, 1] / 100000 - 0.1) < 0.01
    assert np.array_equal(flip_labels(lab, 0.0, rng), lab)


@pytest.mark.parametrize("name", BURST_LIST)
def test_readout_noise_sweep(rtl, name, capsys):
    k = BURSTS[name]
    other = [v for n, v in BURSTS.items() if n != name][0]
    theta, theta_o = TLB.theta_burst(k), TLB.theta_burst(other)
    opt = 100 * joint(theta).min(axis=1).sum()
    right = 100 * ler(theta, rtl(llrs(theta)))
    wrong = 100 * ler(theta, rtl(llrs(theta_o)))
    thr = (right + wrong) / 2
    windows = [TLB.draw_window(theta, TLB.W, np.random.default_rng([TLB.SEED_WINDOW, k, r])) for r in range(REPEATS)]

    rows, t0 = [], time.time()
    for q in Q_LIST:
        for n in N_LIST:
            naive, aware = [], []
            for r in range(REPEATS):
                lab = TLB.draw_labeled(theta, n, np.random.default_rng([TLB.SEED_LABELS, k, n, r]))
                lab = flip_labels(lab, q, np.random.default_rng([SEED_NOISE, k, n, r, int(q * 1000)]))
                naive.append(100 * ler(theta, rtl(TLB.fit_hybrid(lab, windows[r]))))
                aware.append(100 * ler(theta, rtl(fit_aware(lab, windows[r], q))))
            row = dict(burst=name, q=q, N=n, repeats=REPEATS)
            for tag, v in (("naive", naive), ("aware", aware)):
                m, ci, fw = TLB.summarize(v, thr)
                row.update({f"{tag}_ler": m, f"{tag}_ci95": ci, f"{tag}_wrong": fw})
            rows.append(row)
    elapsed = time.time() - t0

    TLB.ART.mkdir(exist_ok=True)
    with open(TLB.ART / f"label_noise_{name.split()[0]}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    with capsys.disabled():
        print(f"\n=== label readout noise, {name}, {REPEATS} repeats, {elapsed:.0f}s ===")
        print(f"optimum {opt:.2f}%   right weights {right:.2f}%   wrong alias {wrong:.2f}%   threshold {thr:.2f}%")
        print(f"{'q':>5} {'N':>5} | {'naive LER':>16} {'wrong':>6} | {'aware LER':>16} {'wrong':>6}")
        for r in rows:
            print(f"{r['q']:5.2f} {r['N']:5d} | {r['naive_ler']:7.2f} +-{r['naive_ci95']:5.2f} {r['naive_wrong']:6.0%} | "
                  f"{r['aware_ler']:7.2f} +-{r['aware_ci95']:5.2f} {r['aware_wrong']:6.0%}")

    by = {(r["q"], r["N"]): r for r in rows}
    for r in rows:
        assert min(r["naive_ler"], r["aware_ler"]) >= opt - 1e-6                    # nothing beats the optimum
    for n in N_LIST:                                                                # q = 0: the two fits coincide
        assert abs(by[(0.0, n)]["naive_ler"] - by[(0.0, n)]["aware_ler"]) < 1e-9
    top = max(N_LIST)                                                               # q = 0.5: labels carry no information
    assert by[(0.5, top)]["naive_ler"] > by[(0.0, top)]["naive_ler"] + 0.5
