"""Label-budget sweep (plan case C4): how many labeled calibration shots N are needed?

Setup (real RTL, osd_mode=1, K=3, exact LER for grading):
  * labeled shots   : N draws from P(s, l)          -> 32-bin counts      (seed stream A)
  * unlabeled window: W draws from P(s)             -> 16-bin histogram   (seed stream B)
  * estimator       : fit_hybrid(labeled32, window16) sees COUNTS ONLY, never theta or truth
  * grading         : exact LER of the RTL decoder under the fitted weights (needs no sampling)

fit_hybrid maximises one likelihood: labeled shots contribute log P(s, l), unlabeled shots
contribute log P(s) = log sum_l P(s, l). Where no labeled shot exists the 16-bin evidence alone
acts, so N = 0 is a pure 16-bin fit.

Run (from the repo root):
    pytest tests/unit/test_label_budget.py -v
Bigger run (N up to 1024, 100 repeats, both bursts; slow):
    AVID_FULL_SWEEP=1 pytest tests/unit/test_label_budget.py -v
Optional cost parameters (microseconds per labeled shot, from Tirtha):
    AVID_T_ROUND_US=... AVID_T_PREP_US=... pytest tests/unit/test_label_budget.py -v
"""
import csv
import inspect
import os
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

try:
    from test_burst_sweeps import BASE, BURSTS, ENG, LG, M, joint, ler, llrs, rtl  # noqa: F401
except ImportError:
    from tests.unit.test_burst_sweeps import BASE, BURSTS, ENG, LG, M, joint, ler, llrs, rtl  # noqa: F401

FULL = os.environ.get("AVID_FULL_SWEEP") == "1"
N_LIST = [0, 64, 128, 256, 512, 1024] if FULL else [0, 64, 256]
REPEATS = 100 if FULL else 20
W = int(os.environ.get("AVID_WINDOW", 4096))          # unlabeled window length (rounds)
P_BURST = 0.08
LAM = 0.5                                              # inherited from fit32; NOT tuned on these scenarios
PRIOR_REF = 1e4                                        # fit32 used 1e4 counts; lam scales with n
SEED_LABELS, SEED_WINDOW = 111, 222                    # two separate random streams
T_ROUND_US = float(os.environ["AVID_T_ROUND_US"]) if "AVID_T_ROUND_US" in os.environ else None
T_PREP_US = float(os.environ["AVID_T_PREP_US"]) if "AVID_T_PREP_US" in os.environ else None
ART = Path(__file__).parent / "artifacts"


def theta_burst(k):
    t = BASE.copy()
    t[k] = P_BURST
    return t


# ---------------------------------------------------------------- simulator side (knows the truth)
def draw_labeled(theta, n, rng):
    p = joint(theta).ravel()
    return rng.multinomial(n, p / p.sum()).reshape(16, 2).astype(float)


def draw_window(theta, w, rng):
    p = joint(theta).sum(axis=1)
    return rng.multinomial(w, p / p.sum()).astype(float)


# ---------------------------------------------------------------- estimator side (counts only)
def fit_hybrid(labeled32, window16, lam=LAM):
    n = float(labeled32.sum() + window16.sum())
    z0 = LG(BASE)

    def nll(z):
        J = ENG.compute_joint_distribution(1 / (1 + np.exp(-z)))
        marg = J.sum(axis=1)
        data = (-(labeled32 * np.log(np.maximum(J, 1e-300))).sum()
                - (window16 * np.log(np.maximum(marg, 1e-300))).sum())
        return (data + lam * (n / PRIOR_REF) * np.sum((z - z0) ** 2)) / n

    r = minimize(nll, z0, method="L-BFGS-B", options={"maxiter": 200})
    return llrs(1 / (1 + np.exp(-r.x)))


def fit_repo16(window16):
    """The repo's own 16-bin estimator on the same finite window (baseline)."""
    h = window16 / window16.sum() * 256
    return np.array(M.fit_lbfgs(h, prior_lambda=0.5, max_iter=15)["llr_q7_8"])


def summarize(vals, thr):
    v = np.asarray(vals)
    ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
    return float(v.mean()), float(ci), float((v > thr).mean())


# ================================================================== tests
def test_estimator_sees_only_counts():
    """Leakage guard: the estimator signature has no theta / truth argument."""
    assert list(inspect.signature(fit_hybrid).parameters) == ["labeled32", "window16", "lam"]


def test_streams_are_separate_and_sized():
    theta = theta_burst(3)
    a = draw_labeled(theta, 64, np.random.default_rng([SEED_LABELS, 3, 64, 0]))
    b = draw_window(theta, W, np.random.default_rng([SEED_WINDOW, 3, 0]))
    assert SEED_LABELS != SEED_WINDOW
    assert a.sum() == 64 and a.shape == (16, 2)
    assert b.sum() == W and b.shape == (16,)
    assert draw_labeled(theta, 0, np.random.default_rng(1)).sum() == 0


@pytest.mark.parametrize("name", list(BURSTS))
def test_label_budget_sweep(rtl, name, capsys):
    k = BURSTS[name]
    other = [v for n, v in BURSTS.items() if n != name][0]
    theta, theta_o = theta_burst(k), theta_burst(other)
    J = joint(theta)
    optimum = 100 * np.minimum(J[:, 0], J[:, 1]).sum()
    right = 100 * ler(theta, rtl(llrs(theta)))             # weights tuned to the true burst
    wrong = 100 * ler(theta, rtl(llrs(theta_o)))           # weights tuned to the aliased burst
    thr = (right + wrong) / 2                              # above this = wrong alias

    t0 = time.time()
    windows = [draw_window(theta, W, np.random.default_rng([SEED_WINDOW, k, rep])) for rep in range(REPEATS)]
    repo = [100 * ler(theta, rtl(fit_repo16(w))) for w in windows]
    rows, allv = [], []
    for n in N_LIST:
        vals = []
        for rep in range(REPEATS):
            lab = draw_labeled(theta, n, np.random.default_rng([SEED_LABELS, k, n, rep]))
            vals.append(100 * ler(theta, rtl(fit_hybrid(lab, windows[rep]))))
        m, ci, fw = summarize(vals, thr)
        cost = n * (T_ROUND_US + T_PREP_US) if (T_ROUND_US is not None and T_PREP_US is not None) else None
        rows.append(dict(burst=name, N=n, repeats=REPEATS, mean_ler=m, ci95=ci, frac_wrong_alias=fw,
                         repo16_mean=float(np.mean(repo)), overhead_pct=100 * n / (n + W), cost_us=cost))
        allv += vals
    elapsed = time.time() - t0

    ART.mkdir(exist_ok=True)
    stem = "label_budget_" + name.split()[0]
    with open(ART / f"{stem}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(range(len(N_LIST)), [r["mean_ler"] for r in rows], yerr=[r["ci95"] for r in rows],
                marker="o", capsize=3, label="hybrid fit (95% CI)")
    ax.axhline(np.mean(repo), ls="--", color="gray", label="repo 16-bin fit")
    ax.axhline(right, ls=":", color="green", label=f"oracle weights ({right:.2f}%)")
    ax.axhline(optimum, ls=":", color="black", label=f"Bayes optimum ({optimum:.2f}%)")
    ax.set_xticks(range(len(N_LIST)))
    ax.set_xticklabels(N_LIST)
    ax.set_xlabel(f"labeled shots N (unlabeled window W={W})")
    ax.set_ylabel("exact LER on RTL (%)")
    ax.set_title(f"{name}: label budget, {REPEATS} repeats")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ART / f"{stem}.png", dpi=150)
    plt.close(fig)

    with capsys.disabled():
        print(f"\n=== label budget, {name} (p={P_BURST}), W={W}, {REPEATS} repeats, {elapsed:.0f}s ===")
        print(f"optimum {optimum:.2f}%   right weights {right:.2f}%   wrong alias {wrong:.2f}%   threshold {thr:.2f}%")
        print(f"repo 16-bin fit on same windows: mean {np.mean(repo):.2f}%")
        print(f"{'N':>6} {'mean LER':>9} {'+-95%':>7} {'wrong alias':>12} {'overhead':>9} {'cost us':>9}")
        for r in rows:
            c = "pending" if r["cost_us"] is None else f"{r['cost_us']:.0f}"
            print(f"{r['N']:>6} {r['mean_ler']:>9.2f} {r['ci95']:>7.2f} {r['frac_wrong_alias']:>11.0%} "
                  f"{r['overhead_pct']:>8.1f}% {c:>9}")

    assert min(allv + repo) >= optimum - 1e-6                         # nothing can beat the Bayes floor
    assert rows[-1]["mean_ler"] <= rows[0]["mean_ler"] + 0.5          # labels do not hurt
