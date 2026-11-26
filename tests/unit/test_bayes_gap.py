"""Where does the gap to the Bayes optimum sit?  (real RTL, osd_mode=1, K=3, exact LER)

For each syndrome s the best possible failure probability is min_l P(s, l). The RTL decoder, run with
the oracle weights llrs(theta), fails with probability f(s) >= min_l P(s, l). The excess f(s) - min_l P(s, l)
summed over s is exactly (LER - Bayes optimum), so the table shows which syndromes cost the gap and
whether the decoder's logical decision there disagrees with the most likely class (MAP).

Run (repo root):  pytest tests/unit/test_bayes_gap.py -v
"""
import numpy as np
import pytest

try:
    from test_burst_sweeps import BASE, BURSTS, LOG, SYN, fail_per_syndrome, joint, llrs, rtl  # noqa: F401
except ImportError:
    from tests.unit.test_burst_sweeps import BASE, BURSTS, LOG, SYN, fail_per_syndrome, joint, llrs, rtl  # noqa: F401

CASES = ["uniform", "Z3 burst", "Z4Z5 burst"]


def theta_for(name, p=0.08):
    t = BASE.copy()
    if name != "uniform":
        t[BURSTS[name]] = p
    return t


@pytest.mark.parametrize("name", CASES)
def test_gap_by_syndrome(rtl, name, capsys):
    theta = theta_for(name)
    J = joint(theta)                                   # (16, 2) exact P(s, l)
    pmin = J.min(axis=1)                               # best possible failure mass per syndrome
    pmap = (J[:, 1] > J[:, 0]).astype(int)             # most likely logical class per syndrome
    corr = [c & 0x1FF for c in rtl(llrs(theta))]       # RTL correction masks under the oracle weights
    f = fail_per_syndrome(theta, corr)                 # decoder failure mass per syndrome
    dec = [int(LOG[c]) for c in corr]                  # logical class the decoder's correction represents
    valid = [int(SYN[c]) == s for s, c in enumerate(corr)]
    excess = f - pmin
    ler, opt = f.sum(), pmin.sum()
    gap = ler - opt

    with capsys.disabled():
        print(f"\n=== gap by syndrome: {name} (oracle weights) ===")
        print(f"LER {100 * ler:.3f}%   Bayes optimum {100 * opt:.3f}%   gap {100 * gap:.3f} points")
        print(f"{'synd':>5} {'P(s)':>9} {'MAP':>4} {'dec':>4} {'valid':>6} {'fail':>10} {'min':>10} {'excess':>10}")
        for s in np.argsort(-excess):
            if excess[s] > 1e-5:
                print(f"0x{s:02x}  {J[s].sum():9.6f} {pmap[s]:>4} {dec[s]:>4} {str(valid[s]):>6} "
                      f"{f[s]:10.6f} {pmin[s]:10.6f} {excess[s]:10.6f}")
        if gap < 1e-5:
            print("(no syndrome loses more than 1e-5)")
        disagree = [s for s in range(16) if valid[s] and dec[s] != pmap[s] and J[s].sum() > 1e-4]
        invalid = [s for s in range(16) if not valid[s]]
        print(f"syndromes where the decoder's class differs from MAP: {[hex(s) for s in disagree]}")
        print(f"syndromes where the correction does not match the syndrome: {[hex(s) for s in invalid]}")

    assert excess.min() >= -1e-9                       # no decoder can beat min_l P(s, l)
    assert abs(excess.sum() - gap) < 1e-12             # decomposition adds up
    assert gap < 0.01                                  # loose sanity bound (< 1 point)
