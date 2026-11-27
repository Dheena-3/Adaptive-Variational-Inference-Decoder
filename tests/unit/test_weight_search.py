"""Can ANY setting of the 9 channel weights reach the Bayes optimum?  (real RTL, osd_mode=1, K=3, exact LER)

The RTL decoder's decisions are a function of the 9 channel LLRs. The oracle weights llrs(theta) give
5.74 / 5.99 / 5.86 % against an optimum of 5.71 %. This searches the 9 weights directly:
  stage 1: random weight vectors (BUDGET/3 evaluations)
  stage 2: hill-climb from the best vector found, with shrinking random steps
The oracle weights are the starting point, so the result can never be worse than the oracle weights.

Reading the result (it is a heuristic search, so it can show reachability but not prove impossibility):
  best LER close to the optimum   -> a better weight set exists; the gap is in how weights are chosen
  best LER stays near the oracle   -> within this search the decoder family does not reach the optimum

Also: a diagnostic for syndrome 0x09 (single most likely error vs most likely class = degeneracy).

Run (repo root):  pytest tests/unit/test_weight_search.py -v -s
Faster:           AVID_SEARCH_BUDGET=300 pytest tests/unit/test_weight_search.py -v -s
"""
import os
import time

import numpy as np
import pytest

try:
    from test_burst_sweeps import BASE, BURSTS, LOG, SYN, err_dist, fail_per_syndrome, joint, llrs, rtl  # noqa: F401
except ImportError:
    from tests.unit.test_burst_sweeps import BASE, BURSTS, LOG, SYN, err_dist, fail_per_syndrome, joint, llrs, rtl  # noqa: F401

BUDGET = int(os.environ.get("AVID_SEARCH_BUDGET", 1500))     # RTL evaluations per case
K_ITER = int(os.environ.get("AVID_K", 3))                     # BP iterations (decoder setting, not QEC rounds)
OSD_MODE = int(os.environ.get("AVID_MODE", 1))                # osd_mode passed to the RTL harness
LO, HI = 32, 2400                                             # search range for each Q7.8 weight
SEED = 4242
CASES = ["uniform", "Z3 burst", "Z4Z5 burst"]
IDX = np.arange(512)


def theta_for(name, p=0.08):
    t = BASE.copy()
    if name != "uniform":
        t[BURSTS[name]] = p
    return t


def search(rtl, theta, budget, rng):
    P = err_dist(theta)
    cache = {}

    def ler_of(w):
        key = tuple(int(x) for x in np.clip(np.round(w), LO, HI))
        if key not in cache:
            corr = np.array([c & 0x1FF for c in rtl(list(key), K=K_ITER, mode=OSD_MODE)])
            c = corr[SYN]
            bad = (SYN != SYN[c]) | (LOG[IDX ^ c] == 1)
            cache[key] = (float(P[bad].sum()), tuple(int(x) for x in corr))
        return cache[key][0], key

    best_l, key0 = ler_of(np.array(llrs(theta), dtype=float))
    best_w, oracle_l = np.array(key0, dtype=float), best_l
    n_random = budget // 3
    for _ in range(n_random):
        l, k = ler_of(rng.uniform(LO, HI, 9))
        if l < best_l:
            best_l, best_w = l, np.array(k, dtype=float)
    n_climb = max(budget - n_random, 1)
    for i in range(n_climb):
        sigma = 200.0 * (1 - i / n_climb) + 5.0
        l, k = ler_of(best_w + rng.normal(0, sigma, 9))
        if l < best_l:
            best_l, best_w = l, np.array(k, dtype=float)
    return oracle_l, best_l, tuple(int(x) for x in best_w), cache


@pytest.mark.parametrize("name", CASES)
def test_weight_search(rtl, name, capsys):
    theta = theta_for(name)
    J = joint(theta)
    opt = J.min(axis=1).sum()
    pmap = (J[:, 1] > J[:, 0]).astype(int)
    rng = np.random.default_rng([SEED, CASES.index(name)])

    t0 = time.time()
    oracle_l, best_l, best_w, cache = search(rtl, theta, BUDGET, rng)
    elapsed = time.time() - t0

    corr_best = cache[best_w][1]
    dec = [int(LOG[c]) for c in corr_best]
    differ = [s for s in range(16) if dec[s] != pmap[s] and J[s].sum() > 1e-4]
    tables = {v[1] for v in cache.values()}
    # the visited decision table that agrees with MAP on the most probability mass
    def mass_disagree(corr):
        return sum(J[s].min() for s in range(16) if int(LOG[corr[s]]) != pmap[s])
    closest = min(tables, key=mass_disagree)
    # per syndrome: did ANY visited table choose the most likely class there?
    never = [s for s in range(16)
             if J[s].sum() > 1e-4 and all(int(LOG[t[s]]) != pmap[s] for t in tables)]

    with capsys.disabled():
        print(f"\n=== weight search: {name} (p=0.08), K={K_ITER}, osd_mode={OSD_MODE}, "
              f"{len(cache)} RTL evaluations, {elapsed:.0f}s ===")
        print(f"oracle weights LER {100 * oracle_l:.3f}%   best found {100 * best_l:.3f}%   Bayes optimum {100 * opt:.3f}%")
        print(f"best weights (Q7.8): {list(best_w)}   oracle weights: {[int(x) for x in llrs(theta)]}")
        print(f"distinct decision tables reached: {len(tables)}")
        print(f"syndromes where the best table still differs from MAP: {[hex(s) for s in differ]}")
        print(f"syndromes where NO visited table ever chose the MAP class: {[hex(s) for s in never]}")
        print(f"most MAP-like table reached: LER {100 * fail_per_syndrome(theta, list(closest)).sum():.3f}%")

    assert best_l <= oracle_l + 1e-12                         # oracle weights are in the search set
    assert best_l >= opt - 1e-9                               # nothing beats the Bayes optimum
    check = fail_per_syndrome(theta, [c & 0x1FF for c in rtl(llrs(theta), K=K_ITER, mode=OSD_MODE)]).sum()
    assert abs(oracle_l - check) < 1e-12


def test_syndrome_0x09_degeneracy(capsys):
    """Is the decoder's wrong class at 0x09 a case of 'most likely error != most likely class'?"""
    theta = BASE.copy()
    P = err_dist(theta)
    J = joint(theta)
    sel = IDX[SYN == 0x09]
    tot = [P[sel[LOG[sel] == l]].sum() for l in (0, 1)]
    assert abs(tot[0] - J[9, 0]) < 1e-9 and abs(tot[1] - J[9, 1]) < 1e-9
    top = sel[np.argsort(-P[sel])][:6]
    with capsys.disabled():
        print("\n=== syndrome 0x09, uniform noise (p=0.01 everywhere) ===")
        print(f"class 0 total {tot[0]:.6f}   class 1 total {tot[1]:.6f}   (P(s) = {sum(tot):.6f})")
        print(f"{'pattern (qubits)':>18} {'class':>6} {'P':>10} {'share of class':>15}")
        for e in top:
            qs = [q for q in range(9) if (e >> q) & 1]
            l = int(LOG[e])
            print(f"{str(qs):>18} {l:>6} {P[e]:10.6f} {P[e] / tot[l]:15.1%}")
        l_top = int(LOG[top[0]])
        l_map = int(tot[1] > tot[0])
        print(f"most likely single pattern is in class {l_top}; most likely class is {l_map}"
              + ("  -> degeneracy" if l_top != l_map else "  -> not degeneracy"))
