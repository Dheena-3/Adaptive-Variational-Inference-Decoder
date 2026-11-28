"""d=3 -> d=5 oracle check (Fri 27).  Standalone, no RTL.  Run: python3 scripts/d5_oracle_check.py
1) reproduces the d=3 repo numbers (Bayes 5.708 %, P(0x2)=0.0909, 0.8243, ~16.7 pp static gap at p=0.2)
2) brute-force check of the DP on a d=5 subset (18 active mechanisms -> 2^18 fault subsets, exact)
3) d=5 burst sweep: worst loss of a decoder that assumes uniform noise (MAP under base), per burst strength
Caveat: ideal-MAP decisions, not the RTL decoder (bp_engine is 9-qubit / 4-bit syndrome only)."""
import itertools, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from surface_oracle import build, joint, bayes, map_decision, ler_of_decision

c3 = build(3); base3 = np.full(21, 0.01); J = joint(c3, base3)
assert abs(100 * bayes(J) - 5.7079) < 1e-3
th = base3.copy(); th[c3["names"].index("Z4Z5")] = 0.08; J = joint(c3, th)
assert abs(J[2].sum() - 0.090946) < 1e-6 and abs(J[2, 0] / J[2].sum() - 0.8243) < 1e-4
print("d=3 reproduces repo numbers: OK")

c5 = build(5); rng = np.random.default_rng(0)
act = rng.choice(len(c5["mech"]), 18, replace=False); th5 = np.zeros(len(c5["mech"]))
th5[act] = rng.uniform(0.01, 0.2, 18); Jd = joint(c5, th5)
Jb = np.zeros_like(Jd)
for bits in itertools.product((0, 1), repeat=18):
    pr, s, l = 1.0, 0, 0
    for b, k in zip(bits, act):
        pr *= th5[k] if b else 1 - th5[k]
        if b: s ^= c5["mech"][k][0]; l ^= c5["mech"][k][1]
    Jb[s, l] += pr
print(f"d=5 DP vs brute force over 2^18 subsets: max diff {np.abs(Jd - Jb).max():.2e}")
assert np.abs(Jd - Jb).max() < 1e-12

for d in (3, 5):
    c = build(d); m = len(c["mech"]); base = np.full(m, 0.01); dec0 = map_decision(joint(c, base))
    print(f"\nd={d}: Bayes(uniform) {100*bayes(joint(c, base)):.3f}%")
    for p in (0.08, 0.12, 0.2, 0.3):
        g = []
        for k in range(m):
            t = base.copy(); t[k] = p; Jk = joint(c, t); g.append((100 * (ler_of_decision(Jk, dec0) - bayes(Jk)), c["names"][k]))
        print(f"  p_burst={p:.2f}: worst loss {max(g)[0]:.2f} pp at {max(g)[1]}")
