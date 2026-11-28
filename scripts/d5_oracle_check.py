"""Exact P(s, l) and Bayes optimum for the rotated surface code at d=3 and d=5 (Z errors, X checks).

Self-contained (numpy only). d=3 is built the same way as the repo's code and must reproduce its numbers
(checks {0,1} {1,2,4,5} {3,4,6,7} {7,8}; logical class = parity on column 0; optimum 5.708 % at p=0.01).
Mechanisms: every single qubit and every nearest-neighbour pair (d=3: 9+12=21, d=5: 25+40=65).
An EXACT alias = two mechanisms with the same syndrome signature but different logical bit: making either
one hot (same rate, all others equal) gives an identical syndrome histogram but opposite best decisions.

Run:  python3 d5_oracle_check.py            (about a second)
"""
import numpy as np


def build(d):
    idx = lambda r, c: r * d + c
    checks = []
    for i in range(-1, d):
        for j in range(-1, d):
            if (i + j) % 2 == 0:
                continue
            qs = [idx(r, c) for r in (i, i + 1) for c in (j, j + 1) if 0 <= r < d and 0 <= c < d]
            if (0 <= i <= d - 2 and 0 <= j <= d - 2) or (i in (-1, d - 1) and 0 <= j <= d - 2):
                checks.append(qs)
    n = d * d
    edges = [(idx(r, c), idx(r, c + 1)) for r in range(d) for c in range(d - 1)] + \
            [(idx(r, c), idx(r + 1, c)) for r in range(d - 1) for c in range(d)]
    supports = [1 << q for q in range(n)] + [(1 << u) | (1 << v) for u, v in edges]
    names = [f"Z{q}" for q in range(n)] + [f"Z{u}Z{v}" for u, v in edges]
    logical = [idx(r, 0) for r in range(d)]                       # column 0
    nchk = len(checks)

    def sig(m):
        s = 0
        for k, qs in enumerate(checks):
            s |= (sum((m >> q) & 1 for q in qs) & 1) << k
        return s

    lbit = lambda m: sum((m >> q) & 1 for q in logical) & 1
    states = [sig(m) | (lbit(m) << nchk) for m in supports]       # (syndrome, logical) flipped by each mechanism
    return dict(d=d, n=n, checks=checks, nchk=nchk, names=names, states=states)


def joint(code, theta):
    nchk = code["nchk"]
    P = np.zeros(1 << (nchk + 1)); P[0] = 1.0; idx = np.arange(P.size)
    for p, st in zip(theta, code["states"]):
        P = (1 - p) * P + p * P[idx ^ st]
    return P.reshape(2, 1 << nchk).T                               # J[s, l]


def report(d, p=0.01, hot=0.08):
    c = build(d); nm = len(c["states"]); nchk = c["nchk"]
    J = joint(c, np.full(nm, p))
    print(f"\n=== d={d}: {c['n']} qubits, {nchk} checks, {nm} mechanisms, {J.size} (s,l) states ===")
    print(f"total prob {J.sum():.12f}   Bayes optimum at p={p}: {100 * J.min(axis=1).sum():.4f} %")
    groups = {}
    for k, st in enumerate(c["states"]):
        groups.setdefault(st & ((1 << nchk) - 1), []).append((c["names"][k], st >> nchk))
    amb = {s: g for s, g in groups.items() if len({l for _, l in g}) == 2 and s != 0}
    print(f"syndromes with exact aliases (same signature, both logical bits): {len(amb)} -> "
          f"{[hex(s) for s in sorted(amb)]}")
    for s in sorted(amb)[:6]:
        zero = [n for n, l in amb[s] if l == 0]; one = [n for n, l in amb[s] if l == 1]
        a = c["names"].index(zero[0]); b = c["names"].index(one[0])
        thA = np.full(nm, p); thA[a] = hot; thB = np.full(nm, p); thB[b] = hot
        JA, JB = joint(c, thA), joint(c, thB)
        same = np.abs(JA.sum(1) - JB.sum(1)).max()
        decA = (JA[:, 1] > JA[:, 0]).astype(int); decB = (JB[:, 1] > JB[:, 0]).astype(int)
        cost = lambda J, dec: sum(J[s_, 1 - dec[s_]] for s_ in range(J.shape[0]))
        print(f"  0x{s:x}: {zero[0]}(l=0) vs {one[0]}(l=1)  max|dP(s)|={same:.1e}  "
              f"optimum A {100 * JA.min(1).sum():.3f}%  A with B's decisions {100 * cost(JA, decB):.3f}%")
    return c, J


if __name__ == "__main__":
    report(3)
    report(5)
