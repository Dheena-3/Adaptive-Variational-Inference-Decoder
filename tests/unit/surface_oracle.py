"""Standalone Bayes oracle for the rotated surface code with correlated ZZ pair faults.

Model (same as the d=3 repo model): Z-type single faults on each data qubit + Z_uZ_v faults on every
nearest-neighbour pair of the d x d grid.  Each mechanism fires independently with its own rate theta_k.
A "shot" = one draw of the mechanisms; the X stabilisers give the syndrome s, the logical-X support gives l.
P(s, l) is computed by a DP over (s, l) states, one XOR-convolution per mechanism.  No RTL needed.
"""
import numpy as np


def build(d):
    n = d * d
    q = lambda r, c: r * d + c
    xs = []                                    # X stabilisers (detect Z faults)
    for i in range(d + 1):
        for j in range(d + 1):
            if (i + j) % 2:                    # odd -> Z-type plaquette, ignore
                continue
            bulk = 1 <= i <= d - 1 and 1 <= j <= d - 1
            top_bottom = i in (0, d) and 1 <= j <= d - 1
            if not (bulk or top_bottom):
                continue
            sup = [q(r, c) for r in (i - 1, i) for c in (j - 1, j) if 0 <= r < d and 0 <= c < d]
            xs.append(sup)
    logical = [q(r, 0) for r in range(d)]      # logical-X support (a column)
    edges = [(q(r, c), q(r, c + 1)) for r in range(d) for c in range(d - 1)] + \
            [(q(r, c), q(r + 1, c)) for r in range(d - 1) for c in range(d)]
    supports = [[a] for a in range(n)] + [list(e) for e in edges]
    names = [f"Z{a}" for a in range(n)] + [f"Z{u}Z{v}" for u, v in edges]
    ns = len(xs)

    def sl(qubits):                            # (syndrome bits, logical bit) caused by a fault on these qubits
        s = 0
        for k, sup in enumerate(xs):
            s |= (sum(x in sup for x in qubits) & 1) << k
        return s, sum(x in logical for x in qubits) & 1
    mech = [sl(sp) for sp in supports]
    return dict(d=d, n=n, ns=ns, xs=xs, logical=logical, supports=supports, names=names, mech=mech)


def joint(code, theta):
    """Exact P(s, l): array (2**ns, 2)."""
    S = 1 << code["ns"]
    P = np.zeros((S, 2)); P[0, 0] = 1.0
    idx = np.arange(S)
    for p, (ms, ml) in zip(theta, code["mech"]):
        if p > 0:
            P = (1 - p) * P + p * P[idx ^ ms][:, [ml, 1 - ml]]
    return P


def ler_of_decision(J, dec):                   # dec[s] in {0,1}: logical class the decoder picks per syndrome
    return float(J[np.arange(len(J)), 1 - dec].sum())


def bayes(J):
    return float(J.min(axis=1).sum())


def map_decision(J):
    return (J[:, 1] > J[:, 0]).astype(int)
