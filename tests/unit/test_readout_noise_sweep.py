"""Label-readout noise on the label-budget sweep (ideal-MAP decoder, standalone, no RTL).

A labeled shot gives (s, l).  The logical bit is read out from the w data qubits of the logical
operator, each with flip probability q_d, so the observed bit is wrong with probability
    q_L = (1 - (1 - 2 q_d)^w) / 2          (w = 3 at d=3: q_L ~ 3 q_d).
Two estimators see the SAME noisy counts:
    naive : treats the noisy labels as true labels
    aware : knows q_L and fits  P_obs(s, l') = (1-q_L) P(s, l') + q_L P(s, 1-l')
Grading: exact LER of the MAP decision under the fitted theta, evaluated under the true theta.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).parent))
from surface_oracle import build, joint, ler_of_decision, bayes, map_decision

LAM, PRIOR_REF, W = 0.5, 1e4, 4096
LG = lambda p: np.log(p / (1 - p))


def q_logical(qd, w):
    return (1 - (1 - 2 * qd) ** w) / 2


def fit_hybrid(code, lab, win, q=0.0, lam=LAM, base=0.01):
    n = lab.sum() + win.sum()
    z0 = np.full(len(code["mech"]), LG(base))
    def nll(z):
        J = joint(code, 1 / (1 + np.exp(-z)))
        Jo = (1 - q) * J + q * J[:, ::-1]
        data = -(lab * np.log(np.maximum(Jo, 1e-300))).sum() - (win * np.log(np.maximum(J.sum(1), 1e-300))).sum()
        return (data + lam * (n / PRIOR_REF) * np.sum((z - z0) ** 2)) / n
    r = minimize(nll, z0, method="L-BFGS-B", options={"maxiter": 60})
    return 1 / (1 + np.exp(-r.x))


def run(d, burst, p_burst, Ns, qds, repeats, seed=7):
    code = build(d); k = code["names"].index(burst); m = len(code["mech"])
    theta = np.full(m, 0.01); theta[k] = p_burst
    J = joint(code, theta); opt = 100 * bayes(J)
    w = len(code["logical"])
    flat = (J / J.sum()).ravel(); marg = J.sum(1)
    print(f"d={d} {burst} p={p_burst}  Bayes {opt:.3f}%  W={W}  repeats={repeats}  logical weight w={w}")
    print(f"{'q_d':>6} {'q_L':>7} {'N':>5} | {'naive LER':>10} {'aware LER':>10} | {'naive wrong':>11} {'aware wrong':>11}")
    out = []
    for qd in qds:
        qL = q_logical(qd, w)
        for N in Ns:
            res = {"naive": [], "aware": []}
            for rep in range(repeats):
                rng = np.random.default_rng([seed, N, rep])
                win = rng.multinomial(W, marg / marg.sum()).astype(float)
                lab = rng.multinomial(N, flat).reshape(-1, 2).astype(float)
                noisy = np.zeros_like(lab)                               # flip each labeled shot's bit with prob q_L
                for s in range(lab.shape[0]):
                    for l in (0, 1):
                        c = int(lab[s, l]); f = rng.binomial(c, qL)
                        noisy[s, l] += c - f; noisy[s, 1 - l] += f
                for kind, qq in (("naive", 0.0), ("aware", qL)):
                    th = fit_hybrid(code, noisy, win, q=qq)
                    res[kind].append(100 * ler_of_decision(J, map_decision(joint(code, th))))
            thr = (100 * ler_of_decision(J, map_decision(joint(code, np.full(m, 0.01)))) + opt) / 2
            row = (qd, qL, N, np.mean(res["naive"]), np.mean(res["aware"]),
                   np.mean(np.array(res["naive"]) > thr), np.mean(np.array(res["aware"]) > thr))
            out.append(row)
            print(f"{qd:6.3f} {qL:7.4f} {N:5d} | {row[3]:10.2f} {row[4]:10.2f} | {row[5]:11.0%} {row[6]:11.0%}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=3); ap.add_argument("--burst", default="Z3")
    ap.add_argument("--p", type=float, default=0.08)
    ap.add_argument("--N", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--qd", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.15])
    ap.add_argument("--repeats", type=int, default=20)
    a = ap.parse_args()
    t = time.time(); run(a.d, a.burst, a.p, a.N, a.qd, a.repeats); print(f"{time.time()-t:.0f}s")
