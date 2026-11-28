"""Benchmark: histogram-only L-BFGS vs logical-aware (s, l) L-BFGS, scored by the bit-exact BP engine.

For each truth (S1 uniform, S2 Z4Z5 burst, S3 Z3 burst) and shot budget N, simulate N calibration
shots, fit theta two ways, convert to Q7.8 weights, and compute the *exact* LER of the BP decoder
(K=3, OSD-0) under the true noise:  LER = sum_s P_true(s, 1 - d(s)),  d(s) = logical parity of the
BP correction at syndrome s.  Also reports the estimator-implied MAP decision LER (Bayes if theta_hat
is right) and the Bayes optimum.

Usage:  PYTHONPATH=. python3 scripts/benchmark_logical_aware_estimator.py [--seeds 20]
Writes: results/logical_aware_benchmark.json and .csv
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from simulation.reference.noise_model import build_d3_logical_aware_model
from simulation.reference.syndrome_stream.causal_stream_runner import CycleAccurateBPEngine

BURST_P, BASE_P = 0.08, 0.01


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--shots", type=int, nargs="+", default=[1000, 10000, 100000])
    a = ap.parse_args()

    m = build_d3_logical_aware_model()
    eng = CycleAccurateBPEngine(code=m.code, basis="X", default_K=3)
    names = [e.name for e in m.events]

    def theta(**kw):
        t = np.full(21, BASE_P)
        for k, v in kw.items():
            t[names.index(k)] = v
        return t

    truths = {"S1 uniform": theta(), "S2 Z4Z5 burst": theta(ZZ_4_5=BURST_P), "S3 Z3 burst": theta(Z_3=BURST_P)}

    def bp_ler(q78, truth):
        d = np.array([int(np.dot(eng.logical_op, eng.decode(s, q78, K=3).correction_bits) & 1) for s in range(16)])
        return m.decision_ler(d, truth)

    rows = []
    for tname, tt in truths.items():
        bayes = m.bayes_optimal_ler(tt)
        oracle = bp_ler(m.compute_q7_8_llrs(tt)[1], tt)
        static = bp_ler(m.compute_q7_8_llrs(np.full(21, BASE_P))[1], tt)
        for n in a.shots:
            old_l, new_l, new_map, new_nolabel_l = [], [], [], []
            for seed in range(a.seeds):
                rng = np.random.default_rng(1000 * seed + n % 997)
                s, l = m.sample_shots(tt, n, rng)
                hist = np.bincount(s, minlength=16).astype(float)
                csl = np.bincount(s * 2 + l, minlength=32).reshape(16, 2).astype(float)
                old = m.fit_lbfgs(hist, prior_lambda=0.05, max_iter=80)
                new = m.fit_lbfgs_labeled(csl, max_iter=300)
                old_l.append(bp_ler(old["llr_q7_8"], tt))
                new_l.append(bp_ler(new["llr_q7_8"], tt))
                new_map.append(m.decision_ler(new["map_logical_decision"], tt))
            rows.append({
                "truth": tname, "shots": n, "seeds": a.seeds,
                "bayes_optimum": bayes, "oracle_weights_bp": oracle, "static_weights_bp": static,
                "old_fit_bp_mean": float(np.mean(old_l)), "old_fit_bp_worst": float(np.max(old_l)),
                "labeled_fit_bp_mean": float(np.mean(new_l)), "labeled_fit_bp_worst": float(np.max(new_l)),
                "labeled_map_decision_mean": float(np.mean(new_map)),
            })
            r = rows[-1]
            print(f"{tname:14s} N={n:7d} | static {100*static:5.2f} oracle {100*oracle:5.2f} | "
                  f"old fit {100*r['old_fit_bp_mean']:5.2f} (worst {100*r['old_fit_bp_worst']:5.2f}) | "
                  f"labeled {100*r['labeled_fit_bp_mean']:5.2f} (worst {100*r['labeled_fit_bp_worst']:5.2f}) | "
                  f"MAP {100*r['labeled_map_decision_mean']:5.2f} | bayes {100*bayes:5.2f}", flush=True)
    out = Path("results"); out.mkdir(exist_ok=True)
    (out / "logical_aware_benchmark.json").write_text(json.dumps(rows, indent=2))
    with (out / "logical_aware_benchmark.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
