"""Calibration cost accounting from the COMMITTED sweep CSVs (no re-run, nothing overwritten).

A labeled shot = prepare logical state + one QEC round + read out data qubits.
  T_shot = t_round + t_extra        t_round = 1 us (reported by Tirtha, 26 Nov)
                                    t_extra = preparation + readout, NOT given: ASSUMED (default 1 us)
Per window of W rounds (W*t_round of normal operation) the calibration costs N*T_shot, so
  time share = N*T_shot / (N*T_shot + W*t_round).
The CalibrationRecord field stores the WHOLE shot time (calibration_cost_us = T_shot), not the difference.

Run (repo root):  python3 cost_accounting.py [--t-extra-us 1.0] [--t-round-us 1.0] [--window 4096]
Output: tests/unit/artifacts/label_budget_cost_assumed_<burst>.csv  (new files; the committed CSVs are untouched)
"""
import argparse
import csv
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--t-round-us", type=float, default=1.0)
ap.add_argument("--t-extra-us", type=float, default=1.0, help="ASSUMED preparation + readout time per labeled shot")
ap.add_argument("--window", type=int, default=4096)
ap.add_argument("--dir", default="tests/unit/artifacts")
a = ap.parse_args()

cases = {"lower bound (T = round only)": a.t_round_us,
         f"ASSUMED (T = round + {a.t_extra_us:g} us)": a.t_round_us + a.t_extra_us,
         "pessimistic (T = round + 2 x assumed)": a.t_round_us + 2 * a.t_extra_us}
for burst in ("Z4Z5", "Z3"):
    src = Path(a.dir) / f"label_budget_{burst}.csv"
    if not src.exists():
        print(f"missing {src}"); continue
    rows = list(csv.DictReader(open(src)))
    out = []
    print(f"\n=== {burst} burst: calibration cost, W = {a.window} rounds = {a.window * a.t_round_us:g} us ===")
    head = f"{'N':>5} {'LER %':>7} {'+-95%':>6} {'wrong alias':>11} | " + " | ".join(f"{k[:22]:>22}" for k in cases)
    print(head)
    for r in rows:
        n = int(r["N"]); line = f"{n:>5} {float(r['mean_ler']):7.2f} {float(r['ci95']):6.2f} {float(r['frac_wrong_alias']):11.0%} | "
        rec = dict(burst=burst, N=n, mean_ler=r["mean_ler"], ci95=r["ci95"], frac_wrong_alias=r["frac_wrong_alias"])
        cells = []
        for k, T in cases.items():
            cost = n * T; share = 100 * cost / (cost + a.window * a.t_round_us) if n else 0.0
            cells.append(f"{cost:8.0f} us {share:5.1f}%")
            rec[f"cost_us[{k[:5]}]"] = cost; rec[f"time_share_pct[{k[:5]}]"] = round(share, 3)
        out.append(rec); print(line + " | ".join(f"{c:>22}" for c in cells))
    dst = Path(a.dir) / f"label_budget_cost_assumed_{burst}.csv"
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"-> {dst}")
print(f"\nASSUMPTION to state: t_extra = {a.t_extra_us:g} us (prep + readout per labeled shot), to be confirmed; "
      f"t_round = {a.t_round_us:g} us from Tirtha.")
