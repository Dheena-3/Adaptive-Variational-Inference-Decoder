"""Cost accounting for the label-budget sweep (closes the 'cost in physical time' item).

Rule used (agreed: TOTAL time, not the difference to a normal shot):
    T_label_shot = T_total_tirtha + T_readout          [us per labeled shot]
    cost(N)      = N * T_label_shot                     [us of machine time taken from normal operation]
    overhead(N)  = N / (N + W)                          [fraction of rounds that are calibration, W = unlabeled window]

T_total_tirtha : the total shot time Tirtha gave (prep + QEC round(s)).
T_readout      : final logical readout (NOT given by Tirtha -> assumption, default 1.0 us, flagged in the output).

Usage (repo root):
    python3 scripts/label_cost.py --t-total-us <Tirtha's number> [--t-readout-us 1.0]
Reads tests/unit/artifacts/label_budget_*.csv, writes label_budget_*_cost.csv next to them.
"""
import argparse, csv, glob, os

ap = argparse.ArgumentParser()
ap.add_argument("--t-total-us", type=float, required=True)
ap.add_argument("--t-readout-us", type=float, default=1.0)
ap.add_argument("--art", default="tests/unit/artifacts")
ap.add_argument("--wrong-max", type=float, default=0.05, help="target: fraction of repeats on the wrong alias")
a = ap.parse_args()
T = a.t_total_us + a.t_readout_us
print(f"T_label_shot = {a.t_total_us} (Tirtha total) + {a.t_readout_us} (readout, ASSUMED) = {T} us")
for path in sorted(glob.glob(os.path.join(a.art, "label_budget_*.csv"))):
    if path.endswith("_cost.csv"):
        continue
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["N"] = int(r["N"]); r["cost_us"] = r["N"] * T
    ok = [r for r in rows if float(r["frac_wrong_alias"]) <= a.wrong_max and r["N"] > 0]
    nstar = ok[0]["N"] if ok else None
    print(f"\n{rows[0]['burst']}")
    print(f"{'N':>6} {'mean LER':>9} {'+-95%':>7} {'wrong':>6} {'overhead':>9} {'cost (ms)':>10}")
    for r in rows:
        print(f"{r['N']:>6} {float(r['mean_ler']):>9.2f} {float(r['ci95']):>7.2f} {float(r['frac_wrong_alias']):>6.0%} "
              f"{float(r['overhead_pct']):>8.1f}% {r['cost_us']/1000:>10.3f}")
    print("smallest N with wrong-alias <= %.0f%% : %s" % (100 * a.wrong_max,
          f"{nstar}  ->  {nstar*T/1000:.3f} ms of calibration" if nstar else "not reached in this grid"))
    out = path.replace(".csv", "_cost.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", out)
