"""Logical-coset oracle: CLI + importable API (d=3 rotated surface code).

Wraps the repo's exact 32-state engine (`LogicalClassDistributionEngine`) and adds
  * named noise scenarios (S1/S2/S3 from the ambiguity study, R0/R1/R3 from the
    independent validation report),
  * an independent brute-force check over all 2^21 fault subsets,
  * a stabilizer-coset structure check (512 patterns -> 32 classes of 16),
  * the same-syndrome catalogue (which syndromes are logically ambiguous),
  * the opposite-truth fixture (same P(s), opposite logical decision),
  * JSON fixture export.

Leakage rule: every function takes only the 21 fault probabilities theta (the
*model*), never sampled errors or logical labels. It is an offline oracle/scorer.

Run from the repo root:
    python -m simulation.reference.noise_model.coset_oracle --scenario S2
    python -m simulation.reference.noise_model.coset_oracle --scenario S2 --check
    python -m simulation.reference.noise_model.coset_oracle --export-fixtures simulation/fixtures/coset
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from simulation.reference.noise_model.logical_class_distribution import LogicalClassDistributionEngine
from simulation.stimulus.noise_framework.config_schema import CANONICAL_12_PAIRS

PAIRS = [tuple(p) for p in CANONICAL_12_PAIRS]
TIE_TOL = 1e-12
ROUND = 12                                   # decimals written to JSON


# ----------------------------------------------------------------- scenarios
def make_theta(single: float = 0.0, pair: float = 0.0, singles: Dict[int, float] | None = None,
               pairs: Dict[tuple, float] | None = None) -> np.ndarray:
    """21-vector: 9 single-qubit rates then the 12 canonical pair rates."""
    t = np.zeros(21)
    t[:9] = single
    t[9:] = pair
    for q, v in (singles or {}).items():
        t[q] = v
    for e, v in (pairs or {}).items():
        t[9 + PAIRS.index(tuple(e))] = v
    return t


SCENARIOS: Dict[str, Dict[str, Any]] = {
    "S1": dict(desc="all 21 mechanisms p=0.01 (ambiguity study, placeholder rates)",
               theta=lambda: make_theta(0.01, 0.01)),
    "S2": dict(desc="p=0.01 everywhere except Z4Z5 burst p=0.08 (ambiguity study)",
               theta=lambda: make_theta(0.01, 0.01, pairs={(4, 5): 0.08})),
    "S3": dict(desc="p=0.01 everywhere except Z3 burst p=0.08 (ambiguity study)",
               theta=lambda: make_theta(0.01, 0.01, singles={3: 0.08})),
    "R0": dict(desc="stationary baseline: singles 0.012, pairs 0.004 (validation report)",
               theta=lambda: make_theta(0.012, 0.004)),
    "R1": dict(desc="hot pair: R0 + q4=0.055 + Z4Z5=0.075 (validation report)",
               theta=lambda: make_theta(0.012, 0.004, singles={4: 0.055}, pairs={(4, 5): 0.075})),
    "R3": dict(desc="hot single (alias swap): R0 + q4=0.055 + Z3=0.075 (validation report)",
               theta=lambda: make_theta(0.012, 0.004, singles={4: 0.055, 3: 0.075})),
}


def scenario_theta(name: str) -> np.ndarray:
    return SCENARIOS[name]["theta"]()


# ----------------------------------------------------------------- basics
def _pauli(basis: str) -> str:
    return "Z" if basis.upper() == "X" else "X"          # X-checks detect Z errors, and vice versa


def mechanism_names(basis: str = "X") -> List[str]:
    p = _pauli(basis)
    return [f"{p}{q}" for q in range(9)] + [f"{p}{u}{p}{v}" for u, v in PAIRS]


def mechanism_table(basis: str = "X") -> List[Dict[str, Any]]:
    eng = LogicalClassDistributionEngine(basis)
    names = mechanism_names(basis)
    rows = []
    for m in range(21):
        s, l = int(eng.syndrome_signatures[m]), int(eng.logical_flips[m])
        rows.append(dict(index=m, name=names[m], support=[int(q) for q in np.flatnonzero(eng.mechanism_supports[m])],
                         syndrome=s, syndrome_hex=f"0x{s:x}", logical_bit=l, class_id=2 * s + l))
    return rows


def alias_catalogue(basis: str = "X") -> Dict[str, Any]:
    """Syndromes where mechanisms with DIFFERENT logical bits collide (logical ambiguity)."""
    by_s: Dict[int, Dict[int, List[str]]] = {}
    for r in mechanism_table(basis):
        by_s.setdefault(r["syndrome"], {}).setdefault(r["logical_bit"], []).append(r["name"])
    groups = {}
    for s in sorted(by_s):
        both = len(by_s[s]) == 2 and s != 0
        groups[f"0x{s:x}"] = dict(ambiguous=both, trivial=(s == 0),
                                  logical_bit_0=sorted(by_s[s].get(0, [])), logical_bit_1=sorted(by_s[s].get(1, [])))
    return dict(basis=basis.upper(), ambiguous_syndromes=[k for k, v in groups.items() if v["ambiguous"]], groups=groups)


# ----------------------------------------------------------------- the oracle
def oracle_report(theta: np.ndarray, basis: str = "X") -> Dict[str, Any]:
    eng = LogicalClassDistributionEngine(basis)
    r = eng.compute_posterior_and_oracle_ler(np.asarray(theta, dtype=float))
    J = r["joint_s_l"]
    ties = [s for s in range(16) if abs(J[s, 0] - J[s, 1]) < TIE_TOL]
    return dict(basis=basis.upper(), theta=[float(x) for x in theta], joint=J, marginal=r["marginal_s"],
                posterior_l1=J[:, 1] / np.maximum(J.sum(1), 1e-300), map_bit=[int(b) for b in r["optimal_logical_decision"]],
                irreducible=J.min(1), ler_oracle=float(r["ler_oracle"]), exact_ties=ties)


def ler_of_decisions(joint: np.ndarray, decisions) -> float:
    """LER(d) = sum_s P(s, 1 - d(s))."""
    return float(sum(joint[s, 1 - int(decisions[s])] for s in range(16)))


# ----------------------------------------------------------------- independent checks
def _luts(eng: LogicalClassDistributionEngine):
    bits = ((np.arange(512)[:, None] >> np.arange(9)[None, :]) & 1).astype(np.uint8)
    syn = np.zeros(512, dtype=np.int64)
    for c in range(4):
        syn |= (((bits @ eng.H[c].astype(np.int64)) & 1) << c)
    log = (bits @ eng.logical_vec.astype(np.int64)) & 1
    return bits, syn, log


def brute_force_joint(theta: np.ndarray, basis: str = "X") -> np.ndarray:
    """Enumerate all 2^21 subsets of fired mechanisms; syndrome/logical come from H directly."""
    eng = LogicalClassDistributionEngine(basis)
    _, syn_lut, log_lut = _luts(eng)
    masks = [int(sum(int(eng.mechanism_supports[m, q]) << q for q in range(9))) for m in range(21)]
    xor = np.zeros(1, dtype=np.int32)
    prob = np.ones(1)
    for m in range(21):
        p = float(theta[m])
        xor = np.concatenate([xor, xor ^ masks[m]])
        prob = np.concatenate([prob * (1.0 - p), prob * p])
    idx = 2 * syn_lut[xor] + log_lut[xor]
    return np.bincount(idx, weights=prob, minlength=32).reshape(16, 2)


def coset_structure(basis: str = "X") -> Dict[str, Any]:
    """512 error patterns -> cosets of the opposite-type stabilizer group; each coset is one (s, l) class."""
    eng = LogicalClassDistributionEngine(basis)
    _, syn_lut, log_lut = _luts(eng)
    other = eng.code.hz if basis.upper() == "X" else eng.code.hx        # stabilizers that act trivially on the logical
    rows = [int(sum(int(b) << q for q, b in enumerate(r))) for r in np.asarray(other)]
    group = {0}
    for k in range(1, 5):
        for combo in itertools.combinations(rows, k):
            v = 0
            for r in combo:
                v ^= r
            group.add(v)
    group_list = sorted(group)
    cosets = {}
    for e in range(512):
        rep = min(e ^ g for g in group_list)
        cosets.setdefault(rep, []).append(e)
    const = all(len({(int(syn_lut[e]), int(log_lut[e])) for e in c}) == 1 for c in cosets.values())
    classes = {(int(syn_lut[c[0]]), int(log_lut[c[0]])) for c in cosets.values()}
    return dict(stabilizer_group_size=len(group_list), n_cosets=len(cosets),
                coset_sizes=sorted({len(c) for c in cosets.values()}), each_coset_constant_s_l=const,
                n_distinct_classes=len(classes), bijection=(len(classes) == 32 == len(cosets) and const))


# ----------------------------------------------------------------- fixtures
def _jsonable(x):
    if isinstance(x, np.ndarray):
        return np.round(x, ROUND).tolist()
    if isinstance(x, (np.floating, float)):
        return round(float(x), ROUND)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def scenario_fixture(name: str, basis: str = "X") -> Dict[str, Any]:
    th = scenario_theta(name)
    rep = oracle_report(th, basis)
    return dict(name=name, description=SCENARIOS[name]["desc"], basis=basis, theta=th, mechanisms=mechanism_names(basis),
                joint_s_l=rep["joint"], marginal_s=rep["marginal"], posterior_l1_given_s=rep["posterior_l1"],
                map_bit=rep["map_bit"], irreducible_error=rep["irreducible"], exact_ties=[f"0x{s:x}" for s in rep["exact_ties"]],
                ler_oracle=rep["ler_oracle"])


def opposite_truth_fixture(name_a: str = "S2", name_b: str = "S3", syndrome: int = 0x4, basis: str = "X") -> Dict[str, Any]:
    """Two worlds with the SAME P(s) but opposite logical decision at `syndrome`."""
    ra, rb = oracle_report(scenario_theta(name_a), basis), oracle_report(scenario_theta(name_b), basis)
    da, db = ra["map_bit"], rb["map_bit"]
    return dict(
        world_a=name_a, world_b=name_b, syndrome=f"0x{syndrome:x}", basis=basis,
        max_abs_diff_marginal_s=float(np.abs(ra["marginal"] - rb["marginal"]).max()),
        posterior_l0_a=float(1 - ra["posterior_l1"][syndrome]), posterior_l0_b=float(1 - rb["posterior_l1"][syndrome]),
        map_bit_a=da[syndrome], map_bit_b=db[syndrome], decisions_a=da, decisions_b=db,
        ler_oracle_a=ra["ler_oracle"], ler_oracle_b=rb["ler_oracle"],
        ler_a_with_decisions_of_b=ler_of_decisions(ra["joint"], db), ler_b_with_decisions_of_a=ler_of_decisions(rb["joint"], da),
        share_of_extra_error_at_syndrome=float(
            (ra["joint"][syndrome, 1 - db[syndrome]] - ra["joint"][syndrome].min())
            / (ler_of_decisions(ra["joint"], db) - ra["ler_oracle"])),
        description="Identical syndrome histogram, opposite optimal logical decision: a histogram-only estimator cannot choose.")


def export_fixtures(outdir: str | Path) -> List[str]:
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    files = {
        "mechanisms.json": {b: mechanism_table(b) for b in ("X", "Z")},
        "alias_catalogue.json": {b: alias_catalogue(b) for b in ("X", "Z")},
        "scenarios.json": {n: scenario_fixture(n) for n in SCENARIOS},
        "opposite_truth_fixture.json": opposite_truth_fixture(),
    }
    for fn, obj in files.items():
        (out / fn).write_text(json.dumps(_jsonable(obj), indent=1, sort_keys=True) + "\n")
    return sorted(files)


# ----------------------------------------------------------------- CLI
def _print_report(name: str, basis: str, rep: Dict[str, Any]) -> None:
    pauli = _pauli(basis)
    print(f"Scenario {name}: {SCENARIOS[name]['desc']}" if name in SCENARIOS else f"Custom scenario")
    print(f"basis {basis}: {pauli}-errors detected by {basis}-checks")
    print(f"{'synd':>5} {'P(s)':>10} {'P(l=0|s)':>9} {'P(l=1|s)':>9} {'MAP':>4} {'irreducible':>12}  note")
    for s in range(16):
        note = "exact tie" if s in rep["exact_ties"] else ""
        print(f"0x{s:02x} {rep['marginal'][s]:>10.6f} {1 - rep['posterior_l1'][s]:>9.4f} {rep['posterior_l1'][s]:>9.4f} "
              f"{rep['map_bit'][s]:>4d} {rep['irreducible'][s]:>12.3e}  {note}")
    print(f"\nBayes-optimal logical error rate (LER_oracle) = {100 * rep['ler_oracle']:.3f} %")
    print("exact ties:", [f"0x{s:x}" for s in rep["exact_ties"]])
    print("logically ambiguous syndromes:", alias_catalogue(basis)["ambiguous_syndromes"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="S2", choices=sorted(SCENARIOS))
    ap.add_argument("--basis", default="X", choices=["X", "Z"])
    ap.add_argument("--check", action="store_true", help="run the independent 2^21 brute-force and coset-structure checks")
    ap.add_argument("--json", metavar="PATH", help="write the full report for this scenario as JSON")
    ap.add_argument("--export-fixtures", metavar="DIR", help="write all fixtures (mechanisms, aliases, scenarios, opposite-truth)")
    a = ap.parse_args(argv)

    th = scenario_theta(a.scenario)
    rep = oracle_report(th, a.basis)
    _print_report(a.scenario, a.basis, rep)
    rc = 0
    if a.check:
        bf = brute_force_joint(th, a.basis)
        d = float(np.abs(bf - rep["joint"]).max())
        cs = coset_structure(a.basis)
        print(f"\n[check] DP vs brute force over 2^21 subsets: max |diff| = {d:.2e}  {'OK' if d < 1e-12 else 'FAIL'}")
        print(f"[check] stabilizer group size {cs['stabilizer_group_size']}, {cs['n_cosets']} cosets of sizes {cs['coset_sizes']}, "
              f"bijection to (s, l) classes: {'OK' if cs['bijection'] else 'FAIL'}")
        rc = 0 if (d < 1e-12 and cs["bijection"]) else 1
    if a.json:
        Path(a.json).write_text(json.dumps(_jsonable(scenario_fixture(a.scenario, a.basis)), indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {a.json}")
    if a.export_fixtures:
        print("\nwrote fixtures to", a.export_fixtures, ":", ", ".join(export_fixtures(a.export_fixtures)))
    return rc


if __name__ == "__main__":
    sys.exit(main())
