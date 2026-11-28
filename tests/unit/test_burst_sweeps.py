"""
Burst sweeps for the syndrome-ambiguity study (real RTL, exact counts).

  1. Burst-STRENGTH sweep : LER vs p_burst (0.01 .. 0.2) for a Z3 burst and a Z4Z5 burst.
  2. Burst-LOCATION sweep : burst (p = 0.08) placed on each of the 21 fault mechanisms.

Decoders compared (all run through the real bp_engine.sv, osd_mode=1, K=3):
  static     weights from uniform p=0.01, never updated
  oracle     weights from the true rates
  16-bin     AVID's current estimator: repo L-BFGS fit to P(s)
  32-bin     proposed estimator: fit to P(s, l)   (prototype, scipy L-BFGS-B on the repo's 32-state DP)
  optimum    Bayes-optimal LER of the model

Caveats: exact counts (no sampling noise); the logical-bit labels are simulated.

Run:  pytest tests/unit/test_burst_sweeps.py -v
Plots are written to tests/unit/artifacts/ and the tables always print (no -s needed).
Needs: verilator, g++, numpy, scipy, matplotlib (RTL tests skip without verilator; plots skip without matplotlib).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.optimize import minimize

from simulation.reference.noise_model.joint_dp_likelihood import build_d3_correlated_noise_model
from simulation.reference.noise_model.logical_class_distribution import LogicalClassDistributionEngine

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

# ------------------------------------------------------------------ model
M = build_d3_correlated_noise_model("X")
ENG = LogicalClassDistributionEngine("X")
H = M.H.astype(int)
LOGICAL = M.code.logical_X[0].astype(int)
EDGES = list(M.code.zz_edges)
SUPPORTS = [1 << q for q in range(9)] + [(1 << u) | (1 << v) for u, v in EDGES]
NAMES = [f"Z{q}" for q in range(9)] + [f"Z{u}Z{v}" for u, v in EDGES]

_bits = lambda e: np.array([(e >> q) & 1 for q in range(9)])
_synd = lambda v: sum((int(H[c] @ v) & 1) << c for c in range(4))
SYN = np.array([_synd(_bits(e)) for e in range(512)])
LOG = np.array([int(LOGICAL @ _bits(e)) & 1 for e in range(512)])
MECH_SYN = [int(SYN[m]) for m in SUPPORTS]               # syndrome signature of each mechanism
AMBIGUOUS = {0x1, 0x2, 0x4, 0x8}                          # single faults of both logical bits land here

BASE = np.full(21, 0.01)
LG = lambda p: np.log(p / (1 - p))


def err_dist(theta):
    P = np.zeros(512); P[0] = 1.0; idx = np.arange(512)
    for p, m in zip(theta, SUPPORTS):
        if p > 0:
            P = (1 - p) * P + p * P[idx ^ m]
    return P


def joint(theta):                                         # (16, 2) array P(s, l)
    return ENG.compute_joint_distribution(theta)


def fail_per_syndrome(theta, corr):
    """Probability that the decoder fails, split by the syndrome that occurred."""
    P = err_dist(theta); f = np.zeros(16)
    for e in range(512):
        c = corr[SYN[e]]
        if SYN[e] != _synd(_bits(c)) or LOG[e ^ c]:
            f[SYN[e]] += P[e]
    return f


llrs = lambda th: M.compute_q7_8_llrs(th)[1]


def fit16(theta):                                         # repo estimator, exact 16-bin histogram
    P = err_dist(theta); h = np.zeros(16)
    for e in range(512):
        h[SYN[e]] += P[e] * 256
    return np.array(M.fit_lbfgs(h, prior_lambda=0.5, max_iter=15)["llr_q7_8"])


def fit32(theta, lam=0.5):                                # proposed: exact 32-bin counts P(s, l)
    counts = joint(theta) * 1e4; n = counts.sum()
    def nll(z):
        J = ENG.compute_joint_distribution(1 / (1 + np.exp(-z)))
        return (-(counts * np.log(np.maximum(J, 1e-300))).sum() + lam * np.sum((z - LG(BASE)) ** 2)) / n
    r = minimize(nll, LG(BASE), method="L-BFGS-B", options={"maxiter": 200})
    return llrs(1 / (1 + np.exp(-r.x)))


# ------------------------------------------------------------------ RTL harness
HARNESS = r'''
#include "Vbp_engine.h"
#include "verilated.h"
#include <cstdio>
static Vbp_engine* t;
static void tick(){ t->clk=0; t->eval(); t->clk=1; t->eval(); }
int main(int argc,char**argv){
  Verilated::commandArgs(argc,argv); t=new Vbp_engine; int K,mode,l[9];
  while(scanf("%d %d",&K,&mode)==2){
    for(int i=0;i<9;i++) if(scanf("%d",&l[i])!=1) return 1;
    for(int s=0;s<16;s++){
      t->rst_n=0; t->syndrome_valid=0; t->syndrome=0; t->syndrome_last_in=0;
      for(int i=0;i<9;i++) t->channel_llr[i]=(uint16_t)(int16_t)l[i];
      t->cfg_max_iter=K; t->cfg_osd_mode=mode;
      for(int i=0;i<3;i++) tick();
      t->rst_n=1; tick(); t->syndrome=s; t->syndrome_valid=1; tick(); t->syndrome_valid=0;
      int got=-1; for(int c=0;c<200;c++){ tick(); if(t->correction_valid){ got=t->correction; break; } }
      printf("%03x%c",got&0xfff,s==15?'\n':' ');
    } fflush(stdout);
  }
}
'''


@pytest.fixture(scope="module")
def rtl(tmp_path_factory):
    if not (shutil.which("verilator") and shutil.which("g++")):
        pytest.skip("verilator/g++ not installed (sudo apt-get install -y verilator g++)")
    build = tmp_path_factory.mktemp("rtl_sweeps")
    (build / "harness.cpp").write_text(HARNESS)
    dec = REPO / "firmware/rtl/decoder"
    subprocess.run(
        ["verilator", "--cc", "--exe", "--build", "-O1", "-Wno-fatal", "-Wno-lint", "-Wno-style",
         "-Wno-WIDTH", "--top-module", "bp_engine", str(dec / "check_node_minsum.sv"),
         str(dec / "bp_engine.sv"), "harness.cpp", "-o", "hbp"],
        cwd=build, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc = subprocess.Popen([str(build / "obj_dir/hbp")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def run(llr, K=3, mode=1):
        proc.stdin.write(f"{K} {mode} " + " ".join(str(int(x)) for x in llr) + "\n"); proc.stdin.flush()
        return [int(x, 16) for x in proc.stdout.readline().split()]

    yield run
    proc.stdin.close(); proc.wait(timeout=10)


def ler(theta, corr):
    return fail_per_syndrome(theta, corr).sum()


# ================================================================== sweep 1: strength
STRENGTHS = [0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.20]
BURSTS = {"Z3 burst": 3, "Z4Z5 burst": 9 + EDGES.index((4, 5))}
DECODERS = ["static", "oracle", "16-bin (AVID)", "32-bin (proposed)", "optimum"]


@pytest.fixture(scope="module")
def strength(rtl):
    """res[burst][decoder] -> list of LER (%) per strength."""
    w_static = llrs(BASE); res = {}
    for name, k in BURSTS.items():
        res[name] = {d: [] for d in DECODERS}
        for p in STRENGTHS:
            th = BASE.copy(); th[k] = p
            for d, w in (("static", w_static), ("oracle", llrs(th)), ("16-bin (AVID)", fit16(th)),
                         ("32-bin (proposed)", fit32(th))):
                res[name][d].append(100 * ler(th, rtl(w, 3, 1)))
            res[name]["optimum"].append(100 * joint(th).min(1).sum())
    return res


class TestStrengthSweep:
    def test_z3_burst_16bin_gap_grows_with_burst(self, strength):
        r = strength["Z3 burst"]
        gap = np.array(r["16-bin (AVID)"]) - np.array(r["optimum"])
        i08, i20 = STRENGTHS.index(0.08), STRENGTHS.index(0.20)
        assert all(np.diff(gap[i08:]) > 0), "gap must grow monotonically once the burst is strong"
        assert gap[i08] == pytest.approx(6.17, abs=0.1)        # 11.88 - 5.71
        assert gap[i20] > 2 * gap[i08]                         # about 17 pp at p = 0.2
        assert gap[i20] > 15

    def test_z3_burst_static_already_equals_oracle(self, strength):
        r = strength["Z3 burst"]
        assert np.allclose(r["static"], r["oracle"], atol=0.01), "static assumes singles, which is right for Z3"

    def test_z3_burst_32bin_matches_oracle(self, strength):
        r = strength["Z3 burst"]
        assert np.max(np.abs(np.array(r["32-bin (proposed)"]) - np.array(r["oracle"]))) < 0.3
        i08 = STRENGTHS.index(0.08)
        assert r["32-bin (proposed)"][i08] == pytest.approx(5.99, abs=0.05)

    def test_z4z5_burst_static_gap_grows_and_oracle_fixes_it(self, strength):
        r = strength["Z4Z5 burst"]
        i04 = STRENGTHS.index(0.04)
        gap_static = np.array(r["static"]) - np.array(r["optimum"])
        assert all(np.diff(gap_static[i04:]) > 0)
        assert gap_static[-1] > 15
        # once the pair is strong enough for the weights to represent it, oracle weights are near-optimal
        assert np.all((np.array(r["oracle"]) - np.array(r["optimum"]))[i04:] < 0.4)

    def test_z4z5_burst_16bin_and_32bin_work_at_strong_bursts(self, strength):
        r = strength["Z4Z5 burst"]
        for i in range(STRENGTHS.index(0.08), len(STRENGTHS)):
            assert r["16-bin (AVID)"][i] == pytest.approx(r["oracle"][i], abs=0.1)   # picks the Z4Z5 alias
            assert r["32-bin (proposed)"][i] == pytest.approx(r["oracle"][i], abs=0.1)

    def test_weak_pair_burst_is_missed_by_everyone(self, strength):
        """Finding: at p = 0.02 even oracle weights cannot represent the pair burst."""
        r = strength["Z4Z5 burst"]; i = STRENGTHS.index(0.02)
        assert r["oracle"][i] - r["optimum"][i] > 0.5

    def test_plot_strength_sweep(self, strength, capsys):
        plt = pytest.importorskip("matplotlib.pyplot")
        ARTIFACTS.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
        style = {"static": ("tab:grey", "--"), "oracle": ("tab:blue", "-"), "16-bin (AVID)": ("tab:red", "-"),
                 "32-bin (proposed)": ("tab:green", "-"), "optimum": ("black", ":")}
        for ax, (name, r) in zip(axes, strength.items()):
            for d in DECODERS:
                c, ls = style[d]
                ax.plot(STRENGTHS, r[d], color=c, ls=ls, marker="o", ms=4, lw=2 if d in ("16-bin (AVID)", "32-bin (proposed)") else 1.4, label=d)
            ax.set_title(f"{name} on the real RTL"); ax.set_xlabel("p_burst"); ax.grid(alpha=.3)
        axes[0].set_ylabel("LER (%)"); axes[0].legend(fontsize=9)
        fig.tight_layout(); out = ARTIFACTS / "burst_strength_sweep.png"; fig.savefig(out, dpi=150); plt.close(fig)
        assert out.exists()
        with capsys.disabled():
            print(f"\n\nBurst-strength sweep, LER (%), real RTL osd_mode=1 K=3  ->  {out}")
            for name, r in strength.items():
                print(f"\n{name}\n{'p_burst':<20}" + "".join(f"{p:>8.2f}" for p in STRENGTHS))
                for d in DECODERS:
                    print(f"{d:<20}" + "".join(f"{v:>8.2f}" for v in r[d]))
            print()


# ================================================================== sweep 2: location
P_LOC = 0.08
BIG = 1.0           # percentage points: a "real loss" over the optimum
SMALL = 0.4         # percentage points: within quantization / tie noise


@pytest.fixture(scope="module")
def location(rtl):
    """loc[k][decoder] -> per-syndrome extra failure probability over the optimum (array of 16)."""
    w_static = llrs(BASE); loc = []
    for k in range(21):
        th = BASE.copy(); th[k] = P_LOC
        opt = joint(th).min(1)
        row = {}
        for d, w in (("static", w_static), ("16-bin (AVID)", fit16(th)), ("32-bin (proposed)", fit32(th))):
            row[d] = fail_per_syndrome(th, rtl(w, 3, 1)) - opt
        loc.append(row)
    return loc


LOC_DEC = ["static", "16-bin (AVID)", "32-bin (proposed)"]
SINGLES_AMBIG = [k for k in range(9) if MECH_SYN[k] in AMBIGUOUS]
PAIRS_AMBIG = [k for k in range(9, 21) if MECH_SYN[k] in AMBIGUOUS]
UNAMBIG = [k for k in range(21) if MECH_SYN[k] not in AMBIGUOUS]


class TestLocationSweep:
    def test_location_groups_are_as_expected(self):
        assert [NAMES[k] for k in SINGLES_AMBIG] == ["Z0", "Z2", "Z3", "Z5", "Z6", "Z8"]
        assert [NAMES[k] for k in PAIRS_AMBIG] == ["Z0Z1", "Z1Z2", "Z3Z4", "Z4Z5", "Z6Z7", "Z7Z8"]

    def test_only_ambiguous_syndrome_bursts_cause_big_losses(self, location):
        for k in UNAMBIG:
            for d in LOC_DEC:
                assert 100 * location[k][d].sum() < SMALL, f"{NAMES[k]} / {d}"

    def test_static_loses_on_every_ambiguous_pair_burst(self, location):
        for k in PAIRS_AMBIG:
            assert 100 * location[k]["static"].sum() > 5, NAMES[k]          # about 6.1 pp
        for k in SINGLES_AMBIG:
            assert 100 * location[k]["static"].sum() < SMALL, NAMES[k]

    def test_16bin_loses_on_every_ambiguous_single_burst(self, location):
        """Mirror image: the L-BFGS estimator always assumes the pair explanation."""
        for k in SINGLES_AMBIG:
            assert 100 * location[k]["16-bin (AVID)"].sum() > 5, NAMES[k]   # about 6.2 pp
        for k in PAIRS_AMBIG:
            assert 100 * location[k]["16-bin (AVID)"].sum() < SMALL, NAMES[k]

    def test_32bin_never_loses(self, location):
        for k in range(21):
            assert 100 * location[k]["32-bin (proposed)"].sum() < SMALL, NAMES[k]

    def test_extra_error_sits_at_the_burst_own_syndrome(self, location):
        """About 93% (PDF) of the extra error is at one syndrome; on the RTL it is >= 90%, at the burst's syndrome."""
        for d, ks in (("static", PAIRS_AMBIG), ("16-bin (AVID)", SINGLES_AMBIG)):
            for k in ks:
                ex = location[k][d]; s_top = int(np.argmax(ex))
                assert s_top == MECH_SYN[k] and s_top in AMBIGUOUS, NAMES[k]
                assert ex[s_top] / ex.sum() > 0.9, NAMES[k]

    def test_plot_location_sweep(self, location, capsys):
        plt = pytest.importorskip("matplotlib.pyplot")
        ARTIFACTS.mkdir(exist_ok=True)
        x = np.arange(21); wdt = 0.27
        fig, ax = plt.subplots(figsize=(13, 4.8))
        for i, (d, c) in enumerate(zip(LOC_DEC, ("tab:grey", "tab:red", "tab:green"))):
            ax.bar(x + (i - 1) * wdt, [100 * location[k][d].sum() for k in range(21)], wdt, color=c, label=d)
        labels = [f"{NAMES[k]}\n0x{MECH_SYN[k]:x}" for k in range(21)]
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        for t, k in zip(ax.get_xticklabels(), range(21)):
            if MECH_SYN[k] in AMBIGUOUS: t.set_fontweight("bold")
        ax.set_ylabel("extra LER over optimum (pp)")
        ax.set_title(f"Burst p={P_LOC} on each fault mechanism (bold = ambiguous syndrome 0x1/0x2/0x4/0x8), real RTL")
        ax.grid(axis="y", alpha=.3); ax.legend()
        fig.tight_layout(); out = ARTIFACTS / "burst_location_sweep.png"; fig.savefig(out, dpi=150); plt.close(fig)
        assert out.exists()
        with capsys.disabled():
            print(f"\n\nBurst-location sweep, extra LER over optimum (pp), p={P_LOC}  ->  {out}")
            print(f"{'burst':<8}{'syn':>5}" + "".join(f"{d:>20}" for d in LOC_DEC) + "   top syndrome (share) of the worst decoder")
            for k in range(21):
                tot = {d: 100 * location[k][d].sum() for d in LOC_DEC}; w = max(tot, key=tot.get)
                ex = location[k][w]; s = int(np.argmax(ex))
                top = f"0x{s:x} ({100*ex[s]/max(ex.sum(),1e-12):.0f}%)" if tot[w] > SMALL else "-"
                print(f"{NAMES[k]:<8}{'0x%x' % MECH_SYN[k]:>5}" + "".join(f"{tot[d]:>20.2f}" for d in LOC_DEC) + f"   {top}")
            print()
