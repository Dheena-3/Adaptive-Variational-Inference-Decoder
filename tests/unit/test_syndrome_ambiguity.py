"""
Syndrome-ambiguity study: Z3 vs Z4Z5 burst on the d=3 rotated surface code.

Checks the claims of "AVID Syndrome ambiguity with different errors" against
  (a) the exact math (32-state joint P(s, l)), and
  (b) the REAL RTL (firmware/rtl/decoder/bp_engine.sv, simulated with Verilator)
      driven by the repo's own Q7.8 LLR quantizer and L-BFGS estimator,
  (c) the logical-aware estimator (fit to labeled (syndrome, logical-bit) counts).

Run (the LER table prints even without -s):
    pytest tests/unit/test_syndrome_ambiguity.py -v
Needs: verilator, g++, numpy.  RTL tests are skipped if verilator is missing.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from simulation.reference.noise_model.joint_dp_likelihood import build_d3_correlated_noise_model
from simulation.reference.noise_model.logical_aware_estimator import build_d3_logical_aware_model

REPO = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------------ model
M = build_d3_correlated_noise_model("X")
LM = build_d3_logical_aware_model("X")             # same 21 mechanisms + logical-bit tables
H = M.H.astype(int)
LOGICAL = M.code.logical_X[0].astype(int)          # support {0, 3, 6}
EDGES = list(M.code.zz_edges)
SUPPORTS = [1 << q for q in range(9)] + [(1 << u) | (1 << v) for u, v in EDGES]

_bits = lambda e: np.array([(e >> q) & 1 for q in range(9)])
_synd = lambda v: sum((int(H[c] @ v) & 1) << c for c in range(4))
SYN = np.array([_synd(_bits(e)) for e in range(512)])
LOG = np.array([int(LOGICAL @ _bits(e)) & 1 for e in range(512)])

# the PDF's three placeholder noise models (21 mechanisms)
S1 = np.full(21, 0.01)
S2 = S1.copy(); S2[9 + EDGES.index((4, 5))] = 0.08   # Z4Z5 burst
S3 = S1.copy(); S3[3] = 0.08                         # Z3 burst
SCENARIOS = {"S1 (uniform)": S1, "S2 (Z4Z5 burst)": S2, "S3 (Z3 burst)": S3}


def err_dist(theta):
    P = np.zeros(512); P[0] = 1.0; idx = np.arange(512)
    for p, m in zip(theta, SUPPORTS):
        if p > 0:
            P = (1 - p) * P + p * P[idx ^ m]
    return P


def joint(theta):
    P = err_dist(theta); J = np.zeros((16, 2))
    for e in range(512):
        J[SYN[e], LOG[e]] += P[e]
    return J


def ler(theta, corr):
    """Exact LER of a decoder given as corr[syndrome] -> 9-bit correction mask."""
    P = err_dist(theta); bad = 0.0
    for e in range(512):
        c = corr[SYN[e]]
        if SYN[e] != _synd(_bits(c)) or LOG[e ^ c]:
            bad += P[e]
    return bad


def llrs(theta):                       # repo's Q7.8 weights from rates (oracle path)
    return M.compute_q7_8_llrs(theta)[1]


def lbfgs_fit(theta, n=256):           # repo's estimator fed the exact histogram
    P = err_dist(theta); h = np.zeros(16)
    for e in range(512):
        h[SYN[e]] += P[e] * n
    return np.array(M.fit_lbfgs(h, prior_lambda=0.5, max_iter=15)["llr_q7_8"])


def labeled_fit(theta, n=256):
    """Logical-aware estimator: fit theta to exact (syndrome, logical-bit) counts n(s, l)
    and return the Q7.8 weights. n only sets the scale (the loss is normalised by N)."""
    return np.array(LM.fit_lbfgs_labeled(joint(theta) * n)["llr_q7_8"])


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


@pytest.fixture(scope="session")
def rtl(tmp_path_factory):
    """Callable rtl(llr, K=3, mode=1) -> 16 corrections from the real bp_engine.sv."""
    if not (shutil.which("verilator") and shutil.which("g++")):
        pytest.skip("verilator/g++ not installed (sudo apt-get install -y verilator g++)")
    build = tmp_path_factory.mktemp("rtl_ambiguity")
    (build / "harness.cpp").write_text(HARNESS)
    dec = REPO / "firmware/rtl/decoder"
    subprocess.run(
        ["verilator", "--cc", "--exe", "--build", "-O1", "-Wno-fatal", "-Wno-lint", "-Wno-style",
         "-Wno-WIDTH", "--top-module", "bp_engine", str(dec / "check_node_minsum.sv"),
         str(dec / "bp_engine.sv"), "harness.cpp", "-o", "hbp"],
        cwd=build, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc = subprocess.Popen([str(build / "obj_dir/hbp")], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True)

    def run(llr, K=3, mode=1):
        proc.stdin.write(f"{K} {mode} " + " ".join(str(int(x)) for x in llr) + "\n")
        proc.stdin.flush()
        return [int(x, 16) for x in proc.stdout.readline().split()]

    yield run
    proc.stdin.close(); proc.wait(timeout=10)


# ================================================================== PART 1: math
class TestPdfMath:
    def test_same_histogram_for_both_bursts(self):
        d = np.abs(joint(S2).sum(1) - joint(S3).sum(1)).max()
        assert d < 1e-12, "Z4Z5 and Z3 bursts must give the same P(s)"

    def test_hidden_logical_odds_at_0x4(self):
        p2 = joint(S2)[4, 0] / joint(S2)[4].sum()
        p3 = joint(S3)[4, 0] / joint(S3)[4].sum()
        assert p2 == pytest.approx(0.8243, abs=1e-3)
        assert p3 == pytest.approx(0.1757, abs=1e-3)

    def test_exact_ties_in_uniform_model(self):
        J = joint(S1)
        ties = [s for s in range(16) if abs(J[s, 0] - J[s, 1]) < 1e-12]
        assert ties == [0x1, 0x2, 0x4, 0x7, 0x8, 0xB, 0xD, 0xE]

    @pytest.mark.parametrize("name", list(SCENARIOS))
    def test_optimal_ler_is_5p71_everywhere(self, name):
        assert 100 * joint(SCENARIOS[name]).min(1).sum() == pytest.approx(5.708, abs=0.005)

    def test_wrong_tuning_costs_12p04_and_93pct_from_0x4(self):
        J2, J3 = joint(S2), joint(S3)
        d2, d3 = J2.argmax(1), J3.argmax(1)
        lerd = lambda J, d: sum(J[s, 1 - d[s]] for s in range(16))
        assert 100 * lerd(J2, d3) == pytest.approx(12.04, abs=0.01)
        assert 100 * lerd(J3, d2) == pytest.approx(12.04, abs=0.01)
        extra = lerd(J2, d3) - J2.min(1).sum()
        share = (J2[4, 1 - d3[4]] - J2[4].min()) / extra
        assert 100 * share == pytest.approx(93, abs=1.0)


# ================================================================== PART 2: real RTL
def _table(rtl):
    weights = {"static": llrs(S1), "oracle Z4Z5": llrs(S2), "oracle Z3": llrs(S3)}
    rows = {}
    for name, th in SCENARIOS.items():
        row = {k: 100 * ler(th, rtl(w, 3, 1)) for k, w in weights.items()}
        row["L-BFGS fit"] = 100 * ler(th, rtl(lbfgs_fit(th), 3, 1))
        row["L-BFGS + l"] = 100 * ler(th, rtl(labeled_fit(th), 3, 1))
        row["Bayes optimum"] = 100 * joint(th).min(1).sum()
        rows[name] = row
    return rows


class TestRealRtl:
    def test_default_mode_ignores_k_and_bp(self, rtl):
        """osd_mode=1 (reset default) output is independent of K and equals osd_mode=2."""
        rng = np.random.default_rng(1)
        for _ in range(30):
            w = rng.integers(64, 2000, 9); base = rtl(w, 3, 1)
            assert all(rtl(w, K, 1) == base for K in range(1, 8))
            assert rtl(w, 3, 2) == base

    def test_static_weights_match_pdf_static_proxy(self, rtl):
        w = llrs(S1)
        got = [100 * ler(th, rtl(w, 3, 1)) for th in (S1, S2, S3)]
        assert got == pytest.approx([5.74, 11.82, 5.99], abs=0.02)

    def test_oracle_weights_fix_their_own_burst_only(self, rtl):
        assert 100 * ler(S2, rtl(llrs(S2), 3, 1)) == pytest.approx(5.86, abs=0.05)
        assert 100 * ler(S3, rtl(llrs(S3), 3, 1)) == pytest.approx(5.99, abs=0.05)
        assert 100 * ler(S3, rtl(llrs(S2), 3, 1)) == pytest.approx(11.94, abs=0.05)

    def test_lbfgs_cannot_tell_the_bursts_apart(self, rtl):
        f2, f3 = lbfgs_fit(S2), lbfgs_fit(S3)
        assert (f2 == f3).all(), "same P(s) -> estimator must return identical weights"

    def test_lbfgs_picks_wrong_alias_for_z3_burst(self, rtl):
        w = lbfgs_fit(S3)
        assert LOG[rtl(w, 3, 1)[0x4]] == 0            # applies l_hat = 0 at syndrome 0x4
        assert 100 * ler(S2, rtl(w, 3, 1)) == pytest.approx(5.92, abs=0.05)   # right by luck
        l3 = 100 * ler(S3, rtl(w, 3, 1))
        assert l3 == pytest.approx(11.88, abs=0.05)                            # wrong here
        assert l3 > 2 * 100 * joint(S3).min(1).sum()                           # >2x optimum

    def test_print_ler_table(self, rtl, capsys):
        """Always shows the LER table (no -s needed)."""
        rows = _table(rtl)
        cols = ["static", "oracle Z4Z5", "oracle Z3", "L-BFGS fit", "L-BFGS + l", "Bayes optimum"]
        with capsys.disabled():
            print("\n\nExact LER (%), real RTL, osd_mode=1, K=3")
            print(f"{'truth':<18}" + "".join(f"{c:>14}" for c in cols))
            for name, row in rows.items():
                print(f"{name:<18}" + "".join(f"{row[c]:>14.2f}" for c in cols))
            print()


# ================================================================== PART 3: logical-aware fit
class TestLogicalAwareEstimator:
    def test_labeled_fit_tells_the_bursts_apart(self):
        """Histogram fit gives identical weights for both bursts; the labeled fit does not."""
        assert (lbfgs_fit(S2) == lbfgs_fit(S3)).all()
        assert not (labeled_fit(S2) == labeled_fit(S3)).all()

    def test_labeled_fit_picks_right_alias_at_0x4(self, rtl):
        assert LOG[rtl(labeled_fit(S2), 3, 1)[0x4]] == 0     # Z4Z5 burst -> l_hat = 0
        assert LOG[rtl(labeled_fit(S3), 3, 1)[0x4]] == 1     # Z3 burst   -> l_hat = 1

    def test_labeled_fit_fixes_z3_burst_on_rtl(self, rtl):
        old = 100 * ler(S3, rtl(lbfgs_fit(S3), 3, 1))        # 11.88
        new = 100 * ler(S3, rtl(labeled_fit(S3), 3, 1))
        assert new < 0.6 * old
        assert new < 100 * joint(S3).min(1).sum() + 1.0      # within 1 point of Bayes

    @pytest.mark.parametrize("name", list(SCENARIOS))
    def test_labeled_fit_within_one_point_of_optimum(self, rtl, name):
        th = SCENARIOS[name]
        got = 100 * ler(th, rtl(labeled_fit(th), 3, 1))
        assert got < 100 * joint(th).min(1).sum() + 1.0

    def test_labeled_fit_vs_oracle_weights(self, rtl, capsys):
        """Informational: how close the labeled-fit LER is to the oracle-weight LER on RTL."""
        with capsys.disabled():
            print("\n\nlabeled fit vs oracle weights (RTL, exact LER %)")
            for name, th in SCENARIOS.items():
                a = 100 * ler(th, rtl(labeled_fit(th), 3, 1))
                b = 100 * ler(th, rtl(llrs(th), 3, 1))
                print(f"{name:<18} labeled {a:6.2f}   oracle {b:6.2f}")