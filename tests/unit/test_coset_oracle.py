"""Tests for the logical-coset oracle module (simulation/reference/noise_model/coset_oracle.py).

Covers: exact joint P(s, l) vs an independent 2^21 brute force (both bases), the stabilizer-coset
structure (512 patterns -> 32 classes of 16), the same-syndrome catalogue, the opposite-truth
fixture, agreement with the independent validation report's optimal LERs, the committed JSON
fixtures, and the CLI. Pure Python: no Verilator needed.

Run:  pytest tests/unit/test_coset_oracle.py -v
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from simulation.reference.noise_model import coset_oracle as co

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "simulation" / "fixtures" / "coset"


# ------------------------------------------------------------ exactness
class TestExactness:
    @pytest.mark.parametrize("basis", ["X", "Z"])
    @pytest.mark.parametrize("name", ["S1", "S2", "S3", "R1"])
    def test_dp_equals_brute_force_over_2_to_21_subsets(self, basis, name):
        th = co.scenario_theta(name)
        dp = co.oracle_report(th, basis)["joint"]
        bf = co.brute_force_joint(th, basis)
        assert np.abs(dp - bf).max() < 1e-12

    @pytest.mark.parametrize("basis", ["X", "Z"])
    def test_dp_equals_brute_force_on_random_rates(self, basis):
        rng = np.random.default_rng(7)
        for _ in range(3):
            th = rng.uniform(0.0, 0.2, 21)
            assert np.abs(co.oracle_report(th, basis)["joint"] - co.brute_force_joint(th, basis)).max() < 1e-12

    def test_probabilities_are_normalized(self):
        for n in co.SCENARIOS:
            assert co.oracle_report(co.scenario_theta(n))["joint"].sum() == pytest.approx(1.0, abs=1e-12)

    def test_oracle_ler_equals_best_decision_rule(self):
        """LER_oracle = sum_s min(P(s,0), P(s,1)) = LER of the MAP decision rule, and no rule beats it."""
        for n in co.SCENARIOS:
            r = co.oracle_report(co.scenario_theta(n))
            assert co.ler_of_decisions(r["joint"], r["map_bit"]) == pytest.approx(r["ler_oracle"], abs=1e-15)
            rng = np.random.default_rng(1)
            for _ in range(50):
                assert co.ler_of_decisions(r["joint"], rng.integers(0, 2, 16)) >= r["ler_oracle"] - 1e-15


# ------------------------------------------------------------ code structure
class TestStructure:
    @pytest.mark.parametrize("basis", ["X", "Z"])
    def test_512_patterns_form_32_classes_of_16(self, basis):
        cs = co.coset_structure(basis)
        assert cs["stabilizer_group_size"] == 16
        assert cs["n_cosets"] == 32 and cs["coset_sizes"] == [16]
        assert cs["each_coset_constant_s_l"] and cs["bijection"]

    def test_x_basis_ambiguous_syndromes_and_0x4_group(self):
        cat = co.alias_catalogue("X")
        assert cat["ambiguous_syndromes"] == ["0x1", "0x2", "0x4", "0x8"]
        g = cat["groups"]["0x4"]
        assert g["logical_bit_1"] == ["Z3", "Z6"] and g["logical_bit_0"] == ["Z4Z5", "Z7Z8"]

    def test_trivial_syndrome_mechanisms_are_not_flagged_ambiguous(self):
        cat = co.alias_catalogue("X")
        assert cat["groups"]["0x0"]["trivial"] and not cat["groups"]["0x0"]["ambiguous"]

    def test_mechanism_table_has_21_rows_with_consistent_class_ids(self):
        for b in ("X", "Z"):
            rows = co.mechanism_table(b)
            assert len(rows) == 21
            assert all(r["class_id"] == 2 * r["syndrome"] + r["logical_bit"] for r in rows)


# ------------------------------------------------------------ the ambiguity fixture
class TestOppositeTruth:
    def test_same_histogram_opposite_decision_at_0x4(self):
        f = co.opposite_truth_fixture("S2", "S3", 0x4)
        assert f["max_abs_diff_marginal_s"] < 1e-12
        assert (f["map_bit_a"], f["map_bit_b"]) == (0, 1)
        assert f["posterior_l0_a"] == pytest.approx(0.8243, abs=1e-3)
        assert f["posterior_l0_b"] == pytest.approx(0.1757, abs=1e-3)

    def test_optimum_is_the_same_but_wrong_tuning_costs_double(self):
        f = co.opposite_truth_fixture("S2", "S3", 0x4)
        assert 100 * f["ler_oracle_a"] == pytest.approx(5.708, abs=0.005)
        assert 100 * f["ler_oracle_b"] == pytest.approx(5.708, abs=0.005)
        assert 100 * f["ler_a_with_decisions_of_b"] == pytest.approx(12.04, abs=0.01)
        assert 100 * f["ler_b_with_decisions_of_a"] == pytest.approx(12.04, abs=0.01)
        assert 100 * f["share_of_extra_error_at_syndrome"] == pytest.approx(93.2, abs=0.2)

    def test_uniform_model_has_the_eight_exact_ties(self):
        assert co.oracle_report(co.scenario_theta("S1"))["exact_ties"] == [0x1, 0x2, 0x4, 0x7, 0x8, 0xB, 0xD, 0xE]


# ------------------------------------------------------------ cross-check with the independent report
class TestIndependentReport:
    """Optimal LERs reported by the independent validation report (Table 2, mean over windows; values as reported)."""

    @pytest.mark.parametrize("name,reported", [("R0", 2.53), ("R1", 4.05), ("R3", 3.20)])
    def test_oracle_matches_reported_optimal(self, name, reported):
        assert 100 * co.oracle_report(co.scenario_theta(name))["ler_oracle"] == pytest.approx(reported, abs=0.01)


# ------------------------------------------------------------ fixtures + CLI
class TestFixturesAndCli:
    def test_committed_fixtures_match_regenerated(self, tmp_path):
        co.export_fixtures(tmp_path)
        for f in ("mechanisms.json", "alias_catalogue.json", "scenarios.json", "opposite_truth_fixture.json"):
            committed, fresh = json.loads((FIXTURES / f).read_text()), json.loads((tmp_path / f).read_text())
            assert _close(committed, fresh), f"{f} is stale: re-run the export command"

    def test_scenario_fixture_has_no_truth_labels(self):
        """Fixtures hold model probabilities and exact distributions only (no sampled errors, no per-shot labels)."""
        s = json.loads((FIXTURES / "scenarios.json").read_text())
        banned = {"physical_error_mask", "labels", "samples", "shots"}
        for v in s.values():
            assert not banned & set(v)

    def test_cli_prints_oracle_ler_and_passes_checks(self):
        out = subprocess.run([sys.executable, "-m", "simulation.reference.noise_model.coset_oracle", "--scenario", "S2", "--check"],
                             cwd=REPO, capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        assert "5.708 %" in out.stdout and "bijection to (s, l) classes: OK" in out.stdout


def _close(a, b, tol=1e-9):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return abs(a - b) <= tol
    return a == b
