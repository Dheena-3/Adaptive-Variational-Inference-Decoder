"""The inline OSD-0 candidate table fixes the logical class at some syndromes (decoder-side floor).

bp_engine.sv selects the correction for each syndrome from a hand-written table of one to four candidate
patterns (case (syndrome_lat) ... ). Channel weights only choose AMONG the candidates. Where every
candidate lies in one logical class, no weights can change the logical decision.

This file reads the table from the RTL source itself (so it follows the RTL if it changes) and asserts:
  * the RTL's parity equations equal the repo's H, and every candidate matches its syndrome
  * which syndromes offer a single logical class (0x0, 0x5, 0x9, 0xA, 0xE, 0xF in the current RTL)
  * 0x9 is a single constant 0x101 = qubits {0, 8}, which is logical class 1
  * per world, the best LER ANY selection inside the table can reach (the floor), and which syndromes
    are forced into the less likely class (exact ties are not counted)
  * on the real RTL (osd_mode=1, K=3) over random weight banks, outputs are always table candidates

REGRESSION GUARD: the expected floors describe the RTL as read on 27 Nov 2026. If the table is changed
on purpose (e.g. a class-0 candidate is added at 0x9), these assertions are meant to fail so the report
numbers are revisited. Scope: X-basis table only, d=3, code-capacity, exact probabilities.

Run (repo root):  pytest tests/unit/test_osd_table_floor.py -v -s
"""
import re

import numpy as np
import pytest

try:
    from test_burst_sweeps import BASE, BURSTS, H, LOG, REPO, SYN, joint, rtl  # noqa: F401
except ImportError:
    from tests.unit.test_burst_sweeps import BASE, BURSTS, H, LOG, REPO, SYN, joint, rtl  # noqa: F401

RTL_FILE = REPO / "firmware/rtl/decoder/bp_engine.sv"
SINGLE_CLASS_EXPECTED = {0x0, 0x5, 0x9, 0xA, 0xE, 0xF}
# world -> (syndromes forced into the less likely class, floor in %, tolerance 6e-4)
WORLDS = {
    "uniform": ([0x9], 5.742),
    "Z3 burst": ([0x9, 0xE], 5.859),
    "Z4Z5 burst": ([0x9], 5.740),
}


def parse_table(text):
    m = re.search(r"Symmetric OSD-0 candidate selection.*?case\s*\(\s*syndrome_lat\s*\)(.*?)endcase", text, re.S)
    assert m, "candidate table not found in bp_engine.sv (was the RTL restructured?)"
    body = re.split(r"\bdefault\s*:", m.group(1))[0]
    parts = re.split(r"(?m)^\s*4'h([0-9A-Fa-f]):", body)
    return {int(lab, 16): [int(x, 16) for x in re.findall(r"9'h([0-9A-Fa-f]+)", chunk)]
            for lab, chunk in zip(parts[1::2], parts[2::2])}


def parse_checks(text):
    out = {}
    for i, rhs in re.findall(r"raw_synd_c\[(\d)\]\s*=\s*([^;]+);", text):
        out.setdefault(int(i), {int(q) for q in re.findall(r"raw_corr_c\[(\d+)\]", rhs)})
    return [out[i] for i in range(4)]


@pytest.fixture(scope="module")
def src():
    return RTL_FILE.read_text()


@pytest.fixture(scope="module")
def table(src):
    return parse_table(src)


def theta_for(name, p=0.08):
    t = BASE.copy()
    if name != "uniform":
        t[BURSTS[name]] = p
    return t


def test_table_covers_every_syndrome(table):
    assert sorted(table) == list(range(16))
    assert all(1 <= len(v) <= 4 for v in table.values())


def test_rtl_parity_equations_equal_repo_H(src):
    repo_rows = [set(np.nonzero(H[i])[0].tolist()) for i in range(4)]
    assert parse_checks(src) == repo_rows


def test_every_candidate_matches_its_syndrome(table):
    for s, cands in table.items():
        for c in cands:
            assert int(SYN[c]) == s, f"candidate {c:03x} does not produce syndrome 0x{s:x}"


def test_single_class_syndromes(table):
    single = {s for s, cands in table.items() if len({int(LOG[c]) for c in cands}) == 1}
    assert single == SINGLE_CLASS_EXPECTED


def test_0x9_is_one_constant_in_class_1(table):
    assert table[0x9] == [0x101]
    assert int(LOG[0x101]) == 1


@pytest.mark.parametrize("name", list(WORLDS))
def test_floor_inside_the_table(table, name, capsys):
    forced_expected, floor_expected = WORLDS[name]
    J = joint(theta_for(name))
    classes = {s: {int(LOG[c]) for c in cands} for s, cands in table.items()}
    best = sum(min(J[s, 1 - c] for c in classes[s]) for s in range(16))
    forced = [s for s in range(16)
              if len(classes[s]) == 1 and abs(J[s, 1] - J[s, 0]) > 1e-12
              and int(J[s, 1] > J[s, 0]) not in classes[s]]
    with capsys.disabled():
        print(f"\n=== OSD-0 table floor: {name} ===")
        print(f"Bayes optimum {100 * J.min(axis=1).sum():.3f}%   best any selection inside the table "
              f"{100 * best:.3f}%")
        for s in forced:
            more = int(J[s, 1] > J[s, 0])
            print(f"  0x{s:X}: only class {sorted(classes[s])[0]} offered, class {more} is likelier "
                  f"({J[s, more] / J[s].sum():.0%}); excess {100 * (J[s, more] - J[s, 1 - more]):.4f} pts")
    assert forced == forced_expected
    assert abs(100 * best - floor_expected) < 6e-4


def test_rtl_only_outputs_table_candidates(rtl, table):
    """Random weight banks never make the RTL output anything outside the candidate table."""
    rng = np.random.default_rng(2026)
    seen = {s: set() for s in range(16)}
    for _ in range(300):
        w = [int(x) for x in rng.uniform(32, 2400, 9)]
        for s, c in enumerate(rtl(w, K=3, mode=1)):
            assert (c & 0x1FF) in table[s], f"syndrome 0x{s:x}: output {c & 0x1FF:03x} is not a table candidate"
            seen[s].add(c & 0x1FF)
    assert seen[0x9] == {0x101}                       # the constant is never overridden by the weights
