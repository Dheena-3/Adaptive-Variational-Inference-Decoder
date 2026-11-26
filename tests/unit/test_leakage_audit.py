"""Leakage audit (plan Tue 24, Gate G1): the estimator may see only counts.

What is audited: fit_hybrid(labeled32, window16) and fit_repo16(window16) from test_label_budget.py.
The 'truth side' is everything the simulator knows: the true rates theta, the burst location,
the exact joint P(s, l), per-shot error patterns, and the RTL grader.

Checks
  1. signature   : the estimators take only count arguments (no theta / truth / seed / rng).
  2. static code : the estimators' code (including nested functions) never names a truth-side
                   function or variable (joint, theta_burst, draw_*, BURSTS, P_BURST, ler, rtl, ...).
  3. invariance  : same counts -> bit-identical output, even after the truth-side configuration
                   in the sweep module is altered (burst moved, strength changed).
  4. control     : the audit can detect dependence: swapping the l columns of the labeled counts
                   DOES change the output (otherwise checks 3 would pass for the wrong reason).
  5. seeds       : labeled-shot and window streams are independent and reproducible.

Not covered here (belongs to Tirtha's pipeline): the slow-loop timing, i.e. that a weight commit
uses only rounds that precede it. This file says nothing about that.

Run (repo root):  pytest tests/unit/test_leakage_audit.py -v      (no Verilator needed)
"""
import inspect

import numpy as np
import pytest

try:
    import test_label_budget as TLB
except ImportError:
    from tests.unit import test_label_budget as TLB

FORBIDDEN = {
    "theta", "theta_o", "theta_burst", "joint", "err_dist", "draw_labeled", "draw_window",
    "BURSTS", "P_BURST", "ler", "rtl", "fail_per_syndrome", "SYN", "LOG", "SUPPORTS",
    "truth", "rng", "seed", "SEED_LABELS", "SEED_WINDOW",
}
ESTIMATORS = {"fit_hybrid": ["labeled32", "window16", "lam"], "fit_repo16": ["window16"]}
Z4Z5 = "Z4Z5 burst"


def code_names(fn):
    """Every name used by fn, including inside nested functions and comprehensions."""
    names, stack = set(), [fn.__code__]
    while stack:
        c = stack.pop()
        names |= set(c.co_names) | set(c.co_varnames) | set(c.co_freevars)
        stack += [k for k in c.co_consts if hasattr(k, "co_names")]
    return names


def counts(name=Z4Z5, n=512, rep=0):
    k = TLB.BURSTS[name]
    theta = TLB.theta_burst(k)
    lab = TLB.draw_labeled(theta, n, np.random.default_rng([TLB.SEED_LABELS, k, n, rep]))
    win = TLB.draw_window(theta, TLB.W, np.random.default_rng([TLB.SEED_WINDOW, k, rep]))
    return lab, win


@pytest.mark.parametrize("fn_name, params", ESTIMATORS.items())
def test_signature_has_only_count_arguments(fn_name, params):
    sig = inspect.signature(getattr(TLB, fn_name))
    assert list(sig.parameters) == params
    assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in sig.parameters.values())  # no *args / **kwargs


@pytest.mark.parametrize("fn_name", ESTIMATORS)
def test_estimator_code_never_names_truth_side_objects(fn_name):
    used = code_names(getattr(TLB, fn_name))
    assert not (used & FORBIDDEN), f"{fn_name} references truth-side names: {sorted(used & FORBIDDEN)}"


def test_same_counts_give_identical_output():
    lab, win = counts()
    a = TLB.fit_hybrid(lab, win)
    b = TLB.fit_hybrid(lab.copy(), win.copy())
    assert np.array_equal(a, b)


def test_output_unchanged_when_truth_configuration_is_altered(monkeypatch):
    lab, win = counts()
    before = TLB.fit_hybrid(lab, win)
    before16 = TLB.fit_repo16(win)
    # alter everything on the truth side of the sweep module
    monkeypatch.setattr(TLB, "P_BURST", 0.20)
    monkeypatch.setattr(TLB, "BURSTS", {"Z3 burst": 9, "Z4Z5 burst": 3})   # burst locations swapped
    monkeypatch.setattr(TLB, "theta_burst", lambda k: np.full(21, 0.5))
    after = TLB.fit_hybrid(lab, win)
    after16 = TLB.fit_repo16(win)
    assert np.array_equal(before, after)
    assert np.array_equal(before16, after16)


def test_control_audit_detects_dependence_on_the_labels():
    """Positive control: the label columns ARE an input, so swapping them must change the answer."""
    lab, win = counts()
    out = TLB.fit_hybrid(lab, win)
    swapped = TLB.fit_hybrid(lab[:, ::-1].copy(), win)
    assert not np.array_equal(out, swapped)


def test_labeled_and_window_streams_are_independent_and_reproducible():
    k = TLB.BURSTS[Z4Z5]
    a1 = np.random.default_rng([TLB.SEED_LABELS, k, 64, 0]).random(1000)
    a2 = np.random.default_rng([TLB.SEED_LABELS, k, 64, 0]).random(1000)
    b = np.random.default_rng([TLB.SEED_WINDOW, k, 0]).random(1000)
    assert np.array_equal(a1, a2)                      # reproducible
    assert not np.array_equal(a1, b)                   # different streams
    assert abs(np.corrcoef(a1, b)[0, 1]) < 0.1         # and uncorrelated
    assert TLB.SEED_LABELS != TLB.SEED_WINDOW


def test_the_window_is_shared_across_n_within_a_repeat():
    """Design check: for a given repeat the unlabeled window does not depend on N (paired design)."""
    theta = TLB.theta_burst(TLB.BURSTS[Z4Z5])
    k = TLB.BURSTS[Z4Z5]
    w1 = TLB.draw_window(theta, TLB.W, np.random.default_rng([TLB.SEED_WINDOW, k, 3]))
    w2 = TLB.draw_window(theta, TLB.W, np.random.default_rng([TLB.SEED_WINDOW, k, 3]))
    assert np.array_equal(w1, w2)
