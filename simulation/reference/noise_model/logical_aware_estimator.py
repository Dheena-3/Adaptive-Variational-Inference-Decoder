"""Logical-aware (syndrome + logical-bit) estimator for the d=3 correlated noise model.

Problem this fixes
------------------
`CorrelatedNoiseModel.fit_lbfgs` fits theta (21 fault rates) to the 16-bin syndrome
histogram P(s). Fault mechanisms with the same syndrome but different logical flips
(e.g. Z3 vs Z4Z5, both firing only check c2) are indistinguishable in P(s), so a Z3
burst and a Z4Z5 burst give the *same* histogram, yet they need opposite corrections.

Fix
---
Use a calibration shot's final data-qubit readout to get the logical bit l of that shot,
and fit theta to the joint counts n(s, l) with the exact 32-state DP likelihood

    P_new(s, l) = (1 - p_e) P_old(s, l) + p_e P_old(s ^ g_e, l ^ l_e)

    L(theta) = -(1/N) [ sum_{s,l} n(s,l) log P(s,l|theta) + sum_s u(s) log P(s|theta) ]
               + (lambda/2) ||theta - theta_prior||^2

where u(s) are optional unlabeled shots (syndrome only), so labeled and unlabeled data
can be mixed. The gradient has the same closed form as the 16-state case:

    dP(s,l)/dp_e = (P(s ^ g_e, l ^ l_e) - P(s,l)) / (1 - 2 p_e)

Label semantics
---------------
l is the parity of the logical-operator support of the *net* error of one shot.
In hardware it comes from a destructive final readout of all data qubits in the basis
that measures the logical operator (X basis for the X-bar = X0 X3 X6 logical here):
    l = parity(readout bits on logical support) XOR prepared_value XOR correction_l
It is only available per shot in calibration shots (prepare, one round, read out). In a
continuous multi-round memory run only the cumulative parity is visible, which carries
almost no information at tied syndromes (P(l=1|s)=0.5), so it is not used here.

Residual gauge (documented, not removable by this data): mechanisms with the same
(syndrome, logical-bit) pair, e.g. {Z3, Z6} or {ZZ45, ZZ78}, remain collinear. Only their
combined rate is identified. That does not change the optimal logical decision, but it
does change how the rate is split over qubits in the BP weights; the prior splits it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from simulation.reference.code_model.css_code_builder import (
    RotatedSurfaceCode,
    build_rotated_d3_code,
)
from simulation.reference.noise_model.joint_dp_likelihood import CorrelatedNoiseModel


# --------------------------------------------------------------------------------------
# Readout -> label
# --------------------------------------------------------------------------------------
def logical_bit_from_readout(
    readout_bits: np.ndarray,
    logical_support: np.ndarray,
    prepared_logical: int = 0,
    correction_logical: int = 0,
) -> int:
    """Logical-flip bit of one calibration shot from its final data-qubit readout.

    readout_bits       : (n_data,) 0/1 outcomes of the final transversal readout.
    logical_support    : (n_data,) 0/1 mask of the logical operator that was read out.
    prepared_logical   : eigenvalue bit the logical was prepared in (0 for |+_L>).
    correction_logical : parity of the applied correction on the logical support
                         (0 in calibration mode where no correction is applied).
    """
    r = np.asarray(readout_bits, dtype=np.uint8).ravel() & 1
    m = np.asarray(logical_support, dtype=np.uint8).ravel() & 1
    if r.shape != m.shape:
        raise ValueError(f"readout shape {r.shape} != logical support shape {m.shape}")
    return int((int(np.dot(r, m)) + int(prepared_logical) + int(correction_logical)) & 1)


@dataclass
class LabeledShotAccumulator:
    """Per-shot accumulator for the 32-state counts n(s, l) plus unlabeled counts u(s)."""

    n_states: int = 16
    logical_support: Optional[np.ndarray] = None
    counts_sl: np.ndarray = field(default=None)  # (n_states, 2)
    counts_s: np.ndarray = field(default=None)  # (n_states,) unlabeled shots

    def __post_init__(self) -> None:
        self.counts_sl = np.zeros((self.n_states, 2), dtype=np.int64)
        self.counts_s = np.zeros(self.n_states, dtype=np.int64)

    def add_shot(self, syndrome: int, logical_bit: Optional[int] = None) -> None:
        s = int(syndrome)
        if not 0 <= s < self.n_states:
            raise ValueError(f"syndrome {s} out of range")
        if logical_bit is None:
            self.counts_s[s] += 1
        else:
            self.counts_sl[s, int(logical_bit) & 1] += 1

    def add_readout_shot(
        self,
        syndrome: int,
        readout_bits: np.ndarray,
        prepared_logical: int = 0,
        correction_logical: int = 0,
    ) -> int:
        """Add one calibration shot from its raw readout bits. Returns the logical bit."""
        if self.logical_support is None:
            raise ValueError("logical_support must be set to label shots from a readout")
        ell = logical_bit_from_readout(
            readout_bits, self.logical_support, prepared_logical, correction_logical
        )
        self.add_shot(syndrome, ell)
        return ell

    def add_batch(self, syndromes: np.ndarray, logical_bits: Optional[np.ndarray] = None) -> None:
        s = np.asarray(syndromes, dtype=np.int64)
        if logical_bits is None:
            self.counts_s += np.bincount(s, minlength=self.n_states)
        else:
            idx = s * 2 + (np.asarray(logical_bits, dtype=np.int64) & 1)
            self.counts_sl += np.bincount(idx, minlength=2 * self.n_states).reshape(
                self.n_states, 2
            )

    @property
    def n_labeled(self) -> int:
        return int(self.counts_sl.sum())

    @property
    def n_total(self) -> int:
        return int(self.counts_sl.sum() + self.counts_s.sum())

    def marginal_histogram(self) -> np.ndarray:
        """16-bin syndrome histogram over all shots (what the old estimator sees)."""
        return (self.counts_sl.sum(axis=1) + self.counts_s).astype(np.float64)

    def reset(self) -> None:
        self.counts_sl.fill(0)
        self.counts_s.fill(0)


# --------------------------------------------------------------------------------------
# Logical-aware model
# --------------------------------------------------------------------------------------
class LogicalAwareNoiseModel(CorrelatedNoiseModel):
    """CorrelatedNoiseModel extended with the logical bit (32-state exact DP)."""

    def __init__(self, code: RotatedSurfaceCode, basis: str = "X", **kwargs) -> None:
        super().__init__(code=code, basis=basis, **kwargs)
        vec = np.asarray(
            code.logical_X[0] if self.basis == "X" else code.logical_Z[0], dtype=np.uint8
        )
        self.logical_support = vec
        self.logical_flips = np.array(
            [int(sum(int(vec[q]) for q in e.qubits) & 1) for e in self.events],
            dtype=np.int64,
        )
        # Mechanisms that change neither syndrome nor logical bit are unobservable.
        self.observable = (self.signatures != 0) | (self.logical_flips != 0)
        self._l_idx = np.arange(2, dtype=np.int64)

    # ---- forward DP ----------------------------------------------------------------
    def _source(self, e_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        s_src = self.xor_table[e_idx][:, None]
        l_src = (self._l_idx ^ int(self.logical_flips[e_idx]))[None, :]
        return s_src, l_src

    def joint_sl_distribution_dp(self, theta: np.ndarray) -> np.ndarray:
        """Exact P(s, l | theta), shape (n_states, 2)."""
        p = np.asarray(theta, dtype=np.float64)
        if p.shape != (self.n_params,):
            raise ValueError(f"Expected theta of shape ({self.n_params},), got {p.shape}")
        dist = np.zeros((self.n_states, 2), dtype=np.float64)
        dist[0, 0] = 1.0
        for e in range(self.n_params):
            pe = float(p[e])
            if not self.observable[e] or pe == 0.0:
                continue
            s_src, l_src = self._source(e)
            dist = (1.0 - pe) * dist + pe * dist[s_src, l_src]
        return dist

    def joint_sl_bruteforce(self, theta: np.ndarray, max_events: int = 18) -> np.ndarray:
        """Vectorised 2^E enumeration of P(s, l) for verification (events with p=0 skipped)."""
        p = np.asarray(theta, dtype=np.float64)
        active = [e for e in range(len(p)) if p[e] > 0.0]
        if len(active) > max_events:
            raise ValueError(f"brute force capped at {max_events} active events")
        k = len(active)
        masks = np.arange(1 << k, dtype=np.int64)
        prob = np.ones(1 << k)
        syn = np.zeros(1 << k, dtype=np.int64)
        lg = np.zeros(1 << k, dtype=np.int64)
        for j, e in enumerate(active):
            fired = (masks >> j) & 1
            prob *= np.where(fired == 1, p[e], 1.0 - p[e])
            syn ^= fired * int(self.signatures[e])
            lg ^= fired * int(self.logical_flips[e])
        dist = np.zeros((self.n_states, 2))
        np.add.at(dist, (syn, lg), prob)
        return dist

    def sl_distribution_and_jacobian(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """P(s,l) shape (S,2) and J[s,l,e] = dP(s,l)/dp_e shape (S,2,E)."""
        p = np.asarray(theta, dtype=np.float64)
        dist = self.joint_sl_distribution_dp(p)
        jac = np.zeros((self.n_states, 2, self.n_params), dtype=np.float64)
        for e in range(self.n_params):
            if not self.observable[e]:
                continue
            denom = 1.0 - 2.0 * float(p[e])
            if abs(denom) < 1e-12:
                denom = 1e-12 if denom >= 0 else -1e-12
            s_src, l_src = self._source(e)
            jac[:, :, e] = (dist[s_src, l_src] - dist) / denom
        return dist, jac

    # ---- loss ----------------------------------------------------------------------
    def negative_log_likelihood_and_grad_sl(
        self,
        theta: np.ndarray,
        counts_sl: np.ndarray,
        counts_s: Optional[np.ndarray] = None,
        prior_theta: Optional[np.ndarray] = None,
        prior_lambda: float = 0.0,
    ) -> Tuple[float, np.ndarray]:
        p = np.clip(np.asarray(theta, dtype=np.float64), 1e-6, 0.49)
        c_sl = np.asarray(counts_sl, dtype=np.float64)
        c_s = np.zeros(self.n_states) if counts_s is None else np.asarray(counts_s, np.float64)
        total = float(c_sl.sum() + c_s.sum())
        if total <= 0.0:
            raise ValueError("no shots to fit")

        dist, jac = self.sl_distribution_and_jacobian(p)
        safe = np.maximum(dist, 1e-15)
        marg = np.maximum(dist.sum(axis=1), 1e-15)
        jac_marg = jac.sum(axis=1)  # (S, E)

        nll = -(np.sum(c_sl * np.log(safe)) + np.sum(c_s * np.log(marg))) / total
        grad = -(
            np.einsum("sl,sle->e", c_sl / safe, jac) + (c_s / marg) @ jac_marg
        ) / total

        if prior_theta is not None and prior_lambda > 0.0:
            diff = p - np.asarray(prior_theta, dtype=np.float64)
            nll += 0.5 * prior_lambda * float(diff @ diff)
            grad = grad + prior_lambda * diff
        return float(nll), grad

    # ---- fit -----------------------------------------------------------------------
    def fit_lbfgs_labeled(
        self,
        counts_sl: np.ndarray,
        counts_s: Optional[np.ndarray] = None,
        init_theta: Optional[np.ndarray] = None,
        prior_theta: Optional[np.ndarray] = None,
        prior_lambda: float = 0.05,
        max_iter: int = 200,
    ) -> Dict[str, object]:
        """L-BFGS-B fit of theta to labeled (s,l) counts (and optional unlabeled counts)."""
        prior = self.default_theta.copy() if prior_theta is None else np.asarray(prior_theta, float)
        x0 = (prior if init_theta is None else np.asarray(init_theta, float)).copy()
        res = minimize(
            fun=lambda x: self.negative_log_likelihood_and_grad_sl(
                x, counts_sl, counts_s, prior_theta=prior, prior_lambda=prior_lambda
            ),
            x0=x0,
            jac=True,
            method="L-BFGS-B",
            bounds=[(1e-5, 0.45)] * self.n_params,
            options={"maxiter": max_iter, "ftol": 1e-13, "gtol": 1e-9},
        )
        theta_hat = np.clip(res.x, 1e-5, 0.45)
        llr_float, llr_q78 = self.compute_q7_8_llrs(theta_hat, pair_aware_projection=True)
        joint = self.joint_sl_distribution_dp(theta_hat)
        out = self.decisions_from_joint(joint)
        out.update(
            {
                "theta": theta_hat,
                "nll": float(res.fun),
                "nit": int(res.nit),
                "success": bool(res.success),
                "llr_float": llr_float,
                "llr_q7_8": llr_q78,
            }
        )
        return out

    def fit_from_accumulator(self, acc: LabeledShotAccumulator, **kwargs) -> Dict[str, object]:
        return self.fit_lbfgs_labeled(acc.counts_sl, acc.counts_s, **kwargs)

    # ---- decisions / metrics ---------------------------------------------------------
    @staticmethod
    def decisions_from_joint(joint: np.ndarray) -> Dict[str, object]:
        marg = np.maximum(joint.sum(axis=1), 1e-15)
        return {
            "posterior_l1_given_s": joint[:, 1] / marg,
            "map_logical_decision": np.argmax(joint, axis=1).astype(np.uint8),
        }

    def decision_ler(self, decisions: np.ndarray, theta_true: np.ndarray) -> float:
        """Exact LER of a decoder that applies logical decision d(s) at each syndrome,
        evaluated under the true theta: sum_s P_true(s, 1 - d(s))."""
        joint = self.joint_sl_distribution_dp(theta_true)
        d = np.asarray(decisions, dtype=np.int64)
        return float(sum(joint[s, 1 - d[s]] for s in range(self.n_states)))

    def bayes_optimal_ler(self, theta_true: np.ndarray) -> float:
        return float(np.sum(np.min(self.joint_sl_distribution_dp(theta_true), axis=1)))

    def sample_shots(
        self, theta: np.ndarray, n_shots: int, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate calibration shots: returns (syndromes, logical bits), each (n_shots,)."""
        p = np.asarray(theta, dtype=np.float64)
        fired = rng.random((n_shots, self.n_params)) < p[None, :]
        syn = np.bitwise_xor.reduce(np.where(fired, self.signatures[None, :], 0), axis=1)
        lg = np.bitwise_xor.reduce(np.where(fired, self.logical_flips[None, :], 0), axis=1)
        return syn.astype(np.int64), lg.astype(np.int64)

    def orbit_rates(self, theta: np.ndarray) -> Dict[Tuple[int, int], float]:
        """Identifiable quantity: XOR-combined rate of each (syndrome, logical-bit) orbit."""
        p = np.asarray(theta, dtype=np.float64)
        out: Dict[Tuple[int, int], float] = {}
        for e in range(self.n_params):
            if not self.observable[e]:
                continue
            key = (int(self.signatures[e]), int(self.logical_flips[e]))
            q = out.get(key, 0.0)
            out[key] = q * (1.0 - p[e]) + (1.0 - q) * p[e]
        return out


def build_d3_logical_aware_model(
    basis: str = "X", include_measurement_faults: bool = False
) -> LogicalAwareNoiseModel:
    return LogicalAwareNoiseModel(
        code=build_rotated_d3_code(),
        basis=basis,
        include_measurement_faults=include_measurement_faults,
    )
