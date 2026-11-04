#!/usr/bin/env python3
"""
tests/test_ab_constraint.py
===========================
Unit tests for prg/utils/ab_constraint.py.

Coverage
--------
- compute_AB      : output shape, residual zero, Σ_V singular
- apply_AB_constraint : return type, residual zero for all regimes,
                             preservation of (C, D, Σ_W, biases),
                             idempotency
- compute_ab_block_residuals / ab_residual_max : the AB test (block residuals),
                             zero on the AB manifold, non-zero off it, strictly
                             stronger than the (PU) regression residuals
- compute_pu_residual / compute_pu_pair_residual / pu_residual_max : the (PU)
                             regression diagnostics, zero on the AB manifold
"""

from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest

from prg.classes.GSSParams import GSSParams
from prg.models.base_gss_model import BaseGSSModel
from prg.utils.ab_constraint import (
    ab_residual_max,
    apply_AB_constraint,
    compute_AB,
    compute_ab_block_residuals,
    compute_pu_pair_residual,
    compute_pu_residual,
    pu_residual_max,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_pd(size, rng, scale=0.5):
    """Return a random positive-definite (size × size) matrix."""
    L = rng.standard_normal((size, size)) * scale
    return L @ L.T + np.eye(size) * 0.2


def _random_inputs(q, s, rng):
    """Build (C, D, Σ_U, Δ, Σ_V) with compatible shapes and PD covariances."""
    C = rng.standard_normal((s, q)) * 0.3
    D = rng.standard_normal((s, s)) * 0.3
    SU = _make_pd(q, rng)
    Dt = rng.standard_normal((q, s)) * 0.1
    SV = _make_pd(s, rng)
    return C, D, SU, Dt, SV


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rng():
    """Shared RNG (fixed seed) for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def params_K2_q1_s1() -> GSSParams:
    """GSSParams loaded dynamically from model_gss_K2_q1_s1."""
    mod = importlib.import_module("prg.models.model_gss_K2_q1_s1")
    cls = next(
        c
        for _, c in inspect.getmembers(mod, inspect.isclass)
        if issubclass(c, BaseGSSModel) and c is not BaseGSSModel
    )
    return GSSParams.from_model(cls())


# ---------------------------------------------------------------------------
# compute_AB
# ---------------------------------------------------------------------------
class TestComputeAB:
    def test_output_shapes(self, rng):
        q, s = 2, 3
        C, D, _, Dt, SV = _random_inputs(q, s, rng)
        A, B = compute_AB(C, D, Dt, SV)
        assert A.shape == (q, q), f"expected ({q},{q}), got {A.shape}"
        assert B.shape == (q, s), f"expected ({q},{s}), got {B.shape}"

    def test_residual_zero(self, rng):
        """A model parametrised by the AB constraint must satisfy AB for any Σ_U."""
        q, s = 2, 3
        C, D, SU, Dt, SV = _random_inputs(q, s, rng)
        A, B = compute_AB(C, D, Dt, SV)
        F = compute_pu_residual(A, B, C, D, SU, Dt, SV)
        assert np.linalg.norm(F, "fro") < 1e-10, (
            f"residual at AB-constraint = {np.linalg.norm(F, 'fro'):.3e}"
        )

    def test_residual_zero_independent_of_SU(self, rng):
        """The AB residual under the AB constraint must remain zero for any Σ_U."""
        q, s = 2, 3
        C, D, _, Dt, SV = _random_inputs(q, s, rng)
        A, B = compute_AB(C, D, Dt, SV)
        for _ in range(5):
            SU_alt = _make_pd(q, rng)
            F = compute_pu_residual(A, B, C, D, SU_alt, Dt, SV)
            assert np.linalg.norm(F, "fro") < 1e-10

    def test_singular_SV_raises(self):
        """Σ_V = 0 must raise ValueError."""
        q, s = 2, 2
        C = np.zeros((s, q))
        D = np.zeros((s, s))
        Dt = np.zeros((q, s))
        SV = np.zeros((s, s))  # singular
        with pytest.raises(ValueError):
            compute_AB(C, D, Dt, SV)

    def test_zero_delta_yields_zero_AB(self, rng):
        """Δ = 0 forces A = B = 0 (degenerate but valid)."""
        q, s = 2, 3
        C = rng.standard_normal((s, q))
        D = rng.standard_normal((s, s))
        Dt = np.zeros((q, s))
        SV = _make_pd(s, rng)
        A, B = compute_AB(C, D, Dt, SV)
        np.testing.assert_array_equal(A, np.zeros((q, q)))
        np.testing.assert_array_equal(B, np.zeros((q, s)))


# ---------------------------------------------------------------------------
# apply_AB_constraint
# ---------------------------------------------------------------------------
class TestApplyABConstraint:
    def test_output_type(self, params_K2_q1_s1):
        result = apply_AB_constraint(params_K2_q1_s1)
        assert isinstance(result, GSSParams)

    def test_residual_zero_for_all_regimes(self, params_K2_q1_s1):
        """Every regime k of the projected model satisfies AB."""
        constrained = apply_AB_constraint(params_K2_q1_s1)
        for k in range(constrained.K):
            A = constrained.f_matrix.A(k)
            B = constrained.f_matrix.B(k)
            C = constrained.f_matrix.C(k)
            D = constrained.f_matrix.D(k)
            SU = constrained.noise_cov.Sigma_U(k)
            Dt = constrained.noise_cov.Delta(k)
            SV = constrained.noise_cov.Sigma_V(k)
            F = compute_pu_residual(A, B, C, D, SU, Dt, SV)
            assert np.linalg.norm(F, "fro") < 1e-10, (
                f"k={k}: residual = {np.linalg.norm(F, 'fro'):.3e}"
            )

    def test_preserves_C_D(self, params_K2_q1_s1):
        constrained = apply_AB_constraint(params_K2_q1_s1)
        for k in range(constrained.K):
            np.testing.assert_array_equal(constrained.f_matrix.C(k), params_K2_q1_s1.f_matrix.C(k))
            np.testing.assert_array_equal(constrained.f_matrix.D(k), params_K2_q1_s1.f_matrix.D(k))

    def test_preserves_noise_cov(self, params_K2_q1_s1):
        constrained = apply_AB_constraint(params_K2_q1_s1)
        for k in range(constrained.K):
            np.testing.assert_array_equal(
                constrained.noise_cov.Sigma_U(k), params_K2_q1_s1.noise_cov.Sigma_U(k)
            )
            np.testing.assert_array_equal(
                constrained.noise_cov.Delta(k), params_K2_q1_s1.noise_cov.Delta(k)
            )
            np.testing.assert_array_equal(
                constrained.noise_cov.Sigma_V(k), params_K2_q1_s1.noise_cov.Sigma_V(k)
            )

    def test_preserves_bias(self, params_K2_q1_s1):
        constrained = apply_AB_constraint(params_K2_q1_s1)
        for k in range(constrained.K):
            np.testing.assert_array_equal(constrained.b(k), params_K2_q1_s1.b(k))

    def test_idempotent(self, params_K2_q1_s1):
        """Applying twice yields the same A, B."""
        first = apply_AB_constraint(params_K2_q1_s1)
        second = apply_AB_constraint(first)
        for k in range(first.K):
            np.testing.assert_allclose(first.f_matrix.A(k), second.f_matrix.A(k), atol=1e-12)
            np.testing.assert_allclose(first.f_matrix.B(k), second.f_matrix.B(k), atol=1e-12)


# ---------------------------------------------------------------------------
# compute_pu_residual (same-regime (PU) regression residual)
# ---------------------------------------------------------------------------
class TestComputePUResidual:
    def test_zero_at_AB(self, rng):
        """The same-regime (PU) residual is zero on the AB-constraint manifold."""
        q, s = 2, 3
        C, D, SU, Dt, SV = _random_inputs(q, s, rng)
        A, B = compute_AB(C, D, Dt, SV)
        F = compute_pu_residual(A, B, C, D, SU, Dt, SV)
        assert F.shape == (s, q)
        assert np.linalg.norm(F, "fro") < 1e-10

    def test_nonzero_for_random_AB(self, rng):
        """The same-regime (PU) residual is generically non-zero for arbitrary A, B."""
        q, s = 2, 3
        C, D, SU, Dt, SV = _random_inputs(q, s, rng)
        A = rng.standard_normal((q, q)) * 0.5
        B = rng.standard_normal((q, s)) * 0.5
        F = compute_pu_residual(A, B, C, D, SU, Dt, SV)
        assert np.linalg.norm(F, "fro") > 1e-3


# ---------------------------------------------------------------------------
# Pairwise (all-pairs) (PU) regression check
# ---------------------------------------------------------------------------
class TestPairwisePU:
    def test_ab_model_fully_compatible(self, params_K2_q1_s1):
        """An AB-constrained model satisfies (PU) across every regime pair."""
        ab = apply_AB_constraint(params_K2_q1_s1)
        max_rel, _ = pu_residual_max(ab)
        assert max_rel < 1e-9, f"AB model violates (PU): {max_rel:.3e}"

    def test_pair_residual_zero_on_AB_manifold(self, rng):
        """β₁(j, k) vanishes for all pairs when every regime is AB-parametrised."""
        q, s = 2, 1
        reg = []
        for _ in range(2):
            C, D, SU, Dt, SV = _random_inputs(q, s, rng)
            A, B = compute_AB(C, D, Dt, SV)
            reg.append((A, B, C, D, SU, Dt, SV))
        for j in range(2):
            for k in range(2):
                Ak, Bk, Ck, Dk, _, Dtk, SVk = reg[k]
                _, _, _, _, SUj, Dtj, SVj = reg[j]
                b1 = compute_pu_pair_residual(Ak, Bk, Ck, Dk, Dtk, SVk, SUj, Dtj, SVj)
                assert np.linalg.norm(b1, "fro") < 1e-9

    def test_pairwise_strictly_stronger_than_same_regime(self):
        """A non-AB model can satisfy *every* same-regime (PU) residual yet
        violate (PU) on a cross pair (j != k); the pairwise check catches it.

        Built in the sub-determined regime K=2, q=2, s=1 (Ks < q+s), where the
        same-regime residual has a non-trivial null space.
        """
        rng = np.random.default_rng(3)
        q, s = 2, 1

        def spd(n):
            M = rng.standard_normal((n, n))
            return M @ M.T + 0.5 * np.eye(n)

        reg = []
        for _ in range(2):
            C = rng.standard_normal((s, q))
            D = rng.standard_normal((s, s))
            SU, SV, Dt = spd(q), spd(s), 0.3 * rng.standard_normal((q, s))
            A, B = compute_AB(C, D, Dt, SV)
            reg.append([A, B, C, D, SU, Dt, SV])

        # Perturb regime 0's (A, B) inside the null space of its same-regime residual.
        r0 = reg[0]
        base = np.concatenate([r0[0].ravel(), r0[1].ravel()])
        nA = q * q

        def mono_of(vec):
            A = vec[:nA].reshape(q, q)
            B = vec[nA:].reshape(q, s)
            return compute_pu_residual(A, B, r0[2], r0[3], r0[4], r0[5], r0[6]).ravel()

        n = base.size
        J = np.zeros((s * q, n))
        rb = mono_of(base)
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1.0
            J[:, i] = mono_of(base + e) - rb
        _, S, Vt = np.linalg.svd(J)
        null_dir = Vt[int((S > 1e-9).sum()) :][0]
        pert = base + null_dir
        reg[0][0] = pert[:nA].reshape(q, q)
        reg[0][1] = pert[nA:].reshape(q, s)

        # Same-regime residual stays ~0 for every regime (necessary condition holds) ...
        mono_max = max(np.linalg.norm(compute_pu_residual(*reg[k]), "fro") for k in range(2))
        assert mono_max < 1e-8, f"same-regime residual unexpectedly nonzero: {mono_max:.2e}"

        # ... yet the pairwise check detects the cross-pair (PU) violation.
        pair_max = 0.0
        for j in range(2):
            for k in range(2):
                b1 = compute_pu_pair_residual(
                    reg[k][0],
                    reg[k][1],
                    reg[k][2],
                    reg[k][3],
                    reg[k][5],
                    reg[k][6],
                    reg[j][4],
                    reg[j][5],
                    reg[j][6],
                )
                pair_max = max(pair_max, float(np.linalg.norm(b1, "fro")))
        assert pair_max > 1e-2, f"pairwise check missed the violation: {pair_max:.2e}"


# ---------------------------------------------------------------------------
# Block AB test: compute_ab_block_residuals / ab_residual_max
# ---------------------------------------------------------------------------
class TestBlockAB:
    def test_zero_on_AB_manifold(self, rng):
        q, s = 2, 3
        C, D, SU, Dt, SV = _random_inputs(q, s, rng)
        A, B = compute_AB(C, D, Dt, SV)
        R_A, R_B = compute_ab_block_residuals(A, B, C, D, Dt, SV)
        assert R_A.shape == (q, q) and R_B.shape == (q, s)
        assert np.linalg.norm(R_A, "fro") < 1e-12 and np.linalg.norm(R_B, "fro") < 1e-12

    def test_residual_is_the_block_difference(self, rng):
        q, s = 2, 1
        C, D, SU, Dt, SV = _random_inputs(q, s, rng)
        A_ab, B_ab = compute_AB(C, D, Dt, SV)
        dA = rng.standard_normal((q, q))
        dB = rng.standard_normal((q, s))
        R_A, R_B = compute_ab_block_residuals(A_ab + dA, B_ab + dB, C, D, Dt, SV)
        np.testing.assert_allclose(R_A, dA, atol=1e-12)
        np.testing.assert_allclose(R_B, dB, atol=1e-12)
        # relative: divided by max(‖A‖, ‖M C‖), resp. max(‖B‖, ‖M D‖)
        rA, rB = compute_ab_block_residuals(A_ab + dA, B_ab + dB, C, D, Dt, SV, relative=True)
        sA = max(np.linalg.norm(A_ab + dA, "fro"), np.linalg.norm(A_ab, "fro"))
        sB = max(np.linalg.norm(B_ab + dB, "fro"), np.linalg.norm(B_ab, "fro"))
        np.testing.assert_allclose(rA, dA / sA, atol=1e-12)
        np.testing.assert_allclose(rB, dB / sB, atol=1e-12)

    def test_relative_residual_is_scale_invariant(self, rng):
        q, s = 2, 2
        C, D, SU, Dt, SV = _random_inputs(q, s, rng)
        A = rng.standard_normal((q, q))
        B = rng.standard_normal((q, s))
        r1 = compute_ab_block_residuals(A, B, C, D, Dt, SV, relative=True)
        r2 = compute_ab_block_residuals(A, B, 7.0 * C, 7.0 * D, Dt, SV, relative=True)
        # scaling (C, D) by 7 scales (M C, M D) by 7: the residual changes, but
        # scaling (A, B, C, D) together leaves the relative residual unchanged
        r3 = compute_ab_block_residuals(7 * A, 7 * B, 7 * C, 7 * D, Dt, SV, relative=True)
        assert not np.allclose(r1[0], r2[0])
        np.testing.assert_allclose(r1[0], r3[0], atol=1e-12)
        np.testing.assert_allclose(r1[1], r3[1], atol=1e-12)

    def test_zero_blocks_give_zero_residual(self):
        q, s = 2, 1
        C = np.zeros((s, q))
        D = np.zeros((s, s))
        Dt = np.zeros((q, s))
        SV = np.eye(s)
        R_A, R_B = compute_ab_block_residuals(
            np.zeros((q, q)), np.zeros((q, s)), C, D, Dt, SV, relative=True
        )
        assert np.all(R_A == 0.0) and np.all(R_B == 0.0)

    def test_ab_model_passes(self, params_K2_q1_s1):
        ab = apply_AB_constraint(params_K2_q1_s1)
        max_rel, _ = ab_residual_max(ab)
        assert max_rel < 1e-12, f"AB model fails the block test: {max_rel:.3e}"

    def test_reports_the_worst_regime_and_block(self, params_K2_q1_s1):
        from prg.classes.FMatrix import FMatrix

        ab = apply_AB_constraint(params_K2_q1_s1)
        K, q, s = ab.K, ab.q, ab.s
        B_list = [ab.f_matrix.B(k).copy() for k in range(K)]
        B_list[1] = B_list[1] + 0.2
        fm = FMatrix(
            K,
            q,
            s,
            [ab.f_matrix.A(k) for k in range(K)],
            B_list,
            [ab.f_matrix.C(k) for k in range(K)],
            [ab.f_matrix.D(k) for k in range(K)],
        )
        broken = GSSParams(
            K=K,
            q=q,
            s=s,
            P=ab.P,
            f_matrix=fm,
            noise_cov=ab.noise_cov,
            pi0=ab.pi0,
            mu_z0_list=[ab.mu_z0(k) for k in range(K)],
            Sigma_z0_list=[ab.Sigma_z0(k) for k in range(K)],
            b_list=[ab.b(k) for k in range(K)],
        )
        max_rel, (k, block) = ab_residual_max(broken)
        assert max_rel > 1e-2 and (k, block) == (1, "B")

    def test_block_test_is_strictly_stronger_than_pu(self):
        """The (PU) regression residuals vanish on a JMLS in the
        cross-annihilation family which is not AB (A_r != M_r C_r): only the
        block test rejects it (the constant-gain filter is biased on it)."""
        from prg.experiments.cns_exactness import jmls_cross_annihilation_model

        p = jmls_cross_annihilation_model()
        pu, _ = pu_residual_max(p)
        assert pu < 1e-12, f"(PU) should hold on the stationary law: {pu:.2e}"
        max_rel, (k, block) = ab_residual_max(p)
        assert max_rel > 0.1 and block == "A", (max_rel, k, block)
