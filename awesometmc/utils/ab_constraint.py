#!/usr/bin/env python3
"""
awesometmc/utils/ab_constraint.py
==========================
Closed-form AB-constrained "AB constraint" parametrisation.

The AB constraint, derived from the Markovianity of (R, Y) (the TAC paper,
Proposition "nsc"), is the pair of **block identities**, for every regime r,

    A_r = M_r C_r,        B_r = M_r D_r,        M_r = Δ_r Σ_V,r⁻¹.

It is necessary and sufficient for the conditional independence of X_{n+1}
and (R_n, Y_n) given (R_{n+1}, Y_{n+1}) to hold **for every** joint covariance
of (X_n, Y_n), and it is what the constant-gain filter (``GSSFilter`` mode
``"ngh_kf"``) requires. The closed form

    A = Δ Σ_V⁻¹ C,        B = Δ Σ_V⁻¹ D

is therefore both the way to *build* an AB model from the free blocks
(C, D, Σ_U, Σ_V, Δ) and the way to *test* one: a model is AB-constrained iff
the block residuals ``A_r − M_r C_r`` and ``B_r − M_r D_r`` vanish at every
regime. This is the test implemented by :func:`ab_residual_max` and used by
:func:`validate_ngh_msm`, ``GSSFilter`` and ``awesometmc.utils.exactness``.

A weaker, *pairwise regression* condition is also available as a diagnostic:
property (PU), the conditional independence of X_{n+1} and Y_n given
(R_n, R_{n+1}, Y_{n+1}), evaluated for **one particular** joint law of
(X_n, Y_n) — the one whose covariance given r_n = j is the noise covariance
Σ_W(j), i.e. the stationary law of an AB model (Proposition 3). AB implies
(PU) for every law, so the (PU) residuals are necessary for AB, but they are
**not sufficient**: the TAC paper (Remark "forms") gives a K = 1 example
(A = 0.5, B = 0, C = 0.5, D = 0.3, Σ_W = I − F Fᵀ, stationary law) with (PU)
satisfied and A − M C ≈ 0.69, and
:func:`awesometmc.experiments.cns_exactness.jmls_cross_annihilation_model` is a
K = 2, q = 2 jump Markov linear system on which the regression residuals
vanish while the constant-gain filter is biased. These diagnostics live under
the ``pu_`` / ``compute_pu_`` names below.

Public API
----------
compute_AB(C, D, Dt, SV) -> (A, B)
    Closed-form A, B from C, D, Δ, Σ_V.
apply_AB_constraint(params, *, logger=None) -> GSSParams
    Return a new GSSParams with each regime's (A_k, B_k) replaced by
    the closed form.
compute_ab_block_residuals(A, B, C, D, Dt, SV, *, relative=False) -> (R_A, R_B)
    Block residuals ``A − M C`` and ``B − M D`` of one regime (optionally
    divided by their scales).
ab_residual_max(params, *, relative=True) -> (max_resid, (k, block))
    The AB test: largest (relative) block residual over all regimes and both
    blocks; 0 ⇔ the model is AB-constrained.
compute_pu_residual(A, B, C, D, SU, Dt, SV) -> ndarray (s, q)
    Same-regime (k, k) regression residual of (PU). Necessary for AB, not
    sufficient (ignores the cross pairs, and is a one-law condition).
compute_pu_pair_residual(A_k, B_k, C_k, D_k, Dt_k, SV_k, SU_j, Dt_j, SV_j) -> (q, s)
    Pairwise regression residual β₁(j, k) of (PU) for the ordered pair
    (source j → target k).
pu_residual_max(params) -> (max_resid, (j, k))
    Largest pairwise (PU) residual over all K² ordered pairs. Necessary for
    AB, not sufficient.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from awesometmc.classes.GSSParams import GSSParams

__all__ = [
    "NGH_MSM_RESID_TOL",
    "apply_AB_constraint",
    "compute_AB",
    "compute_ab_block_residuals",
    "ab_residual_max",
    "compute_pu_residual",
    "compute_pu_pair_residual",
    "pu_residual_max",
    "is_ngh_msm",
    "validate_ngh_msm",
]

# Default tolerance on the *relative* block AB residual used by
# validate_ngh_msm / is_ngh_msm. Mirrors GSSFilter.AB_TOL so that a model
# accepted here is also accepted (no warning) by mode="ngh_kf".
NGH_MSM_RESID_TOL = 1e-6


# ---------------------------------------------------------------------------
# AB block residuals — the AB test
# ---------------------------------------------------------------------------
def _rel_block(R: np.ndarray, X: np.ndarray, X_ab: np.ndarray) -> np.ndarray:
    """``R / max(‖X‖_F, ‖X_ab‖_F)``; ``R`` itself (it is then 0) when both vanish."""
    scale = max(float(np.linalg.norm(X, "fro")), float(np.linalg.norm(X_ab, "fro")))
    return R / scale if scale > 0.0 else R


def compute_ab_block_residuals(
    A: np.ndarray,  # (q, q)
    B: np.ndarray,  # (q, s)
    C: np.ndarray,  # (s, q)
    D: np.ndarray,  # (s, s)
    Dt: np.ndarray,  # (q, s)  Δ
    SV: np.ndarray,  # (s, s)  Σ_V
    *,
    relative: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Block residuals of the AB constraint for one regime:

        R_A = A − M C,        R_B = B − M D,        M = Δ Σ_V⁻¹.

    ``R_A = 0`` and ``R_B = 0`` ⇔ the regime satisfies AB. With
    ``relative=True`` the residuals are divided by the scale of the two terms
    whose difference they are, ``max(‖A‖_F, ‖M C‖_F)`` and
    ``max(‖B‖_F, ‖M D‖_F)`` (a zero scale leaves a zero residual), as in
    :func:`awesometmc.utils.exactness.cross_annihilation_residual`.

    Returns
    -------
    (R_A, R_B) : ndarrays of shapes (q, q) and (q, s).

    Raises
    ------
    ValueError
        If Σ_V is singular or ill-conditioned (cf. :func:`compute_AB`).
    """
    A_ab, B_ab = compute_AB(C, D, Dt, SV)
    R_A = A - A_ab
    R_B = B - B_ab
    if relative:
        R_A = _rel_block(R_A, A, A_ab)
        R_B = _rel_block(R_B, B, B_ab)
    return R_A, R_B


def ab_residual_max(
    params: GSSParams,
    *,
    relative: bool = True,
) -> tuple[float, tuple[int, str]]:
    """Largest block residual of the AB constraint over all regimes.

    Returns ``(max_resid, (k, block))`` where ``max_resid`` is the maximum over
    the regimes ``k`` and the two blocks ``"A"``, ``"B"`` of
    ``‖A_k − M_k C_k‖_F`` and ``‖B_k − M_k D_k‖_F`` (``M_k = Δ_k Σ_V,k⁻¹``) —
    divided, when ``relative=True``, by ``max(‖A_k‖_F, ‖M_k C_k‖_F)``, resp.
    ``max(‖B_k‖_F, ‖M_k D_k‖_F)``. ``max_resid = 0`` ⇔ the model is
    AB-constrained (a CGOMSM2 in the TAC paper's terms), which is the exact
    condition for the constant-gain filter; ``inf`` is returned if some
    regime's Σ_V is singular.

    This is the **complete** AB test. The regression residuals
    :func:`pu_residual_max` / :func:`compute_pu_residual` are necessary but
    not sufficient for AB (see the module docstring).
    """
    fm, nc = params.f_matrix, params.noise_cov
    worst = 0.0
    arg: tuple[int, str] = (0, "A")
    for k in range(params.K):
        try:
            R_A, R_B = compute_ab_block_residuals(
                fm.A(k), fm.B(k), fm.C(k), fm.D(k), nc.Delta(k), nc.Sigma_V(k), relative=relative
            )
        except np.linalg.LinAlgError, ValueError:
            return float("inf"), (k, "A")
        for block, R in (("A", R_A), ("B", R_B)):
            r = float(np.linalg.norm(R, "fro"))
            if r > worst:
                worst, arg = r, (k, block)
    return worst, arg


# ---------------------------------------------------------------------------
# (PU) regression residuals — diagnostics, necessary but not sufficient for AB
# ---------------------------------------------------------------------------
def compute_pu_residual(
    A: np.ndarray,  # (q, q)
    B: np.ndarray,  # (q, s)
    C: np.ndarray,  # (s, q)
    D: np.ndarray,  # (s, s)
    SU: np.ndarray,  # (q, q)  Σ_U
    Dt: np.ndarray,  # (q, s)  Δ
    SV: np.ndarray,  # (s, s)  Σ_V
) -> np.ndarray:
    """
    Same-regime regression residual of property (PU),

        F = (Δᵀ Aᵀ + Σ_V Bᵀ) − P M⁻¹ (Q Aᵀ + R Bᵀ + Δᵀ),

    with the auxiliary blocks

        P = Δᵀ Cᵀ + Σ_V Dᵀ          (s × s)
        Q = C Σ_U + D Δᵀ            (s × q)
        R = C Δ  + D Σ_V  (= Pᵀ)    (s × s)
        M = Q Cᵀ + R Dᵀ + Σ_V       (s × s, symmetric, ≻ 0 if Σ_U, Σ_V ≻ 0).

    ``F = 0`` ⇔ ``Cov(X_{n+1}, Y_n) = Cov(X_{n+1}, Y_{n+1}) Var(Y_{n+1})⁻¹
    Cov(Y_{n+1}, Y_n)`` for the regime pair ``(k, k)``, when the joint
    covariance of ``(X_n, Y_n)`` given ``r_n = k`` is taken equal to the noise
    covariance ``Σ_W(k)`` (the stationary one of an AB model). This is a
    **necessary** condition for AB (AB implies it for every law and every
    pair), but **not sufficient**: it ignores the cross pairs ``(j, k)``,
    ``j ≠ k``, and it is a condition on one particular law (the TAC paper,
    Remark "forms"). The AB test is :func:`ab_residual_max` /
    :func:`compute_ab_block_residuals`.

    Returns
    -------
    F : ndarray of shape (s, q).

    Raises
    ------
    numpy.linalg.LinAlgError
        If M is singular.
    """
    P = Dt.T @ C.T + SV @ D.T
    Q = C @ SU + D @ Dt.T
    R = C @ Dt + D @ SV
    M = Q @ C.T + R @ D.T + SV
    W = Q @ A.T + R @ B.T + Dt.T
    Z = Dt.T @ A.T + SV @ B.T
    residual: np.ndarray = Z - P @ np.linalg.solve(M, W)
    return residual


def _pu_pair_beta(
    A_k: np.ndarray,  # (q, q)  target-regime k dynamics
    B_k: np.ndarray,  # (q, s)
    C_k: np.ndarray,  # (s, q)
    D_k: np.ndarray,  # (s, s)
    Delta_k: np.ndarray,  # (q, s)  Cov(W_X, W_Y | r=k)
    SV_k: np.ndarray,  # (s, s)   Var(W_Y | r=k)
    SU_j: np.ndarray,  # (q, q)  source-regime j joint-covariance blocks
    Delta_j: np.ndarray,  # (q, s)
    SV_j: np.ndarray,  # (s, s)
) -> np.ndarray:
    """Regression of X_{n+1} on (Y_n, Y_{n+1}) given r_n=j, r_{n+1}=k, when the
    joint covariance of (X_n, Y_n) given r_n = j is Σ_W(j).

    Returns ``beta`` of shape (q, 2s); the first s columns are the loading on
    Y_n (``beta_1``), the last s the loading on Y_{n+1}.

    Raises numpy.linalg.LinAlgError if the joint (Y_n, Y_{n+1}) covariance is
    singular.
    """
    cov_Xn1_Yn = A_k @ Delta_j + B_k @ SV_j
    cov_Xn1_Yn1 = (
        A_k @ SU_j @ C_k.T
        + A_k @ Delta_j @ D_k.T
        + B_k @ Delta_j.T @ C_k.T
        + B_k @ SV_j @ D_k.T
        + Delta_k
    )
    cov_Yn_Yn1 = Delta_j.T @ C_k.T + SV_j @ D_k.T
    var_Yn1 = (
        C_k @ SU_j @ C_k.T
        + C_k @ Delta_j @ D_k.T
        + D_k @ Delta_j.T @ C_k.T
        + D_k @ SV_j @ D_k.T
        + SV_k
    )
    V = np.block([[SV_j, cov_Yn_Yn1], [cov_Yn_Yn1.T, var_Yn1]])  # (2s, 2s)
    cov_X = np.hstack([cov_Xn1_Yn, cov_Xn1_Yn1])  # (q, 2s)
    beta: np.ndarray = np.linalg.solve(V, cov_X.T).T  # (q, 2s)
    return beta


def compute_pu_pair_residual(
    A_k: np.ndarray,
    B_k: np.ndarray,
    C_k: np.ndarray,
    D_k: np.ndarray,
    Delta_k: np.ndarray,
    SV_k: np.ndarray,
    SU_j: np.ndarray,
    Delta_j: np.ndarray,
    SV_j: np.ndarray,
) -> np.ndarray:
    """Pairwise regression residual ``β₁(j, k)`` of property (PU) for the
    ordered pair (source j → target k).

    (PU) — the conditional independence of ``X_{n+1}`` and ``Y_n`` given
    ``(r_n = j, r_{n+1} = k, Y_{n+1})`` — holds for the pair iff the regression
    of ``X_{n+1}`` on ``(Y_n, Y_{n+1})`` does *not* load on ``Y_n``. This
    returns that loading ``β₁`` (shape ``(q, s)``), computed with the joint
    covariance of ``(X_n, Y_n)`` given ``r_n = j`` taken equal to ``Σ_W(j)``,
    the stationary one of an AB model.

    AB implies ``β₁(j, k) = 0`` for every pair, so these residuals are
    **necessary** for AB. They are **not sufficient**: the condition is on one
    particular law, and a model can satisfy it at every pair with
    ``A_k ≠ M_k C_k`` (module docstring). The AB test is
    :func:`ab_residual_max`.

    Target-regime arguments ``A_k, B_k, C_k, D_k`` (dynamics), ``Delta_k``
    (``Cov(W_X, W_Y | k)``), ``SV_k`` (``Var(W_Y | k)``); source-regime joint
    noise-covariance blocks ``SU_j, Delta_j, SV_j``.

    Raises numpy.linalg.LinAlgError if the joint (Y_n, Y_{n+1}) covariance is
    singular.
    """
    s = SV_j.shape[0]
    return _pu_pair_beta(A_k, B_k, C_k, D_k, Delta_k, SV_k, SU_j, Delta_j, SV_j)[:, :s]


def pu_residual_max(
    params: GSSParams,
    *,
    relative: bool = True,
) -> tuple[float, tuple[int, int]]:
    """Largest pairwise (PU) regression residual over all K² ordered regime pairs.

    Returns ``(max_resid, (j, k))`` where ``max_resid`` is the maximum over all
    ordered pairs (j, k) of ``‖β₁(j, k)‖_F`` — relative to the regression scale
    ``‖[β₁ β₂](j, k)‖_F`` when ``relative=True``; ``inf`` is returned if some
    pair's (Y_n, Y_{n+1}) covariance is singular.

    ``max_resid = 0`` is **necessary** for AB but **not sufficient** (it is a
    condition on one particular joint law, see :func:`compute_pu_pair_residual`):
    use :func:`ab_residual_max` to test AB. This diagnostic is the (PU)
    property of the stationary law of an AB model; it is 0 on every AB model.
    """
    K = params.K
    A = [params.f_matrix.A(k) for k in range(K)]
    B = [params.f_matrix.B(k) for k in range(K)]
    C = [params.f_matrix.C(k) for k in range(K)]
    D = [params.f_matrix.D(k) for k in range(K)]
    SU = [params.noise_cov.Sigma_U(k) for k in range(K)]
    Dt = [params.noise_cov.Delta(k) for k in range(K)]
    SV = [params.noise_cov.Sigma_V(k) for k in range(K)]

    s = params.s
    worst = 0.0
    arg: tuple[int, int] = (0, 0)
    for j in range(K):
        for k in range(K):
            try:
                beta = _pu_pair_beta(A[k], B[k], C[k], D[k], Dt[k], SV[k], SU[j], Dt[j], SV[j])
            except np.linalg.LinAlgError:
                return float("inf"), (j, k)
            r = float(np.linalg.norm(beta[:, :s], "fro"))
            if relative:
                r /= max(float(np.linalg.norm(beta, "fro")), 1e-300)
            if r > worst:
                worst = r
                arg = (j, k)
    return worst, arg


# ---------------------------------------------------------------------------
# CGOMSM2 validity (the corrected CNS of Proposition 2)
# ---------------------------------------------------------------------------
def _min_eig_sym(M: np.ndarray) -> float:
    """Smallest eigenvalue of the symmetric part of M (real, finite or -inf)."""
    Msym = 0.5 * (M + M.T)
    try:
        return float(np.linalg.eigvalsh(Msym)[0])
    except np.linalg.LinAlgError:
        return float("-inf")


def validate_ngh_msm(
    params: GSSParams,
    *,
    tol: float = NGH_MSM_RESID_TOL,
) -> list[str]:
    """List the ways ``params`` violates the corrected CGOMSM2 condition (Prop. 2).

    Returns an empty list iff ``params`` is a valid CGOMSM2: a model of the AB
    family whose structural hypotheses all hold. Each non-empty entry is a
    human-readable description of one violated condition. Nothing is raised — the
    caller decides what to do (cf. :meth:`GSSParams.check_ngh_msm`).

    Conditions checked (per the corrected Proposition 2)
    ----------------------------------------------------
    1. ``C_k ≠ 0``            ∀ k        — this is a genuine CGOMSM2 (the new
       family), not the degenerate ``C_k = 0`` case, which is a classical CGOMSM1
       (the (H4) family). NB: ``C_k ≠ 0`` is a *family-membership* requirement, not
       a mathematical prerequisite of the AB / NSC. AB is the necessary-and-
       sufficient AB parametrisation for *any* ``C_k`` given only ``Σ_V_k ≻ 0``,
       ``C_k = 0`` included — where it gives the CGOMSM1's ``A_k = 0``,
       ``B_k = Δ_k Σ_V_k⁻¹ D_k``. Full column rank of ``C_k`` (hence ``s ≥ q``) and
       invertibility of ``D_k`` are likewise *not* required: the corrected
       Proposition 2 needs only ``Σ_V_k ≻ 0`` (see the module docstring).
    2. ``Σ_V_k ≻ 0``          ∀ k        — symmetric positive definite;
    3. ``Γ_k = Σ_U_k − Δ_k Σ_V_k⁻¹ Δ_k^T ⪰ 0`` ∀ k — the Schur complement is a
       genuine covariance (⇔ the joint noise covariance Σ_W(k) is PSD);
    4. AB constraint   — the block test of :func:`ab_residual_max`: the relative
       residuals ``‖A_k − M_k C_k‖ / max(‖A_k‖, ‖M_k C_k‖)`` and
       ``‖B_k − M_k D_k‖ / max(‖B_k‖, ‖M_k D_k‖)``, ``M_k = Δ_k Σ_V_k⁻¹``, are
       ``≤ tol`` at every regime. (The pairwise regression residual
       :func:`pu_residual_max` is weaker — it vanishes on models the
       constant-gain filter is biased on — and is not used here.)

    Conditions (1)–(4) are exactly what makes the exact fast filter of
    Proposition 4 (closed form ``M_k = Δ_k Σ_V_k⁻¹``, ``Γ_k`` constant in n)
    applicable. They are *not* imposed at ``GSSParams`` construction, because the
    same class also serves model violating ABs handled by ``mode="gpb2"``.
    """
    issues: list[str] = []
    K = params.K

    for k in range(K):
        C = params.f_matrix.C(k)
        SU = params.noise_cov.Sigma_U(k)
        Dt = params.noise_cov.Delta(k)
        SV = params.noise_cov.Sigma_V(k)

        # ``C_k != 0`` is a *family-membership* check, NOT a mathematical validity
        # condition: AB is the NSC (Prop. 2) for ANY ``C_k`` given only ``Σ_V ≻ 0``
        # — ``C_k = 0`` included (the Lehmann matrix-inversion argument needs no
        # condition on C, and none on D; verified numerically). At ``C_k = 0`` AB
        # gives ``A_k = 0``, ``B_k = Δ_k Σ_V_k⁻¹ D_k``: a perfectly filterable model,
        # but a classical CGOMSM1 (the (H4) family), not the new CGOMSM2 family. We
        # flag it so callers do not mistake the old family for the new. (Full column
        # rank of ``C_k``, hence ``s >= q``, is likewise not required. Whether AB is
        # moreover the *unique* AB parametrisation is a separate identifiability
        # question, governed by ``K·s >= q + s`` — see the module docstring.)
        if float(np.linalg.norm(C, "fro")) <= 1e-12:
            issues.append(
                f"regime {k}: C = 0 — this is a classical CGOMSM1 (the (H4) family), "
                "not a CGOMSM2 (the new family)."
            )

        ev_SV = _min_eig_sym(SV)
        if ev_SV <= 0.0:
            issues.append(f"regime {k}: Σ_V is not positive definite (min eig = {ev_SV:.3e}).")
        else:
            Gamma = SU - Dt @ np.linalg.solve(SV, Dt.T)
            ev_G = _min_eig_sym(Gamma)
            psd_tol = 1e-9 * max(1.0, float(np.linalg.norm(SU, "fro")))
            if ev_G < -psd_tol:
                issues.append(
                    f"regime {k}: Γ = Σ_U − Δ Σ_V⁻¹ Δ^T is not PSD (min eig = {ev_G:.3e}); "
                    f"the joint noise covariance Σ_W is not PSD."
                )

    max_rel, (k, block) = ab_residual_max(params, relative=True)
    if not np.isfinite(max_rel):
        issues.append(f"AB residual is infinite (regime {k}: Σ_V is singular or ill-conditioned).")
    elif max_rel > tol:
        other = "C" if block == "A" else "D"
        issues.append(
            f"AB constraint violated: max relative block residual "
            f"‖{block} − M {other}‖ / max(‖{block}‖, ‖M {other}‖) = {max_rel:.3e} > tol = "
            f"{tol:.1e} (worst: regime {k}, block {block}; M = Δ Σ_V⁻¹). Enforce it "
            f"with apply_AB_constraint (A = Δ Σ_V⁻¹ C, B = Δ Σ_V⁻¹ D)."
        )

    return issues


def is_ngh_msm(params: GSSParams, *, tol: float = NGH_MSM_RESID_TOL) -> bool:
    """``True`` iff ``params`` is a valid CGOMSM2 (cf. :func:`validate_ngh_msm`)."""
    return not validate_ngh_msm(params, tol=tol)


# ---------------------------------------------------------------------------
# Closed-form AB constraint
# ---------------------------------------------------------------------------
def compute_AB(
    C: np.ndarray,  # (s, q)
    D: np.ndarray,  # (s, s)
    Dt: np.ndarray,  # (q, s)  Δ
    SV: np.ndarray,  # (s, s)  Σ_V (symmetric ≻ 0)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Closed-form AB parametrisation:

        A = Δ Σ_V⁻¹ C,        B = Δ Σ_V⁻¹ D.

    A model with these blocks satisfies AB for every regime pair
    (r₁, r₂), independently of Σ_U.

    Parameters
    ----------
    C : ndarray (s, q)
    D : ndarray (s, s)
    Dt : ndarray (q, s)  -- Δ
    SV : ndarray (s, s)  -- Σ_V (symmetric positive definite)

    Returns
    -------
    (A, B) : tuple of ndarrays of shapes (q, q) and (q, s).

    Raises
    ------
    ValueError
        If Σ_V is singular or ill-conditioned.
    """
    cond_SV = np.linalg.cond(SV)
    if cond_SV > 1e12:
        raise ValueError(
            f"Σ_V is ill-conditioned (cond = {cond_SV:.3e}); cannot reliably solve A = Δ Σ_V⁻¹ C."
        )
    try:
        SV_inv_C = np.linalg.solve(SV, C)  # Σ_V⁻¹ C  (s × q)
        SV_inv_D = np.linalg.solve(SV, D)  # Σ_V⁻¹ D  (s × s)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"Σ_V is singular: {exc}") from exc
    A = Dt @ SV_inv_C  # (q, q)
    B = Dt @ SV_inv_D  # (q, s)
    return A, B


# ---------------------------------------------------------------------------
# Application to a GSSParams object
# ---------------------------------------------------------------------------
def apply_AB_constraint(
    params: GSSParams,
    *,
    logger: logging.Logger | None = None,
) -> GSSParams:
    """
    Return a new GSSParams whose A(k), B(k) blocks are replaced by the
    closed form ``A_k = Δ_k Σ_V_k⁻¹ C_k``, ``B_k = Δ_k Σ_V_k⁻¹ D_k``
    for every regime k. C, D, Σ_U, Σ_V, Δ, Π, π₀, μ_z₀, Σ_z₀, b are
    preserved unchanged.

    Parameters
    ----------
    params : GSSParams
    logger : logging.Logger, optional
        If provided, INFO messages report the per-regime ‖A_new − A_old‖_F
        and ‖B_new − B_old‖_F.

    Returns
    -------
    GSSParams
        A *new* object with updated A, B blocks. Because the result satisfies
        the AB constraint by construction, it is returned as the stronger type
        :class:`~awesometmc.classes.GSSParams.NGHMSMParams` **when** the structural
        conditions also hold (C_k ≠ 0, Σ_V_k ≻ 0, Γ_k ⪰ 0). If one fails (e.g.
        C_k = 0 — the degenerate CGOMSM1 family — or a non-PSD Schur complement,
        for which the projected model has no valid exact filter), a base
        :class:`GSSParams` is returned and a warning is logged, preserving the
        "project what you can" contract for CLI/script paths.

    Raises
    ------
    ValueError
        If Σ_V(k) is singular for any k (the AB blocks cannot be computed).
    """
    # Import here to avoid circular imports at module level
    from awesometmc.classes.FMatrix import FMatrix
    from awesometmc.classes.GSSParams import GSSParams as _GSSParams
    from awesometmc.classes.GSSParams import NGHMSMParams as _NGHMSMParams
    from awesometmc.utils.exceptions import ParamError as _ParamError

    log = logger or logging.getLogger("awesomeTMC.ab_constraint")
    K, q, s = params.K, params.q, params.s

    A_list: list[np.ndarray] = []
    B_list: list[np.ndarray] = []
    C_list: list[np.ndarray] = []
    D_list: list[np.ndarray] = []

    for k in range(K):
        C_k = params.f_matrix.C(k)
        D_k = params.f_matrix.D(k)
        Dt_k = params.noise_cov.Delta(k)
        SV_k = params.noise_cov.Sigma_V(k)

        try:
            A_new, B_new = compute_AB(C_k, D_k, Dt_k, SV_k)
        except ValueError as exc:
            raise ValueError(f"AB constraint cannot be applied for regime k={k}: {exc}") from exc

        A_old = params.f_matrix.A(k)
        B_old = params.f_matrix.B(k)
        log.info(
            "k=%d  AB-constraint  ‖ΔA‖_F = %.4g, ‖ΔB‖_F = %.4g",
            k,
            float(np.linalg.norm(A_new - A_old, "fro")),
            float(np.linalg.norm(B_new - B_old, "fro")),
        )
        A_list.append(A_new)
        B_list.append(B_new)
        C_list.append(C_k)
        D_list.append(D_k)

    new_f_matrix = FMatrix(
        K=K,
        q=q,
        s=s,
        A_list=A_list,
        B_list=B_list,
        C_list=C_list,
        D_list=D_list,
    )

    kwargs = dict(
        K=K,
        q=q,
        s=s,
        P=params.P,
        f_matrix=new_f_matrix,
        noise_cov=params.noise_cov,
        pi0=params.pi0,
        mu_z0_list=[params.mu_z0(k) for k in range(K)],
        Sigma_z0_list=[params.Sigma_z0(k) for k in range(K)],
        b_list=[params.b(k) for k in range(K)],
    )
    # A, B are AB-correct by construction, so only the *structural* CNS can
    # fail. Prefer the stronger NGHMSMParams type; if a structural condition is
    # violated, fall back to a base GSSParams (warn, do not raise) so callers
    # that "project what they can" (CLI/script --constraint) keep working.
    try:
        return _NGHMSMParams(**kwargs)
    except _ParamError as exc:
        log.warning(
            "AB constraint applied, but the result is not a fully valid CGOMSM2 "
            "(%s); returning a base GSSParams.",
            exc,
        )
        return _GSSParams(**kwargs)
