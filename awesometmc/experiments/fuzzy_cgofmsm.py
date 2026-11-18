#!/usr/bin/env python3
"""
awesometmc/experiments/fuzzy_cgofmsm.py
=======================================
Reference implementation of the fuzzy switching filter of

    Z. Bouyahia, S. Derrode, W. Pieczynski, "Filtering in Gaussian linear systems
    with fuzzy switches", IEEE Trans. Fuzzy Systems 28(8), 1760-1770, 2020
    (DOI 10.1109/TFUZZ.2019.2921944) -- cited below as [BDP20].

The model is the "conditionally Gaussian observed fuzzy Markov switching model"
(CGOFMSM): a triplet ``(X, R, Y)`` with scalar ``X`` (hidden state) and ``Y``
(observation), and a regime ``R_n`` in ``[0, 1]`` whose laws are densities with
respect to ``nu = delta_0 + delta_1 + Lebesgue(0, 1)``, so that the two "hard"
regimes 0 and 1 carry positive probability and the open interval is "fuzzy".

    Z_{n+1} = A(r_n, r_{n+1}) (Z_n - M(r_n)) + M(r_{n+1}) + B(r_n, r_{n+1}) W_{n+1},
    Z_n = (X_n, Y_n),        A = [[a1, a2], [a3, a4]],

with ``a3 = 0`` in a CGOMSM: ``Y`` does not read ``X``, the pair ``(R, Y)`` is
Markov and the filter is exact. This module follows the paper section by section.

Contents
--------
Section V-A  ``FMC1``, ``FMC2``           the two fuzzy Markov chain laws of (R_1, R_2)
                                          (now in ``awesometmc.fuzzy.laws``, re-exported here)
Section V-B  ``FuzzyGaussianModel``       means linear in r, covariance bilinear in (r_1, r_2)
             ``paper_gaussian_model``     the numerical values of Section VI
Section VI   ``simulate``                 a CGOFMSM sample (r, x, y)
Remark 1     ``FuzzyGrid``                the F-step numerical integration over (0, 1)
                                          (``awesometmc.fuzzy.laws``)
             ``discretise``               the model on the grid: a hard CGOMSM with K = F + 2
                                          regimes and pair-dependent parameters
Section IV   ``fuzzy_filter``             the optimal fuzzy switching recursive filter (OFSRF)
             ``mpm_jumps``                the MPM restoration rule of the jumps (Section VI)
Appendix B   ``fuzzy_smoother``           regime smoother (in the sources of [BDP20], not printed)
Section VI   ``known_jumps_filter``       the "KJ" filter, regimes given
             ``harden``                   the 2-regime "hard" model used as the CGOMSM baseline
Validation   ``enumerate_paths``, ``exact_enumeration_filter``
                                          ground truth by enumeration of all regime paths

Deviations from the letter of [BDP20] are flagged ``NOTE [BDP20]`` in the code.

This module is the scalar reference of the paper. The public interface, for any
dimensions and built on ``PairGSSParams`` and ``GSSFilter``, is ``awesometmc.fuzzy``;
its tests use this module as their oracle.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from awesometmc.fuzzy.laws import FMC1, FMC2, FuzzyGrid, FuzzyMarkovLaw, mpm_jumps

__all__ = [
    "FMC1",
    "FMC2",
    "DiscreteModel",
    "FilterResult",
    "FuzzyGaussianModel",
    "FuzzyGrid",
    "FuzzyMarkovLaw",
    "discretise",
    "enumerate_paths",
    "exact_enumeration_filter",
    "fuzzy_filter",
    "fuzzy_smoother",
    "harden",
    "known_jumps_filter",
    "mpm_jumps",
    "paper_gaussian_model",
    "simulate",
]

LIKELIHOODS = ("paper", "moment")


def _log_normal(y, mean, var):
    return -0.5 * (np.log(2.0 * np.pi * var) + (y - mean) ** 2 / var)


def _logsumexp(a: np.ndarray) -> float:
    m = np.max(a)
    if not np.isfinite(m):
        return float(m)
    return float(m + np.log(np.sum(np.exp(a - m))))


# --------------------------------------------------------------------------
# Section V-B: Gaussian laws given the regimes
# --------------------------------------------------------------------------
def _transition_from_gamma(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(A, Q)`` of ``Z_2 | Z_1`` for a joint covariance ``G`` of ``(Z_1, Z_2)``.

    NOTE [BDP20]: the display after eq. (43) writes ``B`` with an inverse; the
    quantity needed, and computed here, is ``Q = B B^T = G22 - A G12``.
    """
    S11, S21, S22 = G[:2, :2], G[2:, :2], G[2:, 2:]
    A = S21 @ np.linalg.inv(S11)
    Q = S22 - A @ S21.T
    return A, 0.5 * (Q + Q.T)


@dataclass(frozen=True)
class FuzzyGaussianModel:
    """Law of ``(Z_1, Z_2)`` given ``(r_1, r_2)``: means ``M0, M1`` and the four hard
    covariances ``Gamma[i, j]`` (4x4, variables ordered ``x_1, y_1, x_2, y_2``).

    Means are interpolated linearly, eq. (interplineaire), covariances
    bilinearly, eq. (interpbilineaire).

    enforce_cgo
        If True, the cross term ``d = Cov(x_1, y_2)`` is recomputed after
        interpolation as ``d = b c / sigma_Y^2`` (with ``b = Cov(x_1, y_1)``,
        ``c = Cov(y_1, y_2)``), which is the condition ``a3 = 0`` of a CGOMSM.

        NOTE [BDP20]: the paper states the condition as ``d_ij = b_i c_ij`` and
        applies it at the four corners only. Bilinear interpolation does not
        preserve it, and the matrices printed in Section VI do not satisfy it at
        the corners (0,0) and (0,1). With ``enforce_cgo=False`` the model is
        exactly the printed one and ``a3 != 0`` in general.
    """

    M0: np.ndarray
    M1: np.ndarray
    Gamma: dict[tuple[int, int], np.ndarray]
    enforce_cgo: bool = False

    def __post_init__(self) -> None:
        for i, j in itertools.product((0, 1), repeat=2):
            G = np.asarray(self.Gamma[(i, j)], float)
            if G.shape != (4, 4) or not np.allclose(G, G.T):
                raise ValueError(f"Gamma[{i},{j}] must be a symmetric 4x4 matrix")
            if not np.allclose(G[:2, :2], self.Gamma[(i, 0)][:2, :2]) or not np.allclose(
                G[2:, 2:], self.Gamma[(0, j)][2:, 2:]
            ):
                raise ValueError("the marginal blocks of Gamma must depend on one regime only")
        for i in (0, 1):
            if not np.allclose(self.Gamma[(i, 0)][:2, :2], self.Gamma[(0, i)][2:, 2:]):
                raise ValueError("Gamma is not stationary: Var(Z_1 | i) != Var(Z_2 | i)")

    def mean(self, r: float) -> np.ndarray:
        return (1.0 - r) * np.asarray(self.M0, float) + r * np.asarray(self.M1, float)

    def cov(self, r: float) -> np.ndarray:
        """``Var(Z_n | r_n = r)``."""
        return (1.0 - r) * self.Gamma[(0, 0)][:2, :2] + r * self.Gamma[(1, 0)][:2, :2]

    def gamma(self, r1: float, r2: float) -> np.ndarray:
        G = (
            (1.0 - r1) * (1.0 - r2) * self.Gamma[(0, 0)]
            + r1 * r2 * self.Gamma[(1, 1)]
            + r1 * (1.0 - r2) * self.Gamma[(1, 0)]
            + r2 * (1.0 - r1) * self.Gamma[(0, 1)]
        )
        if self.enforce_cgo:
            G = G.copy()
            G[0, 3] = G[3, 0] = G[0, 1] * G[1, 3] / G[1, 1]
        return G

    def transition(self, r1: float, r2: float) -> tuple[np.ndarray, np.ndarray]:
        """``(A, Q)`` with ``Q = B B^T``, for the pair ``(r_n, r_{n+1}) = (r1, r2)``."""
        return _transition_from_gamma(self.gamma(r1, r2))

    def a3(self, r1: float, r2: float) -> float:
        """Coefficient of ``x_n`` in ``y_{n+1}``; zero in a CGOMSM."""
        return float(self.transition(r1, r2)[0][1, 0])


def paper_gaussian_model(enforce_cgo: bool = False) -> FuzzyGaussianModel:
    """The Gaussian laws used in Section VI of [BDP20] (one-column values)."""
    G00 = [[0.5, 0.3, 0.1, 0.06], [0.3, 1.0, 0.35, 0.4], [0.1, 0.35, 0.5, 0.3], [0.06, 0.4, 0.3, 1.0]]  # fmt: skip
    G01 = [[0.5, 0.3, 0.5, 0.135], [0.3, 1.0, 0.33, 0.6], [0.5, 0.33, 0.75, 0.3], [0.135, 0.6, 0.3, 0.5]]  # fmt: skip
    G10 = [[0.75, 0.3, 0.1, 0.24], [0.3, 0.5, 0.35, 0.4], [0.1, 0.35, 0.5, 0.3], [0.24, 0.4, 0.3, 1.0]]  # fmt: skip
    G11 = [[0.75, 0.3, 0.5, 0.18], [0.3, 0.5, 0.33, 0.3], [0.5, 0.33, 0.75, 0.3], [0.18, 0.3, 0.3, 0.5]]  # fmt: skip
    return FuzzyGaussianModel(
        M0=np.array([0.0, 3.0]),
        M1=np.array([0.0, 0.0]),
        Gamma={
            (0, 0): np.array(G00),
            (0, 1): np.array(G01),
            (1, 0): np.array(G10),
            (1, 1): np.array(G11),
        },  # fmt: skip
        enforce_cgo=enforce_cgo,
    )


def _sqrt_psd(Q: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(Q)
    if w.min() < -1e-9 * max(1.0, w.max()):
        raise ValueError(f"transition noise covariance is not positive semi-definite: {w}")
    return V * np.sqrt(np.clip(w, 0.0, None))


def simulate(
    law: FuzzyMarkovLaw, gauss: FuzzyGaussianModel, N: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A CGOFMSM sample ``(r, x, y)`` of length ``N`` (Section VI of [BDP20])."""
    r = law.sample(N, rng)
    z = np.empty((N, 2))
    z[0] = gauss.mean(r[0]) + _sqrt_psd(gauss.cov(r[0])) @ rng.standard_normal(2)
    for n in range(1, N):
        A, Q = gauss.transition(r[n - 1], r[n])
        z[n] = gauss.mean(r[n]) + A @ (z[n - 1] - gauss.mean(r[n - 1])) + _sqrt_psd(Q) @ rng.standard_normal(2)  # fmt: skip
    return r, z[:, 0], z[:, 1]


# --------------------------------------------------------------------------
# Remark 1 of Section IV: numerical integration w.r.t. nu
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DiscreteModel:
    """A CGOFMSM on a grid: a K-regime switching model with pair-dependent parameters.

    ``pi`` (K,), ``P`` (K, K): the regime chain. ``M`` (K, 2), ``S`` (K, 2, 2): mean
    and covariance of ``Z_n`` given the regime. ``A``, ``Q`` (K, K, 2, 2): transition
    of ``Z`` for the pair ``(r_n, r_{n+1}) = (rho_g, rho_h)``.
    """

    nodes: np.ndarray
    pi: np.ndarray
    P: np.ndarray
    M: np.ndarray
    S: np.ndarray
    A: np.ndarray
    Q: np.ndarray

    @property
    def K(self) -> int:
        return int(self.nodes.size)


def _discrete_model(nodes, pi, P, gauss: FuzzyGaussianModel) -> DiscreteModel:
    K = len(nodes)
    A, Q = np.empty((K, K, 2, 2)), np.empty((K, K, 2, 2))
    for g, h in itertools.product(range(K), repeat=2):
        A[g, h], Q[g, h] = gauss.transition(nodes[g], nodes[h])
    return DiscreteModel(
        nodes=np.asarray(nodes, float),
        pi=pi,
        P=P,
        M=np.array([gauss.mean(r) for r in nodes]),
        S=np.array([gauss.cov(r) for r in nodes]),
        A=A,
        Q=Q,
    )


def discretise(law: FuzzyMarkovLaw, gauss: FuzzyGaussianModel, F: int) -> DiscreteModel:
    """The CGOFMSM seen by the filter when the integrals use ``F`` interior steps."""
    grid = FuzzyGrid.midpoint(F)
    pi, P = law.discretise(grid)
    return _discrete_model(grid.nodes, pi, P, gauss)


def harden(law: FuzzyMarkovLaw, gauss: FuzzyGaussianModel) -> DiscreteModel:
    """The 2-regime CGOMSM baseline of Section VI: hard Gaussian laws, hardened chain."""
    pi, P = law.harden()
    return _discrete_model(np.array([0.0, 1.0]), pi, P, gauss)


# --------------------------------------------------------------------------
# Section IV: the optimal fuzzy switching recursive filter
# --------------------------------------------------------------------------
@dataclass
class FilterResult:
    """Output of ``fuzzy_filter``.

    E_x, Var_x (N,)   ``E[X_n | y_1^n]`` and ``Var(X_n | y_1^n)``
    post (N, K)       ``P(r_n = rho_g | y_1^n)`` on the grid nodes
    log_lik           ``log p(y_1^N)``
    m_cond (N, K)     ``E[X_n | r_n = rho_g, y_1^n]`` (kept for the smoother)
    kernels           (N-1, K, K), ``p(r_{n+1}, y_{n+1} | r_n, y_n) / p(y_{n+1} | y_1^n)``,
                      only with ``keep_history=True``
    """

    E_x: np.ndarray
    Var_x: np.ndarray
    post: np.ndarray
    log_lik: float
    m_cond: np.ndarray
    kernels: np.ndarray | None = None


def fuzzy_filter(
    model: DiscreteModel, ys, *, likelihood: str = "paper", keep_history: bool = False
) -> FilterResult:
    """The five-step recursion of Section IV of [BDP20] on a discretised model.

    likelihood
        ``"paper"``: ``p(y_{n+1} | r_n, r_{n+1}, y_n)`` is the Gaussian of mean
        ``a4 (y_n - M^Y) + M^Y`` and variance ``Q_yy``, as written under eq. (p).
        ``"moment"``: the same density is taken from the predicted moments of
        step (ii), which also carry ``a3``. The two coincide when ``a3 = 0``; the
        filter is then exact for the discretised model. When ``a3 != 0`` neither
        is exact: ``(R, Y)`` is no longer Markov.
    """
    if likelihood not in LIKELIHOODS:
        raise ValueError(f"likelihood must be one of {LIKELIHOODS}")
    ys = np.asarray(ys, float).ravel()
    N, K = ys.size, model.K
    MX, MY = model.M[:, 0], model.M[:, 1]
    Sxx, Sxy, Syy = model.S[:, 0, 0], model.S[:, 0, 1], model.S[:, 1, 1]
    A00, A01, A10, A11 = (model.A[..., i, j] for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))
    Q00, Q01, Q11 = model.Q[..., 0, 0], model.Q[..., 0, 1], model.Q[..., 1, 1]
    with np.errstate(divide="ignore"):
        logP, logpi = np.log(model.P), np.log(model.pi)

    E_x, Var_x = np.empty(N), np.empty(N)
    post, m_cond = np.empty((N, K)), np.empty((N, K))
    kernels = np.empty((N - 1, K, K)) if keep_history else None

    # n = 1: law of Z_1 given r_1
    lw = logpi + _log_normal(ys[0], MY, Syy)
    log_lik = _logsumexp(lw)
    p = np.exp(lw - log_lik)
    m = MX + Sxy / Syy * (ys[0] - MY)
    v = Sxx - Sxy**2 / Syy

    def _store(n):
        e = float(p @ m)
        E_x[n], Var_x[n] = e, float(p @ (v + m * m)) - e * e
        post[n], m_cond[n] = p, m

    _store(0)
    for n in range(1, N):
        yp, y = ys[n - 1], ys[n]
        dx, dy = (m - MX)[:, None], (yp - MY)[:, None]
        # step (ii): moments of Z_{n+1} given the pair and y_1^n
        pred_x = A00 * dx + A01 * dy + MX[None, :]
        pred_y = A10 * dx + A11 * dy + MY[None, :]
        al = A00**2 * v[:, None] + Q00
        be = A00 * A10 * v[:, None] + Q01
        de = A10**2 * v[:, None] + Q11
        # step (i): p(r_n, r_{n+1} | y_1^{n+1})
        if likelihood == "paper":
            ll = _log_normal(y, A11 * dy + MY[None, :], Q11)
        else:
            ll = _log_normal(y, pred_y, de)
        with np.errstate(divide="ignore"):
            lj = np.log(p)[:, None] + logP + ll
        c = _logsumexp(lj)
        log_lik += c
        J = np.exp(lj - c)
        if kernels is not None:
            kernels[n - 1] = np.exp(logP + ll - c)
        p_new = J.sum(axis=0)
        # step (iii): moments of X_{n+1} given the pair and y_1^{n+1}
        gain = be / de
        mp = pred_x + gain * (y - pred_y)
        m2p = mp**2 + al - gain * be
        # step (iv): integrate r_n out
        alive = p_new > 0.0
        back = J / np.where(alive, p_new, 1.0)[None, :]
        m_new = np.where(alive, (back * mp).sum(axis=0), MX)
        v_new = np.where(alive, (back * m2p).sum(axis=0) - m_new**2, Sxx)
        p, m, v = p_new, m_new, np.clip(v_new, 0.0, None)
        # step (v)
        _store(n)
    return FilterResult(E_x, Var_x, post, log_lik, m_cond, kernels)


def fuzzy_smoother(
    model: DiscreteModel, ys, *, likelihood: str = "paper"
) -> tuple[np.ndarray, np.ndarray]:
    """``(E[X_n | y_1^N], P(r_n | y_1^N))`` by the backward recursion of Appendix B.

    In a CGOMSM ``X_n`` is independent of the future ``(r, y)`` given
    ``(r_n, y_1^n)``, so smoothing only re-weights the regimes:
    ``E[X_n | y_1^N] = sum_g E[X_n | r_n = g, y_1^n] P(r_n = g | y_1^N)``.
    """
    res = fuzzy_filter(model, ys, likelihood=likelihood, keep_history=True)
    assert res.kernels is not None
    N, K = res.post.shape
    post_s, E_s = np.empty((N, K)), np.empty(N)
    beta = np.ones(K)
    for n in range(N - 1, -1, -1):
        if n < N - 1:
            beta = res.kernels[n] @ beta
        w = res.post[n] * beta
        post_s[n] = w / w.sum()
        E_s[n] = float(post_s[n] @ res.m_cond[n])
    return E_s, post_s


def known_jumps_filter(gauss: FuzzyGaussianModel, r, ys) -> tuple[np.ndarray, np.ndarray]:
    """``(E[X_n | r_1^n, y_1^n], Var)``: the "KJ" filter of Section VI, regimes given."""
    r, ys = np.asarray(r, float), np.asarray(ys, float).ravel()
    N = ys.size
    E_x, Var_x = np.empty(N), np.empty(N)
    S, M = gauss.cov(r[0]), gauss.mean(r[0])
    m = M[0] + S[0, 1] / S[1, 1] * (ys[0] - M[1])
    v = S[0, 0] - S[0, 1] ** 2 / S[1, 1]
    E_x[0], Var_x[0] = m, v
    for n in range(1, N):
        A, Q = gauss.transition(r[n - 1], r[n])
        M_prev, M_new = gauss.mean(r[n - 1]), gauss.mean(r[n])
        d = np.array([m - M_prev[0], ys[n - 1] - M_prev[1]])
        pred = A @ d + M_new
        C = np.outer(A[:, 0], A[:, 0]) * v + Q
        gain = C[0, 1] / C[1, 1]
        m = pred[0] + gain * (ys[n] - pred[1])
        v = C[0, 0] - gain * C[0, 1]
        E_x[n], Var_x[n] = m, v
    return E_x, Var_x


# --------------------------------------------------------------------------
# Ground truth: enumeration of all regime paths
# --------------------------------------------------------------------------
def enumerate_paths(model: DiscreteModel, ys) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All regime paths of positive probability, with their exact Kalman recursions.

    Returns ``(paths, log_w, means)``: ``paths`` (n_paths, N) the regime indices,
    ``log_w`` (n_paths,) the joint log-densities ``log p(path, y_1^N)``, and
    ``means`` (n_paths, N) the filtered means ``E[X_n | path_1^n, y_1^n]``.
    No Markov assumption on ``(R, Y)`` is made: this is exact whatever ``a3``.
    """
    ys = np.asarray(ys, float).ravel()
    N, K = ys.size, model.K
    paths, log_w, means = [], [], []
    for path in itertools.product(range(K), repeat=N):
        g = path[0]
        if model.pi[g] <= 0.0 or any(model.P[a, b] <= 0.0 for a, b in itertools.pairwise(path)):
            continue
        lw = np.log(model.pi[g]) + _log_normal(ys[0], model.M[g, 1], model.S[g, 1, 1])
        m = model.M[g, 0] + model.S[g, 0, 1] / model.S[g, 1, 1] * (ys[0] - model.M[g, 1])
        v = model.S[g, 0, 0] - model.S[g, 0, 1] ** 2 / model.S[g, 1, 1]
        ms = [m]
        for n in range(1, N):
            h = path[n]
            A, Q = model.A[g, h], model.Q[g, h]
            pred = A @ np.array([m - model.M[g, 0], ys[n - 1] - model.M[g, 1]]) + model.M[h]
            C = np.outer(A[:, 0], A[:, 0]) * v + Q
            lw += np.log(model.P[g, h]) + _log_normal(ys[n], pred[1], C[1, 1])
            gain = C[0, 1] / C[1, 1]
            m, v, g = pred[0] + gain * (ys[n] - pred[1]), C[0, 0] - gain * C[0, 1], h
            ms.append(m)
        paths.append(path)
        log_w.append(lw)
        means.append(ms)
    return np.array(paths), np.array(log_w), np.array(means)


def exact_enumeration_filter(model: DiscreteModel, ys) -> tuple[float, np.ndarray, float]:
    """Exact ``(E[X_N | y_1^N], P(r_N | y_1^N), log p(y_1^N))`` of the discretised model,
    by enumeration of the ``K^N`` regime paths (see ``enumerate_paths``)."""
    paths, log_w, means = enumerate_paths(model, ys)
    log_lik = _logsumexp(log_w)
    w = np.exp(log_w - log_lik)
    post = np.bincount(paths[:, -1], weights=w, minlength=model.K)
    return float(w @ means[:, -1]), post, log_lik
