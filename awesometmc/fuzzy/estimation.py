#!/usr/bin/env python3
"""
awesometmc/fuzzy/estimation.py
==============================
Supervised estimation of a fuzzy switching model from a **labelled** sample
``(r_n, x_n, y_n)``, the regimes ``r_n`` in ``[0, 1]`` being known.

This is the setting of Section VII of [BDP20] (Bouyahia, Derrode, Pieczynski, IEEE TFS
2020), where a pseudo ground truth of the jumps is laid over real data and "classical
empirical estimators" give the means and covariances in the hard regimes on the one
hand, the parameters of the law of the jumps on the other.

Law of the jumps -- ``fit_law``
    From the pairs ``(r_n, r_{n+1})``. The masses of the four hard corners come from
    their frequencies. Off the corners:

    FMC1  the density is ``eta (1 - d) + delta d`` with ``d = |r_1 - r_2|``. Two moments
          are linear in ``(eta, delta)``,

              P(off the corners)        = 8/3 eta + 7/3 delta
              E[d ; off the corners]    = 5/6 eta + 3/2 delta,

          which gives a closed-form estimate (``method="moments"``). The default,
          ``method="ml"``, maximises the likelihood of the pairs, which is concave,
          starting from it.
    FMC2  the density is ``gamma`` on the band ``d <= delta``: ``delta`` is the largest
          jump seen off the corners, and ``gamma delta (6 - delta)`` the mass there.

Gaussian laws -- ``fit_gaussian``
    The model states that, given the regimes, ``Z_n = (X_n, Y_n)`` has mean
    ``(1 - r_n) M(0) + r_n M(1)``, variance ``(1 - r_n) S(0) + r_n S(1)`` and
    cross-covariance ``Cov(Z_n, Z_{n+1})`` bilinear in ``(r_n, r_{n+1})`` between four
    hard corners. Two estimators:

    ``method="regression"`` (default)
        least squares on **all** the samples, fuzzy ones included: ``z_n`` on
        ``(1 - r_n, r_n)`` for the means, the squared residuals on the same design for
        ``S(0), S(1)``, and their lagged products on the four bilinear weights for the
        corners. It needs no hard sample at all, and reduces to the empirical
        estimators when every regime is hard.
    ``method="hard"``
        the letter of the paper: empirical means and covariances over the hard
        samples and the hard pairs only. It fails when a hard pair never occurs,
        which is the case of the pseudo ground truth of Section VII (regimes move
        from 0 to 1 through fuzzy values, so the pairs (0, 1) and (1, 0) are absent).

    Two corrections keep the result a valid model, and are reported:

    * a corner whose joint covariance is not positive definite has its
      cross-covariance shrunk towards zero (``cross_shrink < 1``);
    * with ``cgo=True``, ``Cov(X_1, Y_2)`` is replaced at each corner by
      ``Cov(X_1, Y_1) Var(Y_1)^-1 Cov(Y_1, Y_2)``, the identity under which the
      observation does not read the state (``C = 0``) and the filter is exact. The
      block it replaces is not bilinear under ``interpolation="cgo"`` anyway.

``fit_supervised`` chains the two and returns a ``FuzzyCGOMSM`` ready to discretise.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from awesometmc.fuzzy.laws import FMC1, FMC2, FuzzyMarkovLaw
from awesometmc.fuzzy.model import INTERPOLATIONS, FuzzyCGOMSM
from awesometmc.utils.exceptions import ParamError

__all__ = ["FuzzyFit", "GaussianFit", "fit_gaussian", "fit_law", "fit_supervised"]

LAW_FAMILIES = ("fmc1", "fmc2")
GAUSSIAN_METHODS = ("regression", "hard")
_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))
#: smallest eigenvalue accepted for a covariance, relative to its largest
_PD_FLOOR = 1e-8


# ---------------------------------------------------------------------------
# Law of the jumps
# ---------------------------------------------------------------------------


def _regimes(r) -> np.ndarray:
    r = np.asarray(r, dtype=float).ravel()
    if r.size < 2:
        raise ParamError("at least two regimes are needed.")
    if not np.all(np.isfinite(r)) or r.min() < 0.0 or r.max() > 1.0:
        raise ParamError("the regimes must lie in [0, 1].")
    return r


def _pair_statistics(r: np.ndarray) -> tuple[int, int, np.ndarray]:
    """``(n_same, n_cross, d)``: counts of the hard pairs (0,0)/(1,1) and (0,1)/(1,0),
    and the jumps ``|r_1 - r_2|`` of the other pairs."""
    r1, r2 = r[:-1], r[1:]
    hard = ((r1 == 0.0) | (r1 == 1.0)) & ((r2 == 0.0) | (r2 == 1.0))
    n_same = int(np.sum(hard & (r1 == r2)))
    n_cross = int(np.sum(hard & (r1 != r2)))
    return n_same, n_cross, np.abs(r1 - r2)[~hard]


def _fmc1_moments(n_pairs: int, d: np.ndarray) -> tuple[float, float]:
    f, g = d.size / n_pairs, d.sum() / n_pairs
    eta = (1.5 * f - 7.0 / 3.0 * g) * 18.0 / 37.0
    delta = (8.0 / 3.0 * g - 5.0 / 6.0 * f) * 18.0 / 37.0
    return max(eta, 0.0), max(delta, 0.0)


def _fmc1_ml(n_hard: int, d: np.ndarray, start: tuple[float, float]) -> tuple[float, float]:
    """Maximum of ``n_hard log H + sum log(eta (1 - d) + delta d)`` over ``eta, delta >= 0``,
    ``H = 1 - 8/3 eta - 7/3 delta >= 0`` the mass of the four hard corners."""
    if d.size == 0:
        return 0.0, 0.0
    a, b = 8.0 / 3.0, 7.0 / 3.0

    def neg_log_lik(p):
        dens = np.maximum(p[0] * (1.0 - d) + p[1] * d, 1e-300)
        H = max(1.0 - a * p[0] - b * p[1], 1e-300)
        return -(n_hard * np.log(H) + np.log(dens).sum()) / (n_hard + d.size)

    # a feasible start, strictly inside: positive density, positive hard mass
    x0 = np.maximum(np.asarray(start, dtype=float), 1e-6)
    mass = a * x0[0] + b * x0[1]
    if mass >= 1.0 - 1e-6:
        x0 *= (1.0 - 1e-3) / mass
    res = minimize(
        neg_log_lik,
        x0,
        method="SLSQP",
        bounds=[(0.0, None), (0.0, None)],
        constraints=[{"type": "ineq", "fun": lambda p: 1.0 - a * p[0] - b * p[1]}],
        options={"ftol": 1e-14, "maxiter": 500},
    )
    eta, delta = (float(max(v, 0.0)) for v in res.x)
    return eta, delta


def fit_law(r, family: str = "fmc1", method: str = "ml") -> FuzzyMarkovLaw:
    """Law of the fuzzy regime chain estimated from a sequence of regimes in ``[0, 1]``.

    Parameters
    ----------
    r : array (N,)
        Regimes; the values exactly 0 and 1 are the hard ones.
    family : {"fmc1", "fmc2"}
        Law of [BDP20] to fit.
    method : {"ml", "moments"}
        For FMC1 only, see the module docstring. FMC2 has a single estimator.
    """
    if family not in LAW_FAMILIES:
        raise ParamError(f"family must be one of {LAW_FAMILIES}, got {family!r}.")
    if method not in ("ml", "moments"):
        raise ParamError(f"method must be 'ml' or 'moments', got {method!r}.")
    r = _regimes(r)
    n_same, n_cross, d = _pair_statistics(r)
    n_pairs, n_hard = r.size - 1, n_same + n_cross
    share_same = n_same / n_hard if n_hard else 0.5

    if family == "fmc1":
        eta, delta = _fmc1_moments(n_pairs, d)
        if method == "ml":
            eta, delta = _fmc1_ml(n_hard, d, (eta, delta))
        hard_mass = max(1.0 - 8.0 / 3.0 * eta - 7.0 / 3.0 * delta, 0.0)
        if hard_mass == 0.0 and (eta > 0 or delta > 0):
            # no hard pair in the sample: bring the mass off the corners back to one
            scale = 1.0 / (8.0 / 3.0 * eta + 7.0 / 3.0 * delta)
            eta, delta = eta * scale * (1.0 - 1e-12), delta * scale * (1.0 - 1e-12)
        return FMC1(alpha=0.5 * hard_mass * share_same, eta=eta, delta=delta)

    width = float(d.max()) if d.size else 0.0
    if width >= 0.5:
        raise ParamError(
            f"a jump of {width:.3g} off the hard corners is not compatible with FMC2, "
            f"whose band is narrower than 1/2; use family='fmc1'."
        )
    off = d.size / n_pairs
    gamma = off / (width * (6.0 - width)) if width > 0 else 0.0
    return FMC2(alpha=0.5 * (1.0 - off) * share_same, gamma=gamma, delta=width)


# ---------------------------------------------------------------------------
# Gaussian laws given the regimes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GaussianFit:
    """Output of ``fit_gaussian``.

    means                 ``(M(0), M(1))``, each of shape (q+s,)
    gammas                ``{(i, j): covariance of (Z_1, Z_2), shape (2(q+s), 2(q+s))}``
    cross_shrink          ``{(i, j): factor in (0, 1]}`` applied to ``Cov(Z_1, Z_2)`` to
                          keep the corner positive definite; 1 when untouched
    cgo_residual          largest entry of the block ``C`` of the corner transitions
                          before the ``cgo`` correction (0 would mean: nothing to correct)
    cross_rank            rank of the design of the four corners (4 when they are all
                          identified; less when the sample never separates them, in
                          which case the least-norm solution is returned)
    n_hard                ``{0: count, 1: count}`` of the hard samples
    n_hard_pairs          ``{(i, j): count}`` of the hard pairs
    """

    means: tuple[np.ndarray, np.ndarray]
    gammas: dict[tuple[int, int], np.ndarray]
    cross_shrink: dict[tuple[int, int], float]
    cgo_residual: float
    cross_rank: int
    n_hard: dict[int, int]
    n_hard_pairs: dict[tuple[int, int], int]


def _sample(r, x, y) -> tuple[np.ndarray, np.ndarray, int, int]:
    r = _regimes(r)
    x = np.asarray(x, dtype=float).reshape(r.size, -1)
    y = np.asarray(y, dtype=float).reshape(r.size, -1)
    return r, np.hstack([x, y]), x.shape[1], y.shape[1]


def _spd(S: np.ndarray) -> np.ndarray:
    """``S`` with its eigenvalues raised to a small positive floor."""
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    floor = _PD_FLOOR * max(float(w.max()), 1e-300)
    return (V * np.maximum(w, floor)) @ V.T


def _joint(S1: np.ndarray, C: np.ndarray, S2: np.ndarray) -> np.ndarray:
    return np.block([[S1, C], [C.T, S2]])


def _is_pd(G: np.ndarray) -> bool:
    w = np.linalg.eigvalsh(G)
    return bool(w.min() > _PD_FLOOR * max(float(w.max()), 1e-300))


def _shrink_cross(S1: np.ndarray, C: np.ndarray, S2: np.ndarray) -> tuple[np.ndarray, float]:
    """Largest ``lam`` in (0, 1] such that ``[[S1, lam C], [lam C', S2]]`` is positive
    definite, by bisection; returns ``(lam C, lam)``."""
    if _is_pd(_joint(S1, C, S2)):
        return C, 1.0
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if _is_pd(_joint(S1, mid * C, S2)):
            lo = mid
        else:
            hi = mid
    return lo * C, lo


def _hard_moments(r: np.ndarray, z: np.ndarray):
    M, S, C = [], [], {}
    for i in (0, 1):
        zi = z[r == i]
        if zi.shape[0] < 2:
            raise ParamError(
                f"method='hard' needs at least two samples in the hard regime {i}; "
                f"{zi.shape[0]} found. Use method='regression'."
            )
        M.append(zi.mean(axis=0))
        S.append((zi - M[i]).T @ (zi - M[i]) / zi.shape[0])
    for i, j in _CORNERS:
        sel = (r[:-1] == i) & (r[1:] == j)
        if sel.sum() < 2:
            raise ParamError(
                f"method='hard' needs at least two hard pairs (r_n, r_n+1) = ({i}, {j}); "
                f"{int(sel.sum())} found. Use method='regression', which also draws on "
                f"the fuzzy samples."
            )
        C[(i, j)] = (z[:-1][sel] - M[i]).T @ (z[1:][sel] - M[j]) / sel.sum()
    return M, S, C, 4


def _regression_moments(r: np.ndarray, z: np.ndarray):
    d = z.shape[1]
    A = np.column_stack([1.0 - r, r])
    if np.linalg.matrix_rank(A) < 2:
        raise ParamError("the regimes take a single value: the two hard laws are not identified.")
    Mfit = np.linalg.lstsq(A, z, rcond=None)[0]  # (2, d)
    e = z - A @ Mfit
    sq = (e[:, :, None] * e[:, None, :]).reshape(r.size, d * d)
    Sfit = np.linalg.lstsq(A, sq, rcond=None)[0].reshape(2, d, d)
    r1, r2 = r[:-1], r[1:]
    W = np.column_stack([(1 - r1) * (1 - r2), (1 - r1) * r2, r1 * (1 - r2), r1 * r2])
    lag = (e[:-1, :, None] * e[1:, None, :]).reshape(r.size - 1, d * d)
    Cfit, _, rank, _ = np.linalg.lstsq(W, lag, rcond=None)
    C = {corner: Cfit[c].reshape(d, d) for c, corner in enumerate(_CORNERS)}
    return [Mfit[0], Mfit[1]], [Sfit[0], Sfit[1]], C, int(rank)


def fit_gaussian(r, x, y, method: str = "regression", cgo: bool = True) -> GaussianFit:
    """Means and corner covariances of a fuzzy model from a labelled sample.

    Parameters
    ----------
    r : array (N,)
        Regimes in ``[0, 1]``.
    x : array (N,) or (N, q)
        Hidden states.
    y : array (N,) or (N, s)
        Observations.
    method : {"regression", "hard"}
        See the module docstring.
    cgo : bool
        Make the four corners satisfy the identity ``C = 0``.
    """
    if method not in GAUSSIAN_METHODS:
        raise ParamError(f"method must be one of {GAUSSIAN_METHODS}, got {method!r}.")
    r, z, q, _ = _sample(r, x, y)
    M, S, C, rank = (_hard_moments if method == "hard" else _regression_moments)(r, z)
    S = [_spd(Si) for Si in S]

    residual = 0.0
    shrink: dict[tuple[int, int], float] = {}
    gammas: dict[tuple[int, int], np.ndarray] = {}
    for i, j in _CORNERS:
        Cij = C[(i, j)].copy()
        # block C of the transition Z_2 = F Z_1 + W, F = Cov(Z_2, Z_1) Var(Z_1)^-1
        F = np.linalg.solve(S[i], Cij).T
        residual = max(residual, float(np.abs(F[q:, :q]).max()))
        if cgo:
            Cij[:q, q:] = S[i][:q, q:] @ np.linalg.solve(S[i][q:, q:], Cij[q:, q:])
        Cij, shrink[(i, j)] = _shrink_cross(S[i], Cij, S[j])
        gammas[(i, j)] = _joint(S[i], Cij, S[j])
    return GaussianFit(
        means=(M[0], M[1]),
        gammas=gammas,
        cross_shrink=shrink,
        cgo_residual=residual,
        cross_rank=rank,
        n_hard={i: int(np.sum(r == i)) for i in (0, 1)},
        n_hard_pairs={(i, j): int(np.sum((r[:-1] == i) & (r[1:] == j))) for i, j in _CORNERS},
    )


# ---------------------------------------------------------------------------
# Whole model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FuzzyFit:
    """Output of ``fit_supervised``: the fitted ``model`` and, in ``gaussian``, the
    diagnostics of its Gaussian part (see ``GaussianFit``)."""

    model: FuzzyCGOMSM
    gaussian: GaussianFit

    @property
    def law(self) -> FuzzyMarkovLaw:
        return self.model.law


def fit_supervised(
    r,
    x,
    y,
    law: str | FuzzyMarkovLaw = "fmc1",
    method: str = "regression",
    interpolation: str = "cgo",
) -> FuzzyFit:
    """A ``FuzzyCGOMSM`` estimated from a labelled sample ``(r, x, y)``.

    Parameters
    ----------
    r, x, y
        Regimes in ``[0, 1]``, hidden states and observations, of the same length.
    law : {"fmc1", "fmc2"} or FuzzyMarkovLaw
        Family of the law of the jumps to fit, or a law to keep as it is.
    method : {"regression", "hard"}
        Estimator of the Gaussian laws, see ``fit_gaussian``.
    interpolation : {"cgo", "bilinear"}
        Interpolation of the model returned. With ``"cgo"`` the corners are made to
        satisfy ``C = 0``, so that the filter of the discretised model is exact.
    """
    if interpolation not in INTERPOLATIONS:
        raise ParamError(f"interpolation must be one of {INTERPOLATIONS}, got {interpolation!r}.")
    fitted_law = law if isinstance(law, FuzzyMarkovLaw) else fit_law(r, family=law)
    gauss = fit_gaussian(r, x, y, method=method, cgo=interpolation == "cgo")
    _, _, q, s = _sample(r, x, y)
    with warnings.catch_warnings():
        # the corners were corrected above, to rounding: nothing left to report
        warnings.simplefilter("ignore", RuntimeWarning)
        model = FuzzyCGOMSM(
            fitted_law, gauss.means, gauss.gammas, q=q, s=s, interpolation=interpolation
        )
    return FuzzyFit(model=model, gaussian=gauss)
