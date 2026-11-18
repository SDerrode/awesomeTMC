#!/usr/bin/env python3
"""
awesometmc/fuzzy/model.py
=========================
Conditionally Gaussian observed fuzzy Markov switching model (CGOFMSM) of
Bouyahia, Derrode and Pieczynski (IEEE TFS 2020) -- [BDP20] -- for a hidden
state ``X`` of dimension ``q`` and an observation ``Y`` of dimension ``s``.

    Z_{n+1} = F(r_n, r_{n+1}) (Z_n - M(r_n)) + M(r_{n+1}) + W_{n+1},
    Z_n = (X_n, Y_n),      W_{n+1} ~ N(0, Q(r_n, r_{n+1})),

with a regime ``r_n`` in ``[0, 1]`` following a fuzzy Markov chain
(:mod:`awesometmc.fuzzy.laws`). The model is specified, as in the paper, by the
means ``M(0)``, ``M(1)`` and by the covariances ``Gamma[i, j]`` of
``(Z_1, Z_2)`` at the four hard corners ``(r_1, r_2) = (i, j)``; means are
interpolated linearly and covariances bilinearly in between.

A fuzzy model is not filtered as such: its integrals over the regimes are
replaced by a quadrature (``discretise``), which turns it into a switching model
with ``K = F + 2`` hard regimes and parameters depending on the regime pair, i.e.
a :class:`~awesometmc.classes.PairGSSParams.PairGSSParams`. The filters of the
library then apply:

    model = FuzzyCGOMSM(FMC1(0.07, 0.16, 0.05), means=(M0, M1), gammas=Gamma)
    disc = model.discretise(F=5)
    filt = disc.filter()                  # GSSFilter, exact mode "cgo"
    for y in observations:
        res = filt.step(y)
        r_hat = disc.mpm(res.pi)          # MPM estimate of the jump

Interpolation and exactness
---------------------------
The filter is exact when the observation does not read the state, i.e. when the
lower-left block ``C`` of ``F`` is zero for every pair of regimes. On the
covariances this reads

    Cov(Y_2, X_1) = Cov(Y_2, Y_1) Var(Y_1)^{-1} Cov(Y_1, X_1).

Bilinear interpolation does not preserve this identity inside the square, even
when the four corners satisfy it. ``interpolation="cgo"`` (default) therefore
recomputes ``Cov(Y_2, X_1)`` from the identity after interpolating, so that
``C = 0`` everywhere and the filter is exact for the discretised model.
``interpolation="bilinear"`` keeps the letter of [BDP20]; the discretised model
then has ``C != 0`` and is filtered approximately, by ``GSSFilter`` in mode
``"gpb2"``.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np

from awesometmc.classes.PairGSSParams import PairGSSParams
from awesometmc.filter.gss_filter import GSSFilter
from awesometmc.fuzzy.laws import FuzzyGrid, FuzzyMarkovLaw, mpm_jumps
from awesometmc.utils.exceptions import CovarianceError, ParamError

__all__ = ["CORNER_TOL", "INTERPOLATIONS", "FuzzyCGOMSM", "FuzzyDiscretisation"]

# Size of the C block at a hard corner above which interpolation="cgo" warns that it
# changes the covariances given.
CORNER_TOL = 1e-8

INTERPOLATIONS = ("cgo", "bilinear")


def _sqrt_psd(Q: np.ndarray, what: str) -> np.ndarray:
    w, V = np.linalg.eigh(Q)
    if w.min() < -1e-9 * max(1.0, w.max()):
        raise CovarianceError(f"{what} is not positive semi-definite (eigenvalues {w}).")
    return V * np.sqrt(np.clip(w, 0.0, None))


@dataclass(frozen=True)
class FuzzyDiscretisation:
    """A fuzzy model on a grid of regime values.

    nodes (K,)    regime values kept, increasing; nodes of zero probability are dropped
    weights (K,)  their quadrature weights
    params        the switching model on these K regimes, a ``PairGSSParams``
    """

    nodes: np.ndarray
    weights: np.ndarray
    params: PairGSSParams

    @property
    def K(self) -> int:
        return int(self.nodes.size)

    def filter(self, mode: str | None = None) -> GSSFilter:
        """A ``GSSFilter`` on the discretised model: mode ``"cgo"`` (exact) when the
        observation does not read the state, ``"gpb2"`` otherwise."""
        return GSSFilter(self.params, mode=mode)

    def mpm(self, pi) -> np.ndarray:
        """MPM estimate of the jump(s) from regime posterior(s) ``pi`` on the nodes
        ([BDP20], Section VI). Returns an array of shape (N,), or (1,) for one posterior."""
        return mpm_jumps(pi, self.nodes)

    def mean_jump(self, pi) -> np.ndarray:
        """Posterior mean of the regime, ``sum_g pi[g] nodes[g]``."""
        return np.atleast_2d(np.asarray(pi, dtype=float)) @ self.nodes


class FuzzyCGOMSM:
    """
    Conditionally Gaussian observed fuzzy Markov switching model.

    Parameters
    ----------
    law : FuzzyMarkovLaw
        Law of the fuzzy regime chain (e.g. ``FMC1``, ``FMC2``).
    means : pair of arrays (q+s,)
        ``(M(0), M(1))``, the means of ``Z = (X, Y)`` in the two hard regimes.
    gammas : dict {(i, j): array (2(q+s), 2(q+s))}
        Covariance of ``(Z_1, Z_2)`` given ``(r_1, r_2) = (i, j)`` for the four hard
        corners, variables ordered ``(x_1, y_1, x_2, y_2)``. The marginal blocks must
        be consistent: ``Var(Z_1 | i)`` the same for both ``j``, ``Var(Z_2 | j)`` the
        same for both ``i``, and ``Var(Z_1 | i) = Var(Z_2 | i)`` (stationarity).
    q, s : int
        Dimensions of the hidden state and of the observation.
    interpolation : {"cgo", "bilinear"}
        See the module docstring. ``"cgo"`` expects hard corners that satisfy the
        identity (``corner_residual()`` close to zero). It also corrects corners that
        do not, such as two of the matrices printed in Section VI of [BDP20], and then
        emits a ``RuntimeWarning`` since it changes the covariances given. Corners far
        from the identity can leave no valid Gaussian model: ``discretise`` and
        ``simulate`` then raise a ``CovarianceError``.
    """

    def __init__(
        self,
        law: FuzzyMarkovLaw,
        means,
        gammas: dict[tuple[int, int], np.ndarray],
        q: int = 1,
        s: int = 1,
        interpolation: str = "cgo",
    ) -> None:
        if interpolation not in INTERPOLATIONS:
            raise ParamError(
                f"interpolation must be one of {INTERPOLATIONS}, got {interpolation!r}."
            )
        if not (isinstance(q, int) and q >= 1 and isinstance(s, int) and s >= 1):
            raise ParamError("q and s must be positive integers.")
        d = q + s
        M0, M1 = (np.asarray(m, dtype=float).ravel() for m in means)
        if M0.shape != (d,) or M1.shape != (d,):
            raise ParamError(f"means must be two vectors of length q + s = {d}.")
        G: dict[tuple[int, int], np.ndarray] = {}
        for corner in ((0, 0), (0, 1), (1, 0), (1, 1)):
            if corner not in gammas:
                raise ParamError(
                    f"gammas must have the four keys (0,0), (0,1), (1,0), (1,1); {corner} is missing."
                )
            g = np.asarray(gammas[corner], dtype=float)
            if g.shape != (2 * d, 2 * d) or not np.allclose(g, g.T):
                raise ParamError(f"gammas[{corner}] must be a symmetric {2 * d}x{2 * d} matrix.")
            G[corner] = 0.5 * (g + g.T)
        for i, j in G:
            if not np.allclose(G[(i, j)][:d, :d], G[(i, 0)][:d, :d]) or not np.allclose(
                G[(i, j)][d:, d:], G[(0, j)][d:, d:]
            ):
                raise ParamError("the marginal blocks of gammas must depend on one regime only.")
        for i in (0, 1):
            if not np.allclose(G[(i, 0)][:d, :d], G[(0, i)][d:, d:]):
                raise ParamError("gammas is not stationary: Var(Z_1 | i) != Var(Z_2 | i).")
        self.law, self.q, self.s, self.interpolation = law, q, s, interpolation
        self._d, self._M, self._G = d, (M0, M1), G
        residual = self.corner_residual()
        if interpolation == "cgo" and residual > CORNER_TOL:
            warnings.warn(
                f"The hard corners do not satisfy Cov(Y_2, X_1) = Cov(Y_2, Y_1) Var(Y_1)^-1 "
                f"Cov(Y_1, X_1): the block C of the transition reaches {residual:.3g}. "
                f"interpolation='cgo' replaces Cov(Y_2, X_1) by the right-hand side, so the "
                f"model filtered is not exactly the one given; use interpolation='bilinear' "
                f"to keep the covariances as they are.",
                RuntimeWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Gaussian laws given the regimes
    # ------------------------------------------------------------------

    def mean(self, r: float) -> np.ndarray:
        """``E[Z_n | r_n = r]``, shape (q+s,): linear interpolation of the hard means."""
        return (1.0 - r) * self._M[0] + r * self._M[1]

    def cov(self, r: float) -> np.ndarray:
        """``Var(Z_n | r_n = r)``, shape (q+s, q+s)."""
        d = self._d
        return (1.0 - r) * self._G[(0, 0)][:d, :d] + r * self._G[(1, 0)][:d, :d]

    def gamma(self, r1: float, r2: float) -> np.ndarray:
        """Covariance of ``(Z_1, Z_2)`` given ``(r_1, r_2)``, shape (2(q+s), 2(q+s))."""
        G = (
            (1.0 - r1) * (1.0 - r2) * self._G[(0, 0)]
            + r1 * r2 * self._G[(1, 1)]
            + r1 * (1.0 - r2) * self._G[(1, 0)]
            + r2 * (1.0 - r1) * self._G[(0, 1)]
        )
        if self.interpolation == "cgo":
            q, d = self.q, self._d
            x1, y1, y2 = slice(0, q), slice(q, d), slice(d + q, 2 * d)
            G = G.copy()
            G[y2, x1] = G[y2, y1] @ np.linalg.solve(G[y1, y1], G[y1, x1])
            G[x1, y2] = G[y2, x1].T
        return G

    def transition(self, r1: float, r2: float) -> tuple[np.ndarray, np.ndarray]:
        """``(F, Q)`` for the pair ``(r_n, r_{n+1}) = (r1, r2)``: the regression of
        ``Z_2`` on ``Z_1`` and the covariance of its residual."""
        d = self._d
        G = self.gamma(r1, r2)
        F = np.linalg.solve(G[:d, :d], G[:d, d:]).T
        if self.interpolation == "cgo":
            # zero by construction of gamma(); remove the rounding residue so that the
            # discretised model is recognised as exactly CGO
            F[self.q :, : self.q] = 0.0
        Q = G[d:, d:] - F @ G[:d, d:]
        return F, 0.5 * (Q + Q.T)

    def corner_residual(self) -> float:
        """Largest entry, in absolute value, of the block ``C`` at the four hard corners,
        for the covariances as given (before any correction). Zero when the corners
        satisfy the identity of the module docstring."""
        d, q = self._d, self.q
        worst = 0.0
        for G in self._G.values():
            F = np.linalg.solve(G[:d, :d], G[:d, d:]).T
            worst = max(worst, float(np.max(np.abs(F[q:, :q]))))
        return worst

    def c_block(self, r1: float, r2: float) -> np.ndarray:
        """Block ``C`` of ``F(r1, r2)``, shape (s, q): how ``Y_{n+1}`` reads ``X_n``.
        Zero when the filter is exact."""
        return self.transition(r1, r2)[0][self.q :, : self.q]

    # ------------------------------------------------------------------
    # Simulation (continuous regimes)
    # ------------------------------------------------------------------

    def simulate(
        self, N: int, seed: int | np.random.Generator | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A sample ``(r, x, y)`` of length ``N``: ``r`` (N,) in ``[0, 1]``, ``x`` (N, q),
        ``y`` (N, s). The regimes are drawn from the law itself, not from a grid."""
        rng: np.random.Generator = (
            seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        )
        d = self._d
        r = self.law.sample(N, rng)
        z = np.empty((N, d))
        z[0] = self.mean(r[0]) + _sqrt_psd(self.cov(r[0]), "Var(Z_1)") @ rng.standard_normal(d)
        for n in range(1, N):
            F, Q = self.transition(r[n - 1], r[n])
            try:
                root = _sqrt_psd(Q, f"Q({r[n - 1]:.3f}, {r[n]:.3f})")
            except CovarianceError as err:
                raise CovarianceError(self._invalid_message(str(err))) from err
            z[n] = (
                self.mean(r[n])
                + F @ (z[n - 1] - self.mean(r[n - 1]))
                + root @ rng.standard_normal(d)
            )
        return r, z[:, : self.q], z[:, self.q :]

    # ------------------------------------------------------------------
    # Discretisation
    # ------------------------------------------------------------------

    def _on_nodes(self, nodes, weights, pi, P) -> FuzzyDiscretisation:
        K, d = len(nodes), self._d
        F, Q, b = np.empty((K, K, d, d)), np.empty((K, K, d, d)), np.empty((K, K, d, 1))
        for j, k in itertools.product(range(K), repeat=2):
            F[j, k], Q[j, k] = self.transition(nodes[j], nodes[k])
            b[j, k, :, 0] = self.mean(nodes[k]) - F[j, k] @ self.mean(nodes[j])
        try:
            params = PairGSSParams(
                K=K,
                q=self.q,
                s=self.s,
                P=P,
                F=F,
                Sigma_W=Q,
                pi0=pi,
                mu_z0_list=[self.mean(r).reshape(d, 1) for r in nodes],
                Sigma_z0_list=[self.cov(r) for r in nodes],
                b=b,
            )
        except CovarianceError as err:
            raise CovarianceError(self._invalid_message(str(err))) from err
        return FuzzyDiscretisation(np.asarray(nodes, float), np.asarray(weights, float), params)

    def _invalid_message(self, detail: str) -> str:
        msg = f"The covariances do not define a valid Gaussian model (interpolation={self.interpolation!r})."
        if self.interpolation == "cgo" and self.corner_residual() > CORNER_TOL:
            msg += (
                f" The hard corners are far from the structure interpolation='cgo' expects "
                f"(corner_residual() = {self.corner_residual():.3g}): give corners that satisfy "
                f"Cov(Y_2, X_1) = Cov(Y_2, Y_1) Var(Y_1)^-1 Cov(Y_1, X_1), or use "
                f"interpolation='bilinear'."
            )
        return f"{msg}\n{detail}"

    def discretise(self, F: int = 5, grid: FuzzyGrid | None = None) -> FuzzyDiscretisation:
        """The model on a grid: ``F`` interior steps of the midpoint rule, or any ``grid``.

        Nodes of zero stationary probability are dropped (the interior nodes of a
        purely hard law, for instance), so the result may have fewer than ``F + 2``
        regimes.
        """
        grid = FuzzyGrid.midpoint(F) if grid is None else grid
        pi, P = self.law.discretise(grid)
        keep = pi > 0.0
        if keep.sum() < 2:
            raise ParamError("fewer than two regimes have positive probability on this grid.")
        pi, P = pi[keep] / pi[keep].sum(), P[np.ix_(keep, keep)]
        P = P / P.sum(axis=1, keepdims=True)
        return self._on_nodes(grid.nodes[keep], grid.weights[keep], pi, P)

    def harden(self) -> FuzzyDiscretisation:
        """The 2-regime "hard" model: hard Gaussian laws, and the chain obtained by
        integrating the fuzzy law over the four quadrants ([BDP20], Section VI)."""
        pi, P = self.law.harden()
        return self._on_nodes(np.array([0.0, 1.0]), np.ones(2), pi, P)

    # ------------------------------------------------------------------
    # Filter with known regimes
    # ------------------------------------------------------------------

    def known_jumps_filter(self, r, ys) -> tuple[np.ndarray, np.ndarray]:
        """``E[X_n | r_1^n, y_1^n]`` (N, q) and its covariance (N, q, q): the Kalman
        filter of the model when the regimes are given ("KJ" in [BDP20])."""
        q = self.q
        r = np.asarray(r, dtype=float).ravel()
        ys = np.asarray(ys, dtype=float).reshape(r.size, self.s)
        E_x, Var_x = np.empty((r.size, q)), np.empty((r.size, q, q))
        M, S = self.mean(r[0]), self.cov(r[0])
        gain = np.linalg.solve(S[q:, q:], S[q:, :q]).T
        m, V = M[:q] + gain @ (ys[0] - M[q:]), S[:q, :q] - gain @ S[q:, :q]
        E_x[0], Var_x[0] = m, V
        for n in range(1, r.size):
            F, Q = self.transition(r[n - 1], r[n])
            M_prev, M_new = self.mean(r[n - 1]), self.mean(r[n])
            pred = F @ np.concatenate([m - M_prev[:q], ys[n - 1] - M_prev[q:]]) + M_new
            C = F[:, :q] @ V @ F[:, :q].T + Q
            gain = np.linalg.solve(C[q:, q:], C[q:, :q]).T
            m, V = pred[:q] + gain @ (ys[n] - pred[q:]), C[:q, :q] - gain @ C[q:, :q]
            V = 0.5 * (V + V.T)
            E_x[n], Var_x[n] = m, V
        return E_x, Var_x

    def __repr__(self) -> str:
        return (
            f"<FuzzyCGOMSM(law={type(self.law).__name__}, q={self.q}, s={self.s}, "
            f"interpolation={self.interpolation!r})>"
        )
