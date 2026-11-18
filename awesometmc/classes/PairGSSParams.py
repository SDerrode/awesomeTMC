#!/usr/bin/env python3
"""
awesometmc/classes/PairGSSParams.py
===================================
Parameters of a Gaussian switching system whose dynamics depend on the regime
**pair** ``(r_n, r_{n+1})``, not on the arrival regime alone:

    Z_{n+1} = F(r_n, r_{n+1}) Z_n + b(r_n, r_{n+1}) + G(r_n, r_{n+1}) u_n + W_{n+1},
    W_{n+1} | (r_n, r_{n+1}) = (j, k)  ~  N(0, Sigma_W(j, k)).

:class:`~awesometmc.classes.GSSParams.GSSParams` is the special case where
nothing depends on the departure regime ``j``; ``GSSParams.as_pair()`` and
``PairGSSParams.from_arrival()`` broadcast it. The regime chain ``P``, the
initial law ``pi0`` and the initial moments ``mu_z0(k)``, ``Sigma_z0(k)`` are
indexed by a single regime, as in ``GSSParams``.

Pair-dependent parameters are the setting of the conditionally Gaussian
observed Markov switching models of Abbassi, Benboudjema, Derrode and
Pieczynski (IEEE TAC 2015): when the observation row does not read the state,
``C(j, k) = 0`` for every pair, ``(R, Y)`` is Markov and the order-2 collapse
of ``GSSFilter(mode="gpb2")`` is the exact filter. A fuzzy switching model
discretised on a grid is of this kind.

Indexing convention: ``F(j, k)`` is the transition **from** regime ``j`` **to**
regime ``k``, like ``P[j, k]``.

Accepted by :class:`~awesometmc.classes.GSSSimulator.GSSSimulator` and by
:class:`~awesometmc.filter.gss_filter.GSSFilter` in mode ``"gpb2"``. The
constant-gain mode ``"ngh_kf"``, the smoothers and the learning modules take
arrival-indexed parameters only; ``to_arrival()`` converts back when the
parameters do not actually depend on the departure regime.
"""

from __future__ import annotations

import itertools

import numpy as np

from awesometmc.classes.FMatrix import FMatrix
from awesometmc.classes.GSSParams import GSSParams
from awesometmc.classes.NoiseCovariance import GSSNoiseCovariance
from awesometmc.utils.exceptions import CovarianceError, ParamError
from awesometmc.utils.matrix_checks import CovarianceMatrix

__all__ = ["PairGSSParams"]


class PairGSSParams:
    """
    Aggregates and validates the parameters of a pair-dependent GSS model.

    Parameters
    ----------
    K, q, s : int
        Number of regimes, dimensions of the hidden state X and of the observation Y.
    P : ndarray (K, K)
        Row-stochastic transition matrix, ``P[j, k] = P(R_{n+1} = k | R_n = j)``.
    F : array-like (K, K, q+s, q+s)
        ``F[j][k]`` is the transition matrix of ``Z`` for the pair ``(j, k)``.
    Sigma_W : array-like (K, K, q+s, q+s)
        ``Sigma_W[j][k]`` is the noise covariance for the pair ``(j, k)``, symmetric
        positive definite. It is validated for every pair, reachable or not.
    pi0 : ndarray (K,) or None
        Initial distribution of R_0; None means the stationary distribution of ``P``.
    mu_z0_list, Sigma_z0_list : lists of K arrays, shapes (q+s, 1) and (q+s, q+s)
        Initial mean and covariance of Z_0 given R_0 = k.
    b : array-like (K, K, q+s, 1) or (K, K, q+s), optional
        Drift bias ``b[j][k]``; zero if omitted.
    G : array-like (K, K, q+s, p), optional
        Exogenous-input gain ``G[j][k]``; the model is autonomous (p = 0) if omitted.

    Raises
    ------
    ParamError
        Inconsistent dimensions, ``P`` not row-stochastic, invalid ``pi0``.
    CovarianceError
        A ``Sigma_W[j][k]`` or ``Sigma_z0[k]`` is not symmetric positive definite.
    """

    def __init__(
        self,
        K: int,
        q: int,
        s: int,
        P: np.ndarray,
        F,
        Sigma_W,
        pi0: np.ndarray | None,
        mu_z0_list: list[np.ndarray],
        Sigma_z0_list: list[np.ndarray],
        b=None,
        G=None,
    ) -> None:
        GSSParams._validate_scalars(K, q, s)
        GSSParams._validate_P(P, K)
        GSSParams._validate_pi0(pi0, K)
        dim = q + s
        GSSParams._validate_initial_conditions(mu_z0_list, Sigma_z0_list, K, dim)

        self._K, self._q, self._s, self._dim_z = int(K), int(q), int(s), dim
        self._P = np.array(P, dtype=float)
        self._pi0 = (
            GSSParams._compute_stationary(self._P) if pi0 is None else np.array(pi0, dtype=float)
        )

        self._F = self._as_pair_array(F, "F", (dim, dim))
        self._SW = self._as_pair_array(Sigma_W, "Sigma_W", (dim, dim))
        if b is None:
            self._b = np.zeros((K, K, dim, 1))
        else:
            b_arr = np.asarray(b, dtype=float)
            if b_arr.shape == (K, K, dim):
                b_arr = b_arr[..., None]
            self._b = self._as_pair_array(b_arr, "b", (dim, 1))
        if G is None:
            self._p = 0
            self._G = np.zeros((K, K, dim, 0))
        else:
            G_arr = np.asarray(G, dtype=float)
            if G_arr.ndim != 4 or G_arr.shape[:3] != (K, K, dim):
                raise ParamError(f"G must have shape ({K}, {K}, {dim}, p), got {G_arr.shape}.")
            self._p = int(G_arr.shape[3])
            self._G = G_arr

        self._chol_W = np.empty_like(self._SW)
        for j, k in itertools.product(range(K), repeat=2):
            self._chol_W[j, k] = self._checked_cholesky(self._SW[j, k], f"Sigma_W({j}, {k})")
        self._mu_z0 = [np.array(m, dtype=float) for m in mu_z0_list]
        self._Sigma_z0 = [np.array(S, dtype=float) for S in Sigma_z0_list]
        self._chol_z0 = [
            self._checked_cholesky(self._Sigma_z0[k], f"Sigma_z0({k})") for k in range(K)
        ]

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _as_pair_array(self, value, name: str, shape: tuple[int, int]) -> np.ndarray:
        arr = np.array(value, dtype=float)
        expected = (self._K, self._K, *shape)
        if arr.shape != expected:
            raise ParamError(f"{name} must have shape {expected}, got {arr.shape}.")
        if not np.all(np.isfinite(arr)):
            raise ParamError(f"{name} contains non-finite values.")
        return arr

    @staticmethod
    def _checked_cholesky(S: np.ndarray, name: str) -> np.ndarray:
        report = CovarianceMatrix(S).check()
        if not report.is_valid:
            raise CovarianceError(
                f"{name} is not symmetric positive definite.\n{report}", matrix_name=name
            )
        return np.linalg.cholesky(S)

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    @classmethod
    def from_arrival(cls, params: GSSParams) -> PairGSSParams:
        """Broadcast arrival-indexed parameters: ``F(j, k) = F(k)`` for every ``j``, etc."""
        K = params.K
        rng = range(K)
        return cls(
            K=K,
            q=params.q,
            s=params.s,
            P=params.P,
            F=[[params.f_matrix.F(k) for k in rng] for _ in rng],
            Sigma_W=[[params.noise_cov.Sigma_W(k) for k in rng] for _ in rng],
            pi0=params.pi0,
            mu_z0_list=[params.mu_z0(k) for k in rng],
            Sigma_z0_list=[params.Sigma_z0(k) for k in rng],
            b=[[params.b(k) for k in rng] for _ in rng],
            G=[[params.G(k) for k in rng] for _ in rng] if params.p > 0 else None,
        )

    def is_arrival_only(self, tol: float = 0.0) -> bool:
        """True when no parameter depends on the departure regime (within ``tol``)."""
        return all(
            float(np.max(np.abs(arr - arr[:1]), initial=0.0)) <= tol
            for arr in (self._F, self._SW, self._b, self._G)
        )

    def _gss_params(self, K, P, pi0, pairs, mu_z0_list, Sigma_z0_list) -> GSSParams:
        """A ``GSSParams`` with one regime per element of ``pairs`` (a list of ``(j, k)``)."""
        q, s = self._q, self._s
        F = [self._F[j, k] for j, k in pairs]
        SW = [self._SW[j, k] for j, k in pairs]
        return GSSParams(
            K=K,
            q=q,
            s=s,
            P=P,
            f_matrix=FMatrix(
                K=K,
                q=q,
                s=s,
                A_list=[f[:q, :q] for f in F],
                B_list=[f[:q, q:] for f in F],
                C_list=[f[q:, :q] for f in F],
                D_list=[f[q:, q:] for f in F],
            ),
            noise_cov=GSSNoiseCovariance(
                K=K,
                q=q,
                s=s,
                Sigma_U_list=[w[:q, :q] for w in SW],
                Delta_list=[w[:q, q:] for w in SW],
                Sigma_V_list=[w[q:, q:] for w in SW],
            ),
            pi0=pi0,
            mu_z0_list=mu_z0_list,
            Sigma_z0_list=Sigma_z0_list,
            b_list=[self._b[j, k] for j, k in pairs],
            G_list=[self._G[j, k] for j, k in pairs] if self._p > 0 else None,
        )

    def to_arrival(self, tol: float = 0.0) -> GSSParams:
        """The equivalent arrival-indexed ``GSSParams``.

        Raises ``ParamError`` if a parameter depends on the departure regime.
        """
        if not self.is_arrival_only(tol):
            raise ParamError(
                "These parameters depend on the departure regime; they have no "
                "arrival-indexed (GSSParams) form. Use augmented() instead."
            )
        K = self._K
        return self._gss_params(
            K, self._P, self._pi0, [(0, k) for k in range(K)], self._mu_z0, self._Sigma_z0
        )

    def augmented(self) -> GSSParams:
        """The same model as an arrival-indexed ``GSSParams`` on the ``K^2`` augmented
        regimes ``S_n = (r_{n-1}, r_n)``, numbered ``j * K + k``.

        ``S_n`` is a Markov chain and the dynamics depend on its arrival value only, so
        every arrival-indexed tool applies, at the cost of ``K^2`` regimes. The regime
        posterior of the original model is recovered from the augmented one by
        ``pi.reshape(K, K).sum(axis=0)``. The initial law of ``Z`` given
        ``S_0 = (j, k)`` is the one given ``r_0 = k``, and the (irrelevant) first
        component ``j`` is taken uniform.
        """
        K = self._K
        pairs = list(itertools.product(range(K), repeat=2))
        P2 = np.zeros((K * K, K * K))
        for j, k in pairs:
            P2[j * K + k, k * K : (k + 1) * K] = self._P[k]
        return self._gss_params(
            K * K,
            P2,
            np.array([self._pi0[k] / K for _, k in pairs]),
            pairs,
            [self._mu_z0[k] for _, k in pairs],
            [self._Sigma_z0[k] for _, k in pairs],
        )

    # ------------------------------------------------------------------
    # Structure tests
    # ------------------------------------------------------------------

    def is_cgo(self, tol: float = 0.0) -> bool:
        """True when ``C(j, k) = 0`` for every pair: the observation never reads the state.

        Then ``(R, Y)`` is a Markov chain and ``GSSFilter(mode="gpb2")`` is the exact
        filter, whatever the initial law.
        """
        q = self._q
        return float(np.max(np.abs(self._F[:, :, q:, :q]))) <= tol

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def K(self) -> int:
        return self._K

    @property
    def q(self) -> int:
        return self._q

    @property
    def s(self) -> int:
        return self._s

    @property
    def dim_z(self) -> int:
        return self._dim_z

    @property
    def p(self) -> int:
        """Exogenous-input dimension p (0 if the model is autonomous)."""
        return self._p

    @property
    def P(self) -> np.ndarray:
        """Row-stochastic transition matrix, shape (K, K)."""
        return self._P

    @property
    def pi0(self) -> np.ndarray:
        """Initial distribution of R_0, shape (K,)."""
        return self._pi0

    def stationary_distribution(self) -> np.ndarray:
        """Stationary distribution of the regime chain, shape (K,)."""
        return GSSParams._compute_stationary(self._P)

    def F(self, j: int, k: int) -> np.ndarray:
        """Transition matrix for the pair ``(r_n, r_{n+1}) = (j, k)``, shape (q+s, q+s)."""
        return self._F[j, k]

    def Sigma_W(self, j: int, k: int) -> np.ndarray:
        """Noise covariance for the pair ``(j, k)``, shape (q+s, q+s)."""
        return self._SW[j, k]

    def chol_W(self, j: int, k: int) -> np.ndarray:
        """Lower Cholesky factor of ``Sigma_W(j, k)``."""
        return self._chol_W[j, k]

    def b(self, j: int, k: int) -> np.ndarray:
        """Drift bias for the pair ``(j, k)``, shape (q+s, 1)."""
        return self._b[j, k]

    def G(self, j: int, k: int) -> np.ndarray:
        """Exogenous-input gain for the pair ``(j, k)``, shape (q+s, p)."""
        return self._G[j, k]

    def mu_z0(self, k: int) -> np.ndarray:
        """Initial mean of Z_0 given R_0 = k, shape (q+s, 1)."""
        return self._mu_z0[k]

    def Sigma_z0(self, k: int) -> np.ndarray:
        """Initial covariance of Z_0 given R_0 = k, shape (q+s, q+s)."""
        return self._Sigma_z0[k]

    def chol_z0(self, k: int) -> np.ndarray:
        """Lower Cholesky factor of ``Sigma_z0(k)``."""
        return self._chol_z0[k]

    def __repr__(self) -> str:
        kind = "arrival-only" if self.is_arrival_only() else "pair-dependent"
        return f"<PairGSSParams(K={self._K}, q={self._q}, s={self._s}, p={self._p}, {kind})>"
