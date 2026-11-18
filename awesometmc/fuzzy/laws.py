#!/usr/bin/env python3
"""
awesometmc/fuzzy/laws.py
========================
Fuzzy Markov chains: regimes ``R_n`` in ``[0, 1]`` whose laws are densities with
respect to ``nu = delta_0 + delta_1 + Lebesgue(0, 1)``. The two "hard" regimes 0
and 1 carry positive probability; the open interval is "fuzzy".

``FuzzyMarkovLaw``  stationary law of ``(R_1, R_2)``: densities, sampler, discretisation
``FMC1``, ``FMC2``  the two laws of Bouyahia, Derrode and Pieczynski (IEEE TFS 2020),
                    Section V-A -- cited [BDP20]
``FuzzyGrid``       quadrature of ``nu``: the two atoms and interior nodes
``mpm_jumps``       MPM restoration of the jumps from a posterior on a grid ([BDP20], Section VI)

A custom law only has to subclass ``FuzzyMarkovLaw`` and give its joint density
off the four corners, its marginal density on the open interval, the mass of a
hard regime and a sampler of the fuzzy values.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

__all__ = ["FMC1", "FMC2", "FuzzyGrid", "FuzzyMarkovLaw", "mpm_jumps"]


def _is_hard(r: np.ndarray) -> np.ndarray:
    return (r == 0.0) | (r == 1.0)


class FuzzyMarkovLaw(abc.ABC):
    """Stationary law of ``(R_1, R_2)`` on ``[0, 1]^2``, as a density w.r.t. ``nu x nu``.

    The four corners carry masses ``alpha`` (0,0 and 1,1) and ``beta`` (0,1 and 1,0);
    everywhere else (interior and the four open sides) the density is
    ``_interior_joint``. Both laws of [BDP20] are symmetric, so the chain is
    stationary and reversible.
    """

    alpha: float
    beta: float

    @abc.abstractmethod
    def _interior_joint(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """Joint density off the four corners."""

    @abc.abstractmethod
    def _interior_marginal(self, r: np.ndarray) -> np.ndarray:
        """Marginal density for ``r`` in the open interval."""

    @property
    @abc.abstractmethod
    def hard_mass(self) -> float:
        """``p(0) = p(1)``, the mass of each hard regime."""

    @abc.abstractmethod
    def _draw_interior(self, rng: np.random.Generator, prev: float | None) -> float:
        """Draw a fuzzy value: from the marginal if ``prev`` is None, else given ``prev``."""

    # -- densities ---------------------------------------------------------
    def joint(self, r1, r2) -> np.ndarray:
        """``p(r1, r2)`` w.r.t. ``nu x nu`` (broadcasts)."""
        r1, r2 = np.broadcast_arrays(np.asarray(r1, float), np.asarray(r2, float))
        corner = _is_hard(r1) & _is_hard(r2)
        return np.where(
            corner, np.where(r1 == r2, self.alpha, self.beta), self._interior_joint(r1, r2)
        )

    def marginal(self, r) -> np.ndarray:
        """``p(r)`` w.r.t. ``nu``."""
        r = np.asarray(r, float)
        return np.where(_is_hard(r), self.hard_mass, self._interior_marginal(r))

    def conditional(self, r2, r1) -> np.ndarray:
        """``p(r2 | r1)`` w.r.t. ``nu``."""
        return self.joint(r1, r2) / self.marginal(r1)

    @property
    def p_hard(self) -> float:
        """Limit proportion of hard regimes in a sample, ``p(0) + p(1)``."""
        return 2.0 * self.hard_mass

    @property
    def p_fuzzy(self) -> float:
        return 1.0 - self.p_hard

    # -- sampling ----------------------------------------------------------
    def _draw(self, rng: np.random.Generator, prev: float | None) -> float:
        if prev is None:
            m0 = m1 = self.hard_mass
        else:
            d = float(self.marginal(prev))
            m0, m1 = float(self.joint(prev, 0.0)) / d, float(self.joint(prev, 1.0)) / d
        u = rng.random()
        if u < m0:
            return 0.0
        if u < m0 + m1:
            return 1.0
        return self._draw_interior(rng, prev)

    def sample(self, N: int, rng: np.random.Generator) -> np.ndarray:
        """A stationary trajectory ``r_1, ..., r_N`` (hard / fuzzy draw first, as in Section VI)."""
        r = np.empty(N)
        prev: float | None = None
        for n in range(N):
            prev = self._draw(rng, prev)
            r[n] = prev
        return r

    def _rejection(self, rng: np.random.Generator, density, bound: float) -> float:
        while True:
            r = rng.random()
            if r > 0.0 and rng.random() * bound <= float(density(np.asarray(r))):
                return r

    # -- discretisation ----------------------------------------------------
    def discretise(self, grid: FuzzyGrid) -> tuple[np.ndarray, np.ndarray]:
        """``(pi, P)`` of the chain on ``grid``: quadrature of the joint law, renormalised.

        ``W[g, h] = w_g w_h p(rho_g, rho_h)`` is normalised to sum to one, then
        ``pi = W 1`` and ``P = W / pi``. ``W`` is symmetric, so ``pi`` is the exact
        stationary law of ``P``.
        """
        w, rho = grid.weights, grid.nodes
        W = w[:, None] * w[None, :] * self.joint(rho[:, None], rho[None, :])
        W = W / W.sum()
        pi = W.sum(axis=1)
        P = np.divide(W, pi[:, None], out=np.zeros_like(W), where=pi[:, None] > 0)
        return pi, P

    def harden(self, n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
        """``(pi, P)`` of the 2-regime chain obtained by integrating the law over the four
        quadrants of ``[0, 1]^2`` (Section VI, "we harden the Markov laws")."""
        grid = FuzzyGrid.midpoint(2 * (n // 2))  # even: no node on 1/2
        w, rho = grid.weights, grid.nodes
        W = w[:, None] * w[None, :] * self.joint(rho[:, None], rho[None, :])
        up = rho >= 0.5
        J = np.array([[W[~up][:, ~up].sum(), W[~up][:, up].sum()],
                      [W[up][:, ~up].sum(), W[up][:, up].sum()]])  # fmt: skip
        J = J / J.sum()
        pi = J.sum(axis=1)
        return pi, J / pi[:, None]


class FMC1(FuzzyMarkovLaw):
    """First law of [BDP20]: ``p(r1, r2) = eta + (delta - eta) |r1 - r2|`` off the corners.

    ``beta`` follows from the normalisation, eq. (serie2_beta).
    """

    def __init__(self, alpha: float, eta: float, delta: float) -> None:
        self.alpha, self.eta, self.delta = float(alpha), float(eta), float(delta)
        self.beta = (
            1.0 - 2.5 * (self.delta + self.eta) + (self.delta - self.eta) / 6.0
        ) / 2.0 - self.alpha
        if min(self.alpha, self.eta, self.delta) < 0 or self.beta < -1e-12:
            raise ValueError(f"FMC1: negative mass (beta = {self.beta:.4g})")
        self.beta = max(self.beta, 0.0)

    @property
    def hard_mass(self) -> float:
        return self.alpha + self.beta + 0.5 * (self.delta + self.eta)

    def _interior_joint(self, r1, r2):
        return self.eta + (self.delta - self.eta) * np.abs(r1 - r2)

    def _interior_marginal(self, r):
        return 1.5 * (self.delta + self.eta) + (self.delta - self.eta) * (r * r - r)

    def _draw_interior(self, rng, prev):
        if prev is None:
            s = 1.5 * (self.delta + self.eta)
            bound = max(s, s - 0.25 * (self.delta - self.eta))
            return self._rejection(rng, self._interior_marginal, bound)
        return self._rejection(
            rng, lambda r: self._interior_joint(np.asarray(prev), r), max(self.eta, self.delta)
        )


class FMC2(FuzzyMarkovLaw):
    """Second law of [BDP20]: constant density ``gamma`` on the band ``|r2 - r1| <= delta``."""

    def __init__(self, alpha: float, gamma: float, delta: float) -> None:
        self.alpha, self.gamma, self.delta = float(alpha), float(gamma), float(delta)
        if not 0.0 <= self.delta < 0.5:
            raise ValueError("FMC2: delta must lie in [0, 1/2)")
        self.beta = (1.0 - self.gamma * self.delta * (6.0 - self.delta)) / 2.0 - self.alpha
        if min(self.alpha, self.gamma) < 0 or self.beta < -1e-12:
            raise ValueError(f"FMC2: negative mass (beta = {self.beta:.4g})")
        self.beta = max(self.beta, 0.0)

    @property
    def hard_mass(self) -> float:
        return self.alpha + self.beta + self.gamma * self.delta

    def _interior_joint(self, r1, r2):
        # the band is closed, |r2 - r1| <= delta; the slack keeps a node sitting exactly on
        # its edge inside, whatever the rounding of the difference
        return np.where(np.abs(r2 - r1) <= self.delta + 1e-12, self.gamma, 0.0)

    def _interior_marginal(self, r):
        d, g = self.delta, self.gamma
        return np.where(r <= d, g * (d + r + 1.0), np.where(r < 1.0 - d, 2.0 * g * d, g * (2.0 + d - r)))  # fmt: skip

    def _draw_interior(self, rng, prev):
        if prev is None:
            bound = self.gamma * (1.0 + 2.0 * self.delta)
            return self._rejection(rng, self._interior_marginal, bound)
        lo, hi = max(0.0, prev - self.delta), min(1.0, prev + self.delta)
        while True:
            r = lo + (hi - lo) * rng.random()
            if 0.0 < r < 1.0:
                return r


@dataclass(frozen=True)
class FuzzyGrid:
    """Quadrature of ``nu``: nodes in ``[0, 1]`` and their weights.

    The atoms 0 and 1 carry weight 1; the interior nodes carry the weights of a
    quadrature rule on ``(0, 1)``. ``midpoint(F)`` is the default rule; any other
    rule is given by its interior nodes and weights (``custom``).

    NOTE [BDP20]: the paper only says that the integrals over (0, 1) use ``F``
    discrete steps. The midpoint rule (nodes ``(k + 1/2) / F``, weights ``1 / F``)
    never evaluates the interior density at 0 or 1.
    """

    nodes: np.ndarray
    weights: np.ndarray

    @classmethod
    def midpoint(cls, F: int) -> FuzzyGrid:
        """The atoms 0 and 1 and ``F`` equally spaced interior nodes."""
        if F < 0:
            raise ValueError("F must be a non-negative integer")
        inner = (np.arange(F) + 0.5) / F if F > 0 else np.empty(0)
        return cls(
            nodes=np.concatenate(([0.0], inner, [1.0])),
            weights=np.concatenate(([1.0], np.full(F, 1.0 / max(F, 1)), [1.0])),
        )

    @classmethod
    def custom(cls, inner_nodes, inner_weights) -> FuzzyGrid:
        """The atoms 0 and 1 and the given quadrature rule on the open interval."""
        nodes = np.asarray(inner_nodes, dtype=float).ravel()
        weights = np.asarray(inner_weights, dtype=float).ravel()
        if nodes.shape != weights.shape:
            raise ValueError("inner_nodes and inner_weights must have the same length")
        if nodes.size and (nodes.min() <= 0.0 or nodes.max() >= 1.0 or np.any(np.diff(nodes) <= 0)):
            raise ValueError(
                "interior nodes must be increasing and lie in the open interval (0, 1)"
            )
        if np.any(weights <= 0.0):
            raise ValueError("quadrature weights must be positive")
        return cls(
            nodes=np.concatenate(([0.0], nodes, [1.0])),
            weights=np.concatenate(([1.0], weights, [1.0])),
        )

    @property
    def K(self) -> int:
        return int(self.nodes.size)

    @property
    def F(self) -> int:
        """Number of interior nodes."""
        return self.K - 2


def mpm_jumps(post: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """MPM restoration of the jumps, as stated in Section VI of [BDP20].

    ``post`` (N, K) or (K,) is a posterior on the grid ``nodes`` (K,). The estimate is
    hard if ``P(0 | y) + P(1 | y) > 0.5`` (then the more probable of 0 and 1), otherwise
    it is the fuzzy node of largest posterior.
    """
    post = np.atleast_2d(np.asarray(post, dtype=float))
    nodes = np.asarray(nodes, dtype=float)
    p0, p1 = post[:, nodes == 0.0].sum(axis=1), post[:, nodes == 1.0].sum(axis=1)
    hard_value = np.where(p0 > p1, 0.0, 1.0)
    fuzzy = ~_is_hard(nodes)
    if not fuzzy.any():
        return hard_value
    fuzzy_value = nodes[fuzzy][np.argmax(post[:, fuzzy], axis=1)]
    return np.where(p0 + p1 > 0.5, hard_value, fuzzy_value)
