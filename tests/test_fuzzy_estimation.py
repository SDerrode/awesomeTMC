#!/usr/bin/env python3
"""
tests/test_fuzzy_estimation.py
==============================
Supervised estimation of a fuzzy switching model from a sample whose regimes are
known (``awesometmc.fuzzy.estimation``), the setting of Section VII of the 2020 paper.

The estimators are checked on samples drawn from known models, and on a "daily
cycle" pseudo ground truth of the kind used in the paper, where regimes move from 0
to 1 through fuzzy values only.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from awesometmc.fuzzy import (
    FMC1,
    FMC2,
    FuzzyCGOMSM,
    FuzzyFit,
    fit_gaussian,
    fit_law,
    fit_supervised,
)
from awesometmc.fuzzy.estimation import _fmc1_moments, _pair_statistics, _shrink_cross
from awesometmc.utils.exceptions import ParamError
from tests.test_fuzzy_layer import _paper_model, _random_model

N_LAW = 60_000
N_GAUSS = 40_000


def _daily_cycle(days=40, per_day=24):
    """Hard 0 at night, hard 1 in the afternoon, linear ramps in between."""
    t = np.arange(days * per_day) % per_day
    r = np.interp(t, [0, 1, 5, 13, 17, 24], [0.25, 0.0, 0.0, 1.0, 1.0, 0.25])
    r[(t >= 1) & (t <= 5)] = 0.0
    r[(t >= 13) & (t <= 17)] = 1.0
    return r


# --------------------------------------------------------------------------
# Law of the jumps
# --------------------------------------------------------------------------
@pytest.mark.parametrize("truth", [(0.07, 0.16, 0.05), (0.10, 0.21, 0.076), (0.20, 0.05, 0.12)])
@pytest.mark.parametrize("method", ["ml", "moments"])
def test_fmc1_is_recovered(truth, method):
    law = FMC1(*truth)
    r = law.sample(N_LAW, np.random.default_rng(1))
    est = fit_law(r, "fmc1", method=method)
    assert isinstance(est, FMC1)
    assert est.alpha == pytest.approx(law.alpha, abs=0.01)
    assert est.beta == pytest.approx(law.beta, abs=0.01)
    assert est.eta == pytest.approx(law.eta, abs=0.015)
    assert est.delta == pytest.approx(law.delta, abs=0.015)


def test_fmc1_moment_equations():
    """The two moments used in closed form, against a numerical integration."""
    law = FMC1(0.07, 0.16, 0.05)
    u = (np.arange(4000) + 0.5) / 4000
    g1, g2 = np.meshgrid(u, u, indexing="ij")
    interior = law._interior_joint(g1, g2)
    edge = law._interior_joint(np.zeros_like(u), u)
    mass = interior.mean() + 4 * edge.mean()
    first = (np.abs(g1 - g2) * interior).mean() + 4 * (u * edge).mean()
    assert mass == pytest.approx(8 / 3 * law.eta + 7 / 3 * law.delta, abs=1e-6)
    assert first == pytest.approx(5 / 6 * law.eta + 3 / 2 * law.delta, abs=1e-6)
    assert mass + 2 * (law.alpha + law.beta) == pytest.approx(1.0, abs=1e-6)


def test_fmc1_likelihood_is_not_below_the_moment_estimate():
    law = FMC1(0.07, 0.16, 0.05)
    r = law.sample(5000, np.random.default_rng(2))
    n_same, n_cross, d = _pair_statistics(r)

    def log_lik(eta, delta):
        H = 1 - 8 / 3 * eta - 7 / 3 * delta
        return (n_same + n_cross) * np.log(H) + np.log(eta * (1 - d) + delta * d).sum()

    ml = fit_law(r, "fmc1", method="ml")
    mom = _fmc1_moments(r.size - 1, d)
    assert log_lik(ml.eta, ml.delta) >= log_lik(*mom) - 1e-6


def test_fmc2_is_recovered():
    law = FMC2(0.15, 0.60, 0.20)
    r = law.sample(N_LAW, np.random.default_rng(3))
    est = fit_law(r, "fmc2")
    assert isinstance(est, FMC2)
    assert est.delta == pytest.approx(0.20, abs=2e-3) and est.delta <= 0.20 + 1e-12
    assert est.gamma == pytest.approx(0.60, abs=0.02)
    assert est.alpha == pytest.approx(0.15, abs=0.01)


def test_a_purely_hard_chain_gives_a_purely_hard_law():
    rng = np.random.default_rng(4)
    r = np.zeros(20_000)
    for n in range(1, r.size):
        r[n] = r[n - 1] if rng.random() < 0.8 else 1.0 - r[n - 1]
    for family in ("fmc1", "fmc2"):
        est = fit_law(r, family)
        assert est.p_fuzzy == pytest.approx(0.0, abs=1e-12)
        assert est.alpha == pytest.approx(0.4, abs=0.01)
        assert est.beta == pytest.approx(0.1, abs=0.01)


def test_law_validation():
    with pytest.raises(ParamError, match=r"\[0, 1\]"):
        fit_law([0.0, 1.2, 0.5])
    with pytest.raises(ParamError, match="at least two"):
        fit_law([0.5])
    with pytest.raises(ParamError, match="family"):
        fit_law([0.0, 1.0], family="fmc3")
    with pytest.raises(ParamError, match="method"):
        fit_law([0.0, 1.0], method="bayes")
    with pytest.raises(ParamError, match="not compatible with FMC2"):
        fit_law([0.1, 0.9, 0.2], family="fmc2")


# --------------------------------------------------------------------------
# Gaussian laws
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["regression", "hard"])
def test_gaussian_laws_are_recovered_scalar(method):
    model = _paper_model(FMC1(0.07, 0.16, 0.05))
    r, x, y = model.simulate(N=N_GAUSS, seed=5)
    fit = fit_gaussian(r, x, y, method=method)
    assert np.allclose(fit.means[0], model.mean(0.0), atol=0.04)
    assert np.allclose(fit.means[1], model.mean(1.0), atol=0.04)
    for corner, G in fit.gammas.items():
        assert np.allclose(G, model.gamma(*map(float, corner)), atol=0.05)
    assert fit.cross_rank == 4
    assert all(lam == 1.0 for lam in fit.cross_shrink.values())
    assert sum(fit.n_hard_pairs.values()) > 0 and fit.n_hard[0] > 0 and fit.n_hard[1] > 0


def test_gaussian_laws_are_recovered_in_higher_dimensions():
    model = _random_model(seed=31, q=2, s=1)
    r, x, y = model.simulate(N=N_GAUSS, seed=6)
    fit = fit_gaussian(r, x, y)
    assert fit.means[0].shape == (3,)
    for corner, G in fit.gammas.items():
        assert G.shape == (6, 6)
        assert np.allclose(G, model.gamma(*map(float, corner)), atol=0.08)


def test_bilinear_model_is_recovered_without_the_cgo_correction():
    model = _paper_model(FMC1(0.07, 0.16, 0.05), interpolation="bilinear")
    r, x, y = model.simulate(N=N_GAUSS, seed=7)
    fit = fit_gaussian(r, x, y, cgo=False)
    for corner, G in fit.gammas.items():
        assert np.allclose(G, model.gamma(*map(float, corner)), atol=0.05)
    # the printed matrices of the paper do read the state: the estimate sees it
    assert fit.cgo_residual > 0.08


def test_regression_equals_the_empirical_estimators_on_hard_regimes():
    model = _paper_model(FMC1(0.35, 0.0, 0.0))  # no fuzzy mass
    r, x, y = model.simulate(N=5000, seed=8)
    assert set(np.unique(r)) == {0.0, 1.0}
    a, b = fit_gaussian(r, x, y, method="regression"), fit_gaussian(r, x, y, method="hard")
    assert np.allclose(a.means, b.means, atol=1e-10)
    for corner in a.gammas:
        assert np.allclose(a.gammas[corner], b.gammas[corner], atol=1e-10)


def test_estimated_corners_satisfy_the_cgo_identity():
    model = _paper_model(FMC1(0.07, 0.16, 0.05), interpolation="bilinear")
    r, x, y = model.simulate(N=4000, seed=9)
    fit = fit_gaussian(r, x, y, cgo=True)
    assert fit.cgo_residual > 1e-3  # there was something to correct
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "corners do not satisfy" warning
        fitted = FuzzyCGOMSM(FMC1(0.07, 0.16, 0.05), fit.means, fit.gammas)
    assert fitted.corner_residual() < 1e-10


def test_shrinking_restores_a_positive_definite_corner():
    S = np.eye(2)
    C = np.array([[1.2, 0.0], [0.0, 0.3]])  # a "correlation" above one
    shrunk, lam = _shrink_cross(S, C, S)
    assert 0.0 < lam < 1.0 and lam == pytest.approx(1 / 1.2, abs=1e-6)
    assert np.linalg.eigvalsh(np.block([[S, shrunk], [shrunk.T, S]])).min() > 0
    same, one = _shrink_cross(S, 0.5 * C, S)
    assert one == 1.0 and np.array_equal(same, 0.5 * C)


def test_gaussian_validation():
    r = np.array([0.0, 0.5, 1.0, 1.0])
    z = np.arange(4.0)
    with pytest.raises(ParamError, match="method"):
        fit_gaussian(r, z, z, method="em")
    with pytest.raises(ParamError, match="single value"):
        fit_gaussian(np.full(4, 0.5), z, z)
    with pytest.raises(ParamError, match="hard regime 0"):
        fit_gaussian(r, z, z, method="hard")


# --------------------------------------------------------------------------
# Whole model, and the protocol of Section VII
# --------------------------------------------------------------------------
def test_fit_supervised_returns_a_model_close_to_the_truth():
    truth = _paper_model(FMC1(0.07, 0.16, 0.05))
    r, x, y = truth.simulate(N=N_GAUSS, seed=10)
    fit = fit_supervised(r, x, y, law="fmc1")
    assert isinstance(fit, FuzzyFit) and isinstance(fit.law, FMC1)
    assert fit.model.q == 1 and fit.model.s == 1 and fit.model.interpolation == "cgo"
    # the fitted model filters fresh data almost as well as the true one
    _, x_new, y_new = truth.simulate(N=3000, seed=11)
    errors = []
    for model in (truth, fit.model):
        filt = model.discretise(F=3).filter()
        est = np.array([filt.step(v).E_x.ravel() for v in y_new]).ravel()
        errors.append(np.mean((est - np.ravel(x_new)) ** 2))
    assert errors[1] == pytest.approx(errors[0], rel=0.03)


def test_fit_supervised_keeps_a_law_given():
    truth = _paper_model(FMC1(0.07, 0.16, 0.05))
    r, x, y = truth.simulate(N=2000, seed=12)
    law = FMC2(0.15, 0.60, 0.20)
    assert fit_supervised(r, x, y, law=law).law is law
    assert fit_supervised(r, x, y, interpolation="bilinear").model.interpolation == "bilinear"
    with pytest.raises(ParamError, match="interpolation"):
        fit_supervised(r, x, y, interpolation="linear")


def test_daily_cycle_pseudo_ground_truth():
    """Regimes of the kind laid over the data in Section VII: the hard pairs (0, 1) and
    (1, 0) never occur, so the empirical estimators of the paper cannot give the four
    corners, while the regression can."""
    r = _daily_cycle()
    rng = np.random.default_rng(13)
    x = 60 + 25 * r + rng.normal(0, 2, r.size)  # "temperature"
    y = 100 + 80 * r + 1.5 * (x - 60 - 25 * r) + rng.normal(0, 4, r.size)  # "power"
    assert not np.any((r[:-1] == 0) & (r[1:] == 1))

    with pytest.raises(ParamError, match=r"\(0, 1\)"):
        fit_supervised(r, x, y, method="hard")

    fit = fit_supervised(r, x, y)
    assert fit.gaussian.n_hard_pairs[(0, 1)] == 0 and fit.gaussian.n_hard_pairs[(1, 0)] == 0
    assert fit.gaussian.means[0] == pytest.approx([60, 100], abs=2.0)
    assert fit.gaussian.means[1] == pytest.approx([85, 180], abs=2.0)

    disc = fit.model.discretise(F=5)
    E_f = np.array([o.E_x.ravel() for o in map(disc.filter().step, y)]).ravel()
    E_s, _, gamma, _ = disc.smoother(y)
    mse_f, mse_s = np.mean((E_f - x) ** 2), np.mean((E_s.ravel() - x) ** 2)
    # the state is recovered; smoothing brings little here, the regimes being clear-cut
    assert mse_f < np.var(x) / 10 and mse_s < 1.01 * mse_f
    assert np.mean((disc.mpm(gamma) - r) ** 2) < 0.02
