#!/usr/bin/env python3
"""
tests/test_fuzzy_layer.py
=========================
The public fuzzy layer ``awesometmc.fuzzy``: a fuzzy switching model is
discretised into a ``PairGSSParams`` and filtered by ``GSSFilter``.

Oracles: the scalar reference implementation of the 2020 paper
(``awesometmc.experiments.fuzzy_cgofmsm``), and, in higher dimensions, the
enumeration of all regime paths of ``tests.test_pair_gss_params``.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest

from awesometmc.experiments import fuzzy_cgofmsm as ref
from awesometmc.experiments.fuzzy_cgofmsm_paper import PAPER_FIG5, mse_study
from awesometmc.fuzzy import (
    FMC1,
    FMC2,
    FuzzyCGOMSM,
    FuzzyDiscretisation,
    FuzzyGrid,
    mpm_jumps,
)
from awesometmc.utils.exceptions import CovarianceError, ParamError
from tests.test_pair_gss_params import _enumerate

LAWS = {
    "fmc1_fig1a": lambda: FMC1(0.10, 0.21, 0.076),
    "fmc1_fig5": lambda: FMC1(0.07, 0.16, 0.05),
    "fmc2_fig1b": lambda: FMC2(0.15, 0.60, 0.20),
}
# public interpolation -> (enforce_cgo of the reference, likelihood of the reference filter)
VARIANTS = {"cgo": (True, "paper"), "bilinear": (False, "moment")}


def _paper_model(law, interpolation="cgo") -> FuzzyCGOMSM:
    """The Gaussian laws of Section VI. Two of the printed corners violate the CGO
    identity, so interpolation="cgo" warns; the warning has its own test."""
    g = ref.paper_gaussian_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return FuzzyCGOMSM(law, means=(g.M0, g.M1), gammas=g.Gamma, interpolation=interpolation)


def _run(disc: FuzzyDiscretisation, ys, mode=None):
    filt = disc.filter(mode)
    out = [filt.step(y) for y in ys]
    return (
        np.array([o.E_x.ravel() for o in out]),
        np.array([o.Var_x for o in out]),
        np.array([o.pi for o in out]),
        sum(o.log_lik for o in out),
        filt.mode,
    )


def _random_model(seed, q, s, law=None, interpolation="cgo", cgo_corners=True, scale=0.15):
    """A stationary model in dimensions (q, s): Gamma[i, j] built from two hard
    covariances S_0, S_1 and four small transitions, with C = 0 at the corners unless
    ``cgo_corners`` is False."""
    rng = np.random.default_rng(seed)
    d = q + s

    def spd():
        a = rng.standard_normal((d, d))
        return a @ a.T / d + np.eye(d)

    S = [spd(), spd()]
    gammas = {}
    for i, j in itertools.product((0, 1), repeat=2):
        F = scale * rng.standard_normal((d, d))
        if cgo_corners:
            F[q:, :q] = 0.0
        gammas[(i, j)] = np.block([[S[i], S[i] @ F.T], [F @ S[i], S[j]]])
        assert np.linalg.eigvalsh(gammas[(i, j)]).min() > 0.0, "invalid corner: change the seed"
    return FuzzyCGOMSM(
        law or FMC1(0.07, 0.16, 0.05),
        means=(rng.standard_normal(d), rng.standard_normal(d)),
        gammas=gammas,
        q=q,
        s=s,
        interpolation=interpolation,
    )


# --------------------------------------------------------------------------
# Scalar case: the public layer equals the reference of the paper
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", LAWS)
@pytest.mark.parametrize("interpolation", VARIANTS)
def test_public_filter_equals_the_reference(name, interpolation):
    law = LAWS[name]()
    enforce, likelihood = VARIANTS[interpolation]
    model = _paper_model(law, interpolation)
    _, _, y = model.simulate(200, seed=1)
    for F in (0, 1, 4, 10):
        disc = model.discretise(F)
        e_x, var_x, pi, log_lik, mode = _run(disc, y)
        assert mode == ("cgo" if interpolation == "cgo" else "gpb2")
        r_model = ref.discretise(law, ref.paper_gaussian_model(enforce), F)
        r = ref.fuzzy_filter(r_model, y[:, 0], likelihood=likelihood)
        assert np.allclose(disc.nodes, r_model.nodes)
        assert np.allclose(e_x[:, 0], r.E_x, atol=1e-10)
        assert np.allclose(var_x[:, 0, 0], r.Var_x, atol=1e-10)
        assert np.allclose(pi, r.post, atol=1e-10)
        assert log_lik == pytest.approx(r.log_lik, abs=1e-8)


@pytest.mark.parametrize("interpolation", VARIANTS)
def test_gaussian_laws_equal_the_reference(interpolation):
    model = _paper_model(LAWS["fmc1_fig5"](), interpolation)
    g = ref.paper_gaussian_model(VARIANTS[interpolation][0])
    for r1, r2 in itertools.product((0.0, 0.3, 0.5, 1.0), repeat=2):
        assert np.allclose(model.gamma(r1, r2), g.gamma(r1, r2))
        F, Q = model.transition(r1, r2)
        A, Qr = g.transition(r1, r2)
        assert np.allclose(F, A, atol=1e-12) and np.allclose(Q, Qr, atol=1e-12)
        assert np.allclose(model.mean(r1), g.mean(r1)) and np.allclose(model.cov(r1), g.cov(r1))


def test_interpolation_cgo_is_exact_and_bilinear_is_not():
    cgo, bil = (_paper_model(LAWS["fmc1_fig5"](), i) for i in ("cgo", "bilinear"))
    rs = np.linspace(0.0, 1.0, 6)
    assert all(np.all(cgo.c_block(a, b) == 0.0) for a in rs for b in rs)
    assert abs(bil.c_block(0.5, 0.5)).max() > 1e-2  # inside the square
    assert abs(bil.c_block(0.0, 0.0)).max() > 1e-2  # a corner printed in the paper
    assert cgo.discretise(3).params.is_cgo() and not bil.discretise(3).params.is_cgo()
    assert bil.discretise(3).filter().mode == "gpb2"


@pytest.mark.parametrize("interpolation", VARIANTS)
def test_known_jumps_filter_equals_the_reference(interpolation):
    model = _paper_model(LAWS["fmc2_fig1b"](), interpolation)
    r, _, y = model.simulate(150, seed=2)
    e_x, var_x = model.known_jumps_filter(r, y)
    e_ref, v_ref = ref.known_jumps_filter(
        ref.paper_gaussian_model(VARIANTS[interpolation][0]), r, y
    )
    assert np.allclose(e_x[:, 0], e_ref, atol=1e-10) and np.allclose(
        var_x[:, 0, 0], v_ref, atol=1e-10
    )


def test_hardened_model_equals_the_reference():
    law = LAWS["fmc1_fig5"]()
    model = _paper_model(law)
    _, _, y = model.simulate(150, seed=3)
    hard = model.harden()
    assert hard.K == 2 and np.array_equal(hard.nodes, [0.0, 1.0])
    e_x, _, pi, log_lik, mode = _run(hard, y)
    r = ref.fuzzy_filter(ref.harden(law, ref.paper_gaussian_model(True)), y[:, 0])
    assert mode == "cgo"
    assert np.allclose(e_x[:, 0], r.E_x, atol=1e-10) and np.allclose(pi, r.post, atol=1e-10)


# --------------------------------------------------------------------------
# Any dimensions
# --------------------------------------------------------------------------
@pytest.mark.parametrize("q,s", [(2, 1), (1, 2), (2, 2)])
def test_filter_is_exact_in_higher_dimensions(q, s):
    """interpolation="cgo": the discretised model has C = 0 and the filter equals the
    enumeration of all regime paths."""
    model = _random_model(10 * q + s, q, s)
    _, _, y = model.simulate(6, seed=4)
    disc = model.discretise(1)  # K = 3, 729 paths
    assert disc.params.is_cgo() and (disc.params.q, disc.params.s) == (q, s)
    e_x, _, pi, log_lik, mode = _run(disc, y)
    ys = [yn.reshape(s, 1) for yn in y]
    e_ref, pi_ref, ll_ref = _enumerate(disc.params, ys)
    assert mode == "cgo"
    assert np.allclose(e_x[-1], e_ref, atol=1e-10) and np.allclose(pi[-1], pi_ref, atol=1e-10)
    assert log_lik == pytest.approx(ll_ref, abs=1e-9)


@pytest.mark.parametrize("q,s", [(2, 1), (1, 2)])
def test_discretised_model_is_stationary(q, s):
    """F S(r_n) F^T + Q = S(r_{n+1}) and b = M(r_{n+1}) - F M(r_n), for both interpolations."""
    for interpolation in VARIANTS:
        model = _random_model(20 + q, q, s, interpolation=interpolation)
        disc = model.discretise(2)
        p, nodes = disc.params, disc.nodes
        for j, k in itertools.product(range(disc.K), repeat=2):
            F = p.F(j, k)
            assert np.allclose(F @ model.cov(nodes[j]) @ F.T + p.Sigma_W(j, k), model.cov(nodes[k]))
            assert np.allclose(F @ model.mean(nodes[j]) + p.b(j, k).ravel(), model.mean(nodes[k]))
        assert np.allclose(p.pi0 @ p.P, p.pi0)


def test_known_jumps_filter_in_higher_dimensions():
    """On a constant hard path the known-jumps filter is the filter of a one-regime model."""
    model = _random_model(31, 2, 1)
    r = np.zeros(40)
    _, _, y = model.simulate(40, seed=5)
    e_x, var_x = model.known_jumps_filter(r, y)
    assert e_x.shape == (40, 2) and var_x.shape == (40, 2, 2)
    assert all(np.linalg.eigvalsh(v).min() > 0.0 for v in var_x)
    # steady state: the covariance converges
    assert np.allclose(var_x[-1], var_x[-2], atol=1e-8)


def test_corners_far_from_cgo_are_reported():
    """interpolation="cgo" on corners that violate the identity: a warning at
    construction, and a clear error if no valid Gaussian model is left."""
    g = ref.paper_gaussian_model()
    law = LAWS["fmc1_fig5"]()
    with pytest.warns(RuntimeWarning, match="hard corners do not satisfy"):
        paper = FuzzyCGOMSM(law, (g.M0, g.M1), g.Gamma)
    assert paper.corner_residual() == pytest.approx(0.1463, abs=1e-4)
    paper.discretise(4)  # a small violation is corrected without harm
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning for bilinear, nor for CGO corners
        FuzzyCGOMSM(law, (g.M0, g.M1), g.Gamma, interpolation="bilinear")
        assert _random_model(1, 2, 1).corner_residual() < 1e-12
    with pytest.warns(RuntimeWarning):
        far = _random_model(7, 2, 2, cgo_corners=False, scale=0.25)  # invalid once corrected
    with pytest.raises(CovarianceError, match="far from the structure"):
        far.discretise(5)
    with pytest.raises(CovarianceError, match="far from the structure"):
        far.simulate(4000, seed=0)
    # the same covariances are a valid model for the bilinear interpolation
    _random_model(7, 2, 2, cgo_corners=False, scale=0.25, interpolation="bilinear").discretise(5)


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def test_simulation_is_reproducible_and_follows_the_model():
    model = _random_model(40, 2, 1)
    r, x, y = model.simulate(30_000, seed=6)
    r2, x2, y2 = model.simulate(30_000, seed=6)
    assert np.array_equal(r, r2) and np.array_equal(x, x2) and np.array_equal(y, y2)
    assert x.shape == (30_000, 2) and y.shape == (30_000, 1)
    hard = (r == 0.0) | (r == 1.0)
    assert hard.mean() == pytest.approx(model.law.p_hard, abs=0.015)
    for value in (0.0, 1.0):
        z = np.hstack([x, y])[r == value]
        assert np.allclose(z.mean(axis=0), model.mean(value), atol=0.06)
        assert np.allclose(np.cov(z.T), model.cov(value), atol=0.12)
    assert isinstance(model.simulate(5, seed=np.random.default_rng(0))[0], np.ndarray)


# --------------------------------------------------------------------------
# Grid, nodes, MPM
# --------------------------------------------------------------------------
def test_zero_probability_nodes_are_dropped():
    model = _paper_model(FMC1(0.4, 0.0, 0.0))  # purely hard law
    _, _, y = model.simulate(100, seed=7)
    disc = model.discretise(3)
    assert disc.K == 2 and np.array_equal(disc.nodes, [0.0, 1.0])
    e_x, _, pi, log_lik, _ = _run(disc, y)
    e_0, _, pi_0, ll_0, _ = _run(model.discretise(0), y)
    e_h, _, pi_h, ll_h, _ = _run(model.harden(), y)
    assert np.allclose(e_x, e_0) and np.allclose(pi, pi_0) and log_lik == pytest.approx(ll_0)
    assert np.allclose(e_x, e_h, atol=1e-10) and log_lik == pytest.approx(ll_h, abs=1e-8)


def test_custom_grid():
    model = _paper_model(LAWS["fmc1_fig5"]())
    x, w = np.polynomial.legendre.leggauss(4)
    grid = FuzzyGrid.custom(0.5 * (x + 1.0), 0.5 * w)  # Gauss-Legendre on (0, 1)
    disc = model.discretise(grid=grid)
    assert disc.K == 6 and np.allclose(disc.weights[1:-1].sum(), 1.0)
    _, _, y = model.simulate(60, seed=8)
    e_x, _, pi, _, mode = _run(disc, y)
    assert mode == "cgo" and np.all(np.isfinite(e_x)) and np.allclose(pi.sum(axis=1), 1.0)
    for bad in (
        ([0.5, 0.2], [0.5, 0.5]),
        ([0.0, 0.5], [0.5, 0.5]),
        ([0.5], [0.5, 0.5]),
        ([0.5], [-1.0]),
    ):
        with pytest.raises(ValueError):
            FuzzyGrid.custom(*bad)
    with pytest.raises(ValueError):
        FuzzyGrid.midpoint(-1)


def test_mpm_and_mean_jump():
    model = _paper_model(LAWS["fmc1_fig5"]())
    disc = model.discretise(3)
    post = np.array(
        [
            [0.40, 0.10, 0.20, 0.10, 0.20],  # hard mass 0.6 -> hard, 0 wins
            [0.10, 0.05, 0.05, 0.10, 0.70],  # hard, 1 wins
            [0.20, 0.10, 0.45, 0.05, 0.20],  # hard mass 0.4 -> fuzzy, middle node
            [0.30, 0.35, 0.10, 0.05, 0.20],  # hard mass 0.5, not > 0.5 -> fuzzy, first node
        ]
    )
    assert np.allclose(disc.mpm(post), [0.0, 1.0, disc.nodes[2], disc.nodes[1]])
    assert disc.mpm(post[0]).shape == (1,)
    assert np.allclose(disc.mean_jump(post), post @ disc.nodes)
    assert np.allclose(
        mpm_jumps(np.array([[0.3, 0.7], [0.6, 0.4]]), np.array([0.0, 1.0])), [1.0, 0.0]
    )
    # a grid without the atom 1 (dropped node): the rule still applies
    assert np.allclose(
        mpm_jumps(np.array([[0.2, 0.8], [0.7, 0.3]]), np.array([0.0, 0.5])), [0.5, 0.0]
    )


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def test_invalid_models_are_rejected():
    g = ref.paper_gaussian_model()
    law = LAWS["fmc1_fig5"]()
    with pytest.raises(ParamError):
        FuzzyCGOMSM(law, (g.M0, g.M1), g.Gamma, interpolation="linear")
    with pytest.raises(ParamError):
        FuzzyCGOMSM(law, (g.M0, g.M1[:1]), g.Gamma)
    with pytest.raises(ParamError):
        FuzzyCGOMSM(law, (g.M0, g.M1), {k: v for k, v in g.Gamma.items() if k != (1, 1)})
    with pytest.raises(ParamError):
        FuzzyCGOMSM(law, (g.M0, g.M1), g.Gamma, q=2, s=1)  # 4x4 matrices cannot be q + s = 3
    bad = {k: v.copy() for k, v in g.Gamma.items()}
    bad[(0, 1)][0, 0] = 0.9
    with pytest.raises(ParamError):
        FuzzyCGOMSM(law, (g.M0, g.M1), bad)
    asym = {k: v.copy() for k, v in g.Gamma.items()}
    asym[(0, 0)][0, 3] += 0.1
    with pytest.raises(ParamError):
        FuzzyCGOMSM(law, (g.M0, g.M1), asym)


# --------------------------------------------------------------------------
# Figure 5 of the paper through the public interface
# --------------------------------------------------------------------------
def test_figure_5_through_the_public_interface():
    """Same errors as the reference implementation, hence the same reproduction of Fig. 5."""
    law = FMC1(0.07, 0.16, 0.05)
    gauss = ref.paper_gaussian_model(enforce_cgo=True)
    kw = dict(F_list=PAPER_FIG5["F"], N=300, n_runs=3, seed=0)
    reference = mse_study(law, gauss, api="reference", **kw)
    public = mse_study(law, gauss, api="public", **kw)
    for a, b in zip(reference, public):
        assert a["F"] == b["F"]
        for key in ("mse_x_fuzzy", "mse_r_fuzzy", "mse_x_hard", "mse_r_hard", "mse_x_kj"):
            assert a[key] == pytest.approx(b[key], abs=1e-9), key
    # the shape of Fig. 5: the error drops from F = 0 and has levelled off by F = 3
    x = [row["mse_x_fuzzy"] for row in public]
    assert x[0] > x[1] > x[2] and abs(x[3] - x[-1]) < 0.01
