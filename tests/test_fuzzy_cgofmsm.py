#!/usr/bin/env python3
"""
tests/test_fuzzy_cgofmsm.py
===========================
Guards the reference implementation of the fuzzy switching filter of Bouyahia,
Derrode and Pieczynski (IEEE TFS 2020), ``awesometmc/experiments/fuzzy_cgofmsm.py``.

The filter is checked against three independent references: the enumeration of
all regime paths, a joint-Gaussian conditioning written here, and the existing
pair-dependent CGO filter of ``imm_cgo_projection``.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import integrate

from awesometmc.experiments.fuzzy_cgofmsm import (
    FMC1,
    FMC2,
    FuzzyGaussianModel,
    FuzzyGrid,
    discretise,
    enumerate_paths,
    exact_enumeration_filter,
    fuzzy_filter,
    fuzzy_smoother,
    harden,
    known_jumps_filter,
    mpm_jumps,
    paper_gaussian_model,
    simulate,
)
from awesometmc.experiments.imm_cgo_projection import cgo_filter

LAWS = {
    "fmc1_fig1a": lambda: FMC1(0.10, 0.21, 0.076),
    "fmc1_fig5": lambda: FMC1(0.07, 0.16, 0.05),
    "fmc2_fig1b": lambda: FMC2(0.15, 0.60, 0.20),
}


def _nu_integral(f, breaks=()):
    """Integral of ``f`` w.r.t. nu = delta_0 + delta_1 + Lebesgue(0, 1)."""
    pts = [b for b in breaks if 0.0 < b < 1.0] or None
    inner = integrate.quad(lambda r: float(f(r)), 0.0, 1.0, points=pts, limit=200)[0]
    return float(f(0.0)) + float(f(1.0)) + inner


def _breaks(law, r1=None):
    d = getattr(law, "delta", 0.0)
    out = [d, 1.0 - d]
    if r1 is not None:
        out += [r1, r1 - d, r1 + d]
    return out


# --------------------------------------------------------------------------
# Section V-A: the fuzzy Markov chain laws
# --------------------------------------------------------------------------
def test_paper_values_of_beta_and_p_hard():
    """beta and p_H quoted in the captions of Fig. 1 of the paper."""
    a, b = LAWS["fmc1_fig1a"](), LAWS["fmc2_fig1b"]()
    assert a.beta == pytest.approx(0.03, abs=5e-3) and a.p_hard == pytest.approx(0.55, abs=5e-3)
    assert b.beta == pytest.approx(0.0, abs=5e-3) and b.p_hard == pytest.approx(0.54, abs=5e-3)
    # Fig. 5 quotes beta = 0.158 and p_H = 0.69; the formula gives p_H = 2/3
    c = LAWS["fmc1_fig5"]()
    assert c.beta == pytest.approx(0.158, abs=1e-3) and c.p_hard == pytest.approx(2 / 3, abs=1e-9)


@pytest.mark.parametrize("name", LAWS)
def test_marginal_is_normalised_and_matches_the_joint(name):
    law = LAWS[name]()
    assert _nu_integral(law.marginal, _breaks(law)) == pytest.approx(1.0, abs=1e-9)
    for r1 in (0.0, 0.07, 0.2, 0.5, 0.93, 1.0):
        got = _nu_integral(lambda r2, r1=r1: law.joint(r1, r2), _breaks(law, r1))
        assert got == pytest.approx(float(law.marginal(r1)), abs=1e-9)
        cond = _nu_integral(lambda r2, r1=r1: law.conditional(r2, r1), _breaks(law, r1))
        assert cond == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("name", LAWS)
def test_sampler_follows_the_law(name):
    law = LAWS[name]()
    r = law.sample(60_000, np.random.default_rng(3))
    hard = (r == 0.0) | (r == 1.0)
    assert hard.mean() == pytest.approx(law.p_hard, abs=0.012)
    assert (r == 0.0).mean() == pytest.approx(law.hard_mass, abs=0.012)
    assert r[~hard].mean() == pytest.approx(0.5, abs=0.02)
    # transition 0 -> 1 has probability beta / p(0)
    from0 = r[1:][r[:-1] == 0.0]
    assert (from0 == 1.0).mean() == pytest.approx(law.beta / law.hard_mass, abs=0.01)
    if isinstance(law, FMC2):
        both = ~hard[1:] & ~hard[:-1]
        assert np.all(np.abs(np.diff(r))[both] <= law.delta + 1e-12)


@pytest.mark.parametrize("name", LAWS)
def test_discretised_chain_is_stationary_and_converges(name):
    law = LAWS[name]()
    for F in (0, 1, 4, 10):
        pi, P = law.discretise(FuzzyGrid.midpoint(F))
        assert pi.sum() == pytest.approx(1.0)
        assert np.allclose(P[pi > 0].sum(axis=1), 1.0)
        assert np.allclose(pi @ P, pi)
    pi, _ = law.discretise(FuzzyGrid.midpoint(400))
    assert pi[0] == pytest.approx(law.hard_mass, abs=2e-3)


def test_hard_law_has_no_fuzzy_mass():
    law = FMC1(0.4, 0.0, 0.0)  # beta = 0.1: a plain 2-state chain
    assert law.beta == pytest.approx(0.1) and law.p_fuzzy == pytest.approx(0.0)
    expected = np.array([[0.8, 0.2], [0.2, 0.8]])
    assert np.allclose(law.harden()[1], expected)
    pi, P = law.discretise(FuzzyGrid.midpoint(3))
    assert np.allclose(pi[1:-1], 0.0) and np.allclose(P[[0, -1]][:, [0, -1]], expected)
    assert set(np.unique(law.sample(500, np.random.default_rng(0)))) <= {0.0, 1.0}


# --------------------------------------------------------------------------
# Section V-B: the Gaussian laws
# --------------------------------------------------------------------------
def test_transition_reproduces_the_stationary_covariance():
    for enforce in (False, True):
        g = paper_gaussian_model(enforce)
        for r1, r2 in itertools.product((0.0, 0.2, 0.5, 0.9, 1.0), repeat=2):
            A, Q = g.transition(r1, r2)
            assert np.linalg.eigvalsh(Q).min() > 0.0
            assert np.allclose(A @ g.cov(r1) @ A.T + Q, g.cov(r2))


def test_printed_matrices_are_not_a_cgomsm():
    """Finding on [BDP20] Section VI: a3 = 0 holds at two corners out of four only,
    and bilinear interpolation does not preserve it."""
    g = paper_gaussian_model()
    assert g.a3(1, 0) == pytest.approx(0.0, abs=1e-12)
    assert g.a3(1, 1) == pytest.approx(0.0, abs=1e-12)
    assert g.a3(0, 0) == pytest.approx(-0.1463, abs=1e-4)
    assert g.a3(0, 1) == pytest.approx(-0.1098, abs=1e-4)
    assert abs(g.a3(0.5, 0.5)) > 1e-2


def test_enforce_cgo_gives_a3_zero_everywhere():
    g = paper_gaussian_model(enforce_cgo=True)
    rs = np.linspace(0.0, 1.0, 11)
    assert max(abs(g.a3(a, b)) for a in rs for b in rs) < 1e-12


def test_inconsistent_gamma_is_rejected():
    g = paper_gaussian_model()
    bad = {k: v.copy() for k, v in g.Gamma.items()}
    bad[(0, 1)][0, 0] = 0.9  # Var(x_1 | r_1 = 0) would depend on r_2
    with pytest.raises(ValueError):
        FuzzyGaussianModel(g.M0, g.M1, bad)


# --------------------------------------------------------------------------
# Section IV: the filter
# --------------------------------------------------------------------------
def _sample(law, gauss, N, seed):
    return simulate(law, gauss, N, np.random.default_rng(seed))


@pytest.mark.parametrize("name", LAWS)
@pytest.mark.parametrize("likelihood", ["paper", "moment"])
def test_filter_is_exact_for_a_cgomsm(name, likelihood):
    """With a3 = 0 the recursion equals the enumeration of all K^N regime paths."""
    law, gauss = LAWS[name](), paper_gaussian_model(enforce_cgo=True)
    for F, N, seed in ((1, 7, 0), (2, 5, 1), (0, 9, 2)):
        _, _, ys = _sample(law, gauss, N, seed)
        model = discretise(law, gauss, F)
        res = fuzzy_filter(model, ys, likelihood=likelihood)
        e_x, post, log_lik = exact_enumeration_filter(model, ys)
        assert res.E_x[-1] == pytest.approx(e_x, abs=1e-10)
        assert np.allclose(res.post[-1], post, atol=1e-10)
        assert res.log_lik == pytest.approx(log_lik, abs=1e-9)


def test_filter_is_not_exact_with_the_printed_matrices():
    """With the matrices as printed (a3 != 0) the recursion is only an approximation."""
    law, gauss = LAWS["fmc1_fig5"](), paper_gaussian_model()
    _, _, ys = _sample(law, gauss, 7, 4)
    model = discretise(law, gauss, 1)
    _, _, log_lik = exact_enumeration_filter(model, ys)
    assert abs(fuzzy_filter(model, ys, likelihood="paper").log_lik - log_lik) > 1e-3


def test_filter_variance_matches_enumeration():
    law, gauss = LAWS["fmc1_fig5"](), paper_gaussian_model(enforce_cgo=True)
    _, _, ys = _sample(law, gauss, 6, 7)
    model = discretise(law, gauss, 1)
    res = fuzzy_filter(model, ys)
    assert np.all(res.Var_x > 0.0)
    # second moment by enumeration: E[x^2] = sum_paths w (m^2 + v); only the mean is
    # exposed, so check the law of total variance lower bound instead
    paths, log_w, means = enumerate_paths(model, ys)
    w = np.exp(log_w - np.logaddexp.reduce(log_w))
    between = float(w @ means[:, -1] ** 2) - float(w @ means[:, -1]) ** 2
    assert res.Var_x[-1] >= between - 1e-12


def test_fuzzy_nodes_are_inert_without_fuzzy_mass():
    """Hard limit: with a purely hard law, F has no effect."""
    law, gauss = FMC1(0.4, 0.0, 0.0), paper_gaussian_model(enforce_cgo=True)
    _, _, ys = _sample(law, gauss, 200, 5)
    ref = fuzzy_filter(discretise(law, gauss, 0), ys)
    for F in (1, 4):
        res = fuzzy_filter(discretise(law, gauss, F), ys)
        assert np.allclose(res.E_x, ref.E_x, atol=1e-10)
        assert res.log_lik == pytest.approx(ref.log_lik, abs=1e-9)
        assert np.allclose(res.post[:, 1:-1], 0.0)


@pytest.mark.parametrize("name", LAWS)
def test_agrees_with_the_library_pair_dependent_cgo_filter(name):
    """Cross-check against ``imm_cgo_projection.cgo_filter`` (zero-mean models)."""
    law = LAWS[name]()
    base = paper_gaussian_model(enforce_cgo=True)
    gauss = FuzzyGaussianModel(np.zeros(2), np.zeros(2), base.Gamma, enforce_cgo=True)
    _, _, ys = _sample(law, gauss, 120, 6)
    model = discretise(law, gauss, 3)
    K = model.K
    blocks = {
        (g, h): dict(
            A=model.A[g, h, :1, :1],
            B=model.A[g, h, :1, 1:],
            D=model.A[g, h, 1:, 1:],
            SU=model.Q[g, h, :1, :1],
            Dl=model.Q[g, h, :1, 1:],
            SV=model.Q[g, h, 1:, 1:],
        )
        for g, h in itertools.product(range(K), repeat=2)
    }
    p_true = SimpleNamespace(
        K=K,
        q=1,
        s=1,
        P=model.P,
        pi0=model.pi,
        mu_z0=lambda r: np.zeros(2),
        Sigma_z0=lambda r: model.S[r],
    )
    e_x, pi = cgo_filter(blocks, p_true, ys)
    res = fuzzy_filter(model, ys)
    assert np.allclose(res.E_x, e_x.ravel(), atol=1e-8)
    assert np.allclose(res.post, pi, atol=1e-8)


def test_estimate_converges_with_F():
    law, gauss = LAWS["fmc1_fig5"](), paper_gaussian_model(enforce_cgo=True)
    _, _, ys = _sample(law, gauss, 300, 8)
    ref = fuzzy_filter(discretise(law, gauss, 60), ys).E_x
    err = [np.abs(fuzzy_filter(discretise(law, gauss, F), ys).E_x - ref).max() for F in (2, 8, 30)]
    assert err[0] > err[1] > err[2]


# --------------------------------------------------------------------------
# Known jumps, smoother, MPM rule
# --------------------------------------------------------------------------
def _joint_gaussian(gauss, r):
    """Mean and covariance of (z_1, ..., z_N) given the regime path ``r``."""
    N = len(r)
    mean = np.concatenate([gauss.mean(x) for x in r])
    C = np.zeros((2 * N, 2 * N))
    for n in range(N):
        C[2 * n : 2 * n + 2, 2 * n : 2 * n + 2] = block = gauss.cov(r[n])
        for k in range(n + 1, N):
            block = gauss.transition(r[k - 1], r[k])[0] @ block
            C[2 * k : 2 * k + 2, 2 * n : 2 * n + 2] = block
            C[2 * n : 2 * n + 2, 2 * k : 2 * k + 2] = block.T
    return mean, C


def _condition_x_on_y(mean, C, n, ys, upto):
    """E[x_n | y_1, ..., y_upto] by Gaussian conditioning (indices are 0-based, upto exclusive)."""
    iy = [2 * k + 1 for k in range(upto)]
    K = C[2 * n, iy] @ np.linalg.inv(C[np.ix_(iy, iy)])
    return float(mean[2 * n] + K @ (np.asarray(ys)[:upto] - mean[iy]))


@pytest.mark.parametrize("enforce", [False, True])
def test_known_jumps_filter_is_the_gaussian_conditional(enforce):
    gauss = paper_gaussian_model(enforce)
    r, _, ys = _sample(LAWS["fmc1_fig5"](), gauss, 8, 9)
    mean, C = _joint_gaussian(gauss, r)
    e_x, var_x = known_jumps_filter(gauss, r, ys)
    for n in range(len(r)):
        assert e_x[n] == pytest.approx(_condition_x_on_y(mean, C, n, ys, n + 1), abs=1e-10)
    assert np.all(var_x > 0.0)


def test_appendix_b_property_holds_iff_cgomsm():
    """E[x_n | path, y_1^N] = E[x_n | path, y_1^n] in a CGOMSM: the future brings nothing."""
    law, N, n = LAWS["fmc1_fig5"](), 8, 3
    for enforce, same in ((True, True), (False, False)):
        gauss = paper_gaussian_model(enforce)
        r, _, ys = _sample(law, gauss, N, 10)
        mean, C = _joint_gaussian(gauss, r)
        filt, smooth = (
            _condition_x_on_y(mean, C, n, ys, n + 1),
            _condition_x_on_y(mean, C, n, ys, N),
        )
        assert (abs(filt - smooth) < 1e-10) is same


@pytest.mark.parametrize("name", LAWS)
def test_smoother_equals_enumeration(name):
    law, gauss = LAWS[name](), paper_gaussian_model(enforce_cgo=True)
    _, _, ys = _sample(law, gauss, 6, 11)
    model = discretise(law, gauss, 1)
    e_s, post_s = fuzzy_smoother(model, ys)
    paths, log_w, means = enumerate_paths(model, ys)
    w = np.exp(log_w - np.logaddexp.reduce(log_w))
    for n in range(len(ys)):
        assert np.allclose(
            post_s[n], np.bincount(paths[:, n], weights=w, minlength=model.K), atol=1e-10
        )
        assert e_s[n] == pytest.approx(float(w @ means[:, n]), abs=1e-10)
    res = fuzzy_filter(model, ys)
    assert e_s[-1] == pytest.approx(res.E_x[-1]) and np.allclose(post_s[-1], res.post[-1])


def test_mpm_rule_of_section_vi():
    nodes = FuzzyGrid.midpoint(3).nodes
    post = np.array(
        [
            [0.40, 0.10, 0.20, 0.10, 0.20],  # hard mass 0.6 -> hard, 0 wins
            [0.10, 0.05, 0.05, 0.10, 0.70],  # hard, 1 wins
            [0.20, 0.10, 0.45, 0.05, 0.20],  # hard mass 0.4 -> fuzzy, middle node
            [0.30, 0.35, 0.10, 0.05, 0.20],  # hard mass 0.5, not > 0.5 -> fuzzy, first node
        ]
    )
    assert np.allclose(mpm_jumps(post, nodes), [0.0, 1.0, nodes[2], nodes[1]])
    assert np.allclose(
        mpm_jumps(np.array([[0.3, 0.7], [0.6, 0.4]]), np.array([0.0, 1.0])), [1.0, 0.0]
    )


def test_hardened_model_is_a_two_regime_chain():
    law, gauss = LAWS["fmc2_fig1b"](), paper_gaussian_model()
    model = harden(law, gauss)
    assert model.K == 2 and np.allclose(model.P.sum(axis=1), 1.0)
    assert np.allclose(model.pi, 0.5) and np.allclose(model.pi @ model.P, model.pi)
