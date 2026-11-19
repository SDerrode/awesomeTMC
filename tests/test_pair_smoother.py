#!/usr/bin/env python3
"""
tests/test_pair_smoother.py
===========================
Fixed-interval smoothing of models whose parameters depend on the regime pair
(``PairGSSParams``) when the observation does not read the state (``C = 0``), and
its use by the fuzzy layer (``FuzzyDiscretisation.smoother``).

Oracles: the existing smoother of the arrival-indexed family ``C = 0`` applied to
the augmented regime ``S_n = (r_{n-1}, r_n)``, which is exact but costs ``K^4``; the
scalar reference of the 2020 paper (``fuzzy_cgofmsm.fuzzy_smoother``); the filter.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from awesometmc.classes import PairGSSParams
from awesometmc.experiments import fuzzy_cgofmsm as ref
from awesometmc.filter import (
    GSSFilter,
    chain_log_likelihood,
    regime_smoother,
    reweight_smoother,
)
from awesometmc.fuzzy import FMC1
from tests.test_fuzzy_layer import LAWS, _paper_model, _random_model
from tests.test_pair_gss_params import _random_pair_params, _simulate

TOL = 1e-10


def _observations(params, N, seed):
    return _simulate(params, N, seed)[-1]


def _filter(params, ys):
    filt = GSSFilter(params, mode="cgo")
    out = [filt.step(y) for y in ys]
    return (
        np.array([o.E_x.ravel() for o in out]),
        np.array([o.Var_x for o in out]),
        np.array([o.pi for o in out]),
        sum(o.log_lik for o in out),
    )


# --------------------------------------------------------------------------
# Core: reweight_smoother / regime_smoother on PairGSSParams
# --------------------------------------------------------------------------
@pytest.mark.parametrize("K, q, s", [(2, 1, 1), (3, 1, 1), (2, 2, 1), (2, 1, 2), (3, 2, 2)])
def test_pair_smoother_equals_the_augmented_oracle(K, q, s):
    params = _random_pair_params(seed=10 * K + q + s, K=K, q=q, s=s)
    ys = _observations(params, 40, seed=1)
    E, V, gamma, ll = reweight_smoother(params, ys)
    Ea, Va, ga, lla = reweight_smoother(params.augmented(), ys)
    assert np.allclose(E, Ea, atol=TOL)
    assert np.allclose(V, Va, atol=TOL)
    assert ll == pytest.approx(lla, abs=1e-9)
    # the augmented regime (r_{n-1}, r_n) marginalised on its second component
    assert np.allclose(gamma[1:], ga.reshape(-1, K, K).sum(axis=1)[1:], atol=TOL)


def test_pair_smoother_equals_the_arrival_smoother_on_a_broadcast_model():
    arrival = _random_pair_params(seed=3, K=2, q=2, s=1).augmented()  # a GSSParams, C = 0
    pair = PairGSSParams.from_arrival(arrival)
    ys = _observations(pair, 30, seed=2)
    for a, b in zip(reweight_smoother(pair, ys), reweight_smoother(arrival, ys), strict=True):
        assert np.allclose(a, b, atol=TOL)
    for a, b in zip(
        regime_smoother(pair, ys, kernel="c0"),
        regime_smoother(arrival, ys, kernel="c0"),
        strict=True,
    ):
        assert np.allclose(a, b, atol=TOL)


def test_likelihood_of_the_chain_equals_the_likelihood_of_the_filter():
    params = _random_pair_params(seed=5, K=3, q=1, s=2)
    ys = _observations(params, 60, seed=4)
    ll_filter = _filter(params, ys)[3]
    assert chain_log_likelihood(params, ys, kernel="c0") == pytest.approx(ll_filter, abs=1e-9)
    assert reweight_smoother(params, ys)[3] == pytest.approx(ll_filter, abs=1e-9)


def test_smoothed_pairs_are_consistent_with_the_smoothed_regimes():
    params = _random_pair_params(seed=6, K=3)
    ys = _observations(params, 25, seed=5)
    gamma, pairs, _ = regime_smoother(params, ys, kernel="c0")
    assert pairs.shape == (24, 3, 3)
    assert np.allclose(gamma.sum(axis=1), 1.0)
    assert np.allclose(pairs.sum(axis=2), gamma[:-1], atol=TOL)
    assert np.allclose(pairs.sum(axis=1), gamma[1:], atol=TOL)


def test_last_smoothed_step_is_the_filtered_one_and_one_step_is_the_filter():
    params = _random_pair_params(seed=7, K=2, q=2, s=2)
    ys = _observations(params, 20, seed=6)
    Ef, Vf, pif, _ = _filter(params, ys)
    E, V, gamma, _ = reweight_smoother(params, ys)
    assert np.allclose(E[-1], Ef[-1], atol=TOL)
    assert np.allclose(V[-1], Vf[-1], atol=TOL)
    assert np.allclose(gamma[-1], pif[-1], atol=TOL)
    E1, V1, g1, ll1 = reweight_smoother(params, ys[:1])
    Ef1, Vf1, pif1, llf1 = _filter(params, ys[:1])
    assert np.allclose(E1, Ef1, atol=TOL) and np.allclose(V1, Vf1, atol=TOL)
    assert np.allclose(g1, pif1, atol=TOL) and ll1 == pytest.approx(llf1, abs=1e-10)


def test_ab_kernels_are_refused_for_pair_models():
    params = _random_pair_params(seed=8)
    ys = _observations(params, 5, seed=7)
    for kernel in ("ab", "lag1"):
        with pytest.raises(ValueError, match="arrival-indexed"):
            regime_smoother(params, ys, kernel=kernel)
    with pytest.raises(ValueError, match="arrival-indexed"):
        regime_smoother(params, ys)  # the default kernel is "ab"


def test_regime_summaries_of_the_filter():
    params = _random_pair_params(seed=9, K=3, q=2, s=1)
    ys = _observations(params, 6, seed=8)
    filt = GSSFilter(params, mode="cgo")
    with pytest.raises(RuntimeError, match="at least one call"):
        filt.regime_summaries()
    for y in ys:
        res = filt.step(y)
        pi, m, V = filt.regime_summaries()
        assert pi.shape == (3,) and m.shape == (3, 2) and V.shape == (3, 2, 2)
        assert np.allclose(pi, res.pi)
        assert np.allclose(pi @ m, res.E_x.ravel(), atol=TOL)
    pi[:] = 0.0  # a copy: the filter is not disturbed
    assert filt.regime_summaries()[0].sum() == pytest.approx(1.0)
    with pytest.raises(RuntimeError, match='mode "cgo" only'):
        GSSFilter(params, mode="gpb2").regime_summaries()


# --------------------------------------------------------------------------
# Fuzzy layer
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", LAWS)
@pytest.mark.parametrize("F", [0, 1, 4])
def test_fuzzy_smoother_equals_the_reference(name, F):
    law = LAWS[name]()
    model = _paper_model(law)
    _, _, y = model.simulate(N=120, seed=11)
    disc = model.discretise(F=F)
    E, V, gamma, ll = disc.smoother(y)
    assert E.shape == (120, 1) and V.shape == (120, 1, 1) and gamma.shape == (120, disc.K)
    rm = ref.discretise(law, ref.paper_gaussian_model(enforce_cgo=True), F)
    E_ref, post_ref = ref.fuzzy_smoother(rm, np.asarray(y).ravel())
    assert np.allclose(E.ravel(), np.asarray(E_ref).ravel(), atol=TOL)
    assert np.allclose(gamma, post_ref, atol=TOL)
    assert np.all(V >= -1e-12)
    assert ll == pytest.approx(sum(o.log_lik for o in map(disc.filter().step, y)), abs=1e-8)


@pytest.mark.parametrize("q, s", [(2, 1), (1, 2), (2, 2)])
def test_fuzzy_smoother_in_higher_dimensions(q, s):
    model = _random_model(seed=20 + q + 3 * s, q=q, s=s)
    _, _, y = model.simulate(N=30, seed=12)
    disc = model.discretise(F=1)
    E, V, gamma, ll = disc.smoother(y)
    Ea, Va, _, lla = reweight_smoother(disc.params.augmented(), y)
    assert np.allclose(E, Ea, atol=TOL) and np.allclose(V, Va, atol=TOL)
    assert ll == pytest.approx(lla, abs=1e-9)


def test_smoothing_beats_filtering_on_states_and_jumps():
    model = _paper_model(FMC1(0.07, 0.16, 0.05))
    disc = model.discretise(F=5)
    filt_x = filt_r = smooth_x = smooth_r = 0.0
    for seed in range(8):
        r, x, y = model.simulate(N=400, seed=100 + seed)
        filt = disc.filter()
        out = [filt.step(v) for v in y]
        E_f = np.array([o.E_x.ravel() for o in out]).ravel()
        r_f = disc.mpm(np.array([o.pi for o in out]))
        E_s, _, gamma, _ = disc.smoother(y)
        filt_x += np.mean((E_f - np.ravel(x)) ** 2)
        smooth_x += np.mean((E_s.ravel() - np.ravel(x)) ** 2)
        filt_r += np.mean((r_f - r) ** 2)
        smooth_r += np.mean((disc.mpm(gamma) - r) ** 2)
    assert smooth_x < filt_x
    assert smooth_r < filt_r


def test_bilinear_interpolation_is_smoothed_with_a_warning():
    model = _paper_model(FMC1(0.07, 0.16, 0.05), interpolation="bilinear")
    _, _, y = model.simulate(N=40, seed=13)
    disc = model.discretise(F=2)
    with pytest.warns(RuntimeWarning):
        E, V, gamma, ll = disc.smoother(y)
    assert np.all(np.isfinite(E)) and np.isfinite(ll)
    assert np.allclose(gamma.sum(axis=1), 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _paper_model(FMC1(0.07, 0.16, 0.05)).discretise(F=2).smoother(y)  # exact: silent
