#!/usr/bin/env python3
"""
tests/test_cgo_filter_mode.py
=============================
``GSSFilter(mode="cgo")``: the exact filter of the models whose observation does
not read the state (C = 0), for arrival-indexed and pair-indexed parameters.

On these models the mode must return what ``mode="gpb2"`` returns (same means,
second moments, regime posteriors, innovations and likelihoods) and what the
enumeration of all regime paths returns, without running a Kalman update.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from awesometmc.classes import PairGSSParams
from awesometmc.experiments.models_paper import get_params
from awesometmc.experiments.run_simulations import _params_from_dict
from awesometmc.filter import GSSFilter
from tests.test_pair_gss_params import _enumerate, _filter, _random_pair_params, _simulate

CASES = [(2, 1, 1), (3, 1, 1), (2, 2, 1), (2, 1, 2), (3, 2, 2), (5, 1, 1)]


def _assert_same(got, ref, atol=1e-10):
    for a, b in zip(got, ref):
        assert np.allclose(a.E_x, b.E_x, atol=atol) and np.allclose(a.E_xx, b.E_xx, atol=atol)
        assert np.allclose(a.pi, b.pi, atol=atol)
        assert np.allclose(a.innovation, b.innovation, atol=atol)
        assert a.log_lik == pytest.approx(b.log_lik, abs=atol)
        assert a.n == b.n and a.E_x.shape == b.E_x.shape and a.E_xx.shape == b.E_xx.shape


@pytest.mark.parametrize("K,q,s", CASES)
def test_cgo_equals_gpb2_on_pair_dependent_models(K, q, s):
    pair = _random_pair_params(100 + 7 * K + 3 * q + s, K=K, q=q, s=s)
    _, _, ys = _simulate(pair, 120, seed=1)
    _assert_same(_filter(pair, ys, mode="cgo"), _filter(pair, ys, mode="gpb2"))


@pytest.mark.parametrize("K,q,s,N", [(2, 1, 1, 8), (3, 1, 1, 5), (2, 2, 1, 7), (2, 1, 2, 7)])
def test_cgo_equals_the_path_enumeration(K, q, s, N):
    pair = _random_pair_params(200 + K + q + s, K=K, q=q, s=s)
    _, _, ys = _simulate(pair, N, seed=2)
    out = _filter(pair, ys, mode="cgo")
    e_x, pi, log_lik = _enumerate(pair, ys)
    assert np.allclose(out[-1].E_x.ravel(), e_x, atol=1e-10)
    assert np.allclose(out[-1].pi, pi, atol=1e-10)
    assert sum(r.log_lik for r in out) == pytest.approx(log_lik, abs=1e-9)


@pytest.mark.parametrize("p", [1, 2])
def test_cgo_equals_gpb2_with_an_exogenous_input(p):
    pair = _random_pair_params(300 + p, K=2, q=2, s=1, p=p)
    N = 80
    u = np.random.default_rng(3).standard_normal((N, p))
    _, _, ys = _simulate(pair, N, seed=4, u=u)
    _assert_same(_filter(pair, ys, us=u, mode="cgo"), _filter(pair, ys, us=u, mode="gpb2"))


def test_cgo_on_arrival_indexed_params():
    """A GSSParams with C = 0 is accepted as is."""
    pair = _random_pair_params(400, K=3, q=2, s=1)
    arrival = PairGSSParams(
        K=pair.K,
        q=pair.q,
        s=pair.s,
        P=pair.P,
        F=[[pair.F(0, k) for k in range(pair.K)]] * pair.K,
        Sigma_W=[[pair.Sigma_W(0, k) for k in range(pair.K)]] * pair.K,
        pi0=pair.pi0,
        mu_z0_list=[pair.mu_z0(k) for k in range(pair.K)],
        Sigma_z0_list=[pair.Sigma_z0(k) for k in range(pair.K)],
        b=[[pair.b(0, k) for k in range(pair.K)]] * pair.K,
    ).to_arrival()
    _, _, ys = _simulate(arrival, 100, seed=5)
    assert GSSFilter(arrival).mode == "gpb2"  # the default of a GSSParams is unchanged
    _assert_same(_filter(arrival, ys, mode="cgo"), _filter(arrival, ys, mode="gpb2"))


def test_default_mode_of_pair_params():
    assert GSSFilter(_random_pair_params(500, cgo=True)).mode == "cgo"
    assert GSSFilter(_random_pair_params(500, cgo=False)).mode == "gpb2"


def test_cgo_warns_when_c_is_not_zero_and_filters_the_model_without_c():
    pair = _random_pair_params(600, K=2, q=1, s=1, cgo=False)
    K = pair.K
    F0 = np.array([[pair.F(j, k) for k in range(K)] for j in range(K)])
    F0[:, :, pair.q :, : pair.q] = 0.0
    frozen = PairGSSParams(
        K=K,
        q=pair.q,
        s=pair.s,
        P=pair.P,
        F=F0,
        Sigma_W=[[pair.Sigma_W(j, k) for k in range(K)] for j in range(K)],
        pi0=pair.pi0,
        mu_z0_list=[pair.mu_z0(k) for k in range(K)],
        Sigma_z0_list=[pair.Sigma_z0(k) for k in range(K)],
        b=[[pair.b(j, k) for k in range(K)] for j in range(K)],
    )
    _, _, ys = _simulate(pair, 60, seed=6)
    with pytest.warns(RuntimeWarning, match="assumes C = 0"):
        filt = GSSFilter(pair, mode="cgo")
    got = [filt.step(y) for y in ys]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning on a genuine CGO model
        ref = _filter(frozen, ys, mode="cgo")
    _assert_same(got, ref, atol=0.0)
    # and it differs from the filter of the model as given
    exact_ish = _filter(pair, ys, mode="gpb2")
    assert max(float(np.abs(a.E_x - b.E_x).max()) for a, b in zip(got, exact_ish)) > 1e-6


def test_ab_models_are_not_cgo():
    """The paper models M1-M3 have C != 0: the mode warns on them."""
    params = _params_from_dict(get_params("M1"))
    with pytest.warns(RuntimeWarning, match="assumes C = 0"):
        GSSFilter(params, mode="cgo")


def test_unreachable_regimes_do_not_break_the_recursion():
    """Deterministic alternation: half of the pairs have probability zero."""
    base = _random_pair_params(700, K=2, q=2, s=1)
    K = base.K
    pair = PairGSSParams(
        K=K,
        q=base.q,
        s=base.s,
        P=np.array([[0.0, 1.0], [1.0, 0.0]]),
        F=[[base.F(j, k) for k in range(K)] for j in range(K)],
        Sigma_W=[[base.Sigma_W(j, k) for k in range(K)] for j in range(K)],
        pi0=np.array([1.0, 0.0]),
        mu_z0_list=[base.mu_z0(k) for k in range(K)],
        Sigma_z0_list=[base.Sigma_z0(k) for k in range(K)],
    )
    _, _, ys = _simulate(pair, 40, seed=7)
    got, ref = _filter(pair, ys, mode="cgo"), _filter(pair, ys, mode="gpb2")
    assert all(np.all(np.isfinite(r.E_x)) and np.all(np.isfinite(r.E_xx)) for r in got)
    _assert_same(got, ref, atol=1e-9)
    assert np.allclose([r.pi[n % 2] for n, r in enumerate(got)], 1.0)


def test_reset_and_run_helpers():
    pair = _random_pair_params(800, K=3, q=1, s=1)
    _, _, ys = _simulate(pair, 50, seed=8)
    filt = GSSFilter(pair, mode="cgo")
    first = [filt.step(y) for y in ys]
    filt.reset()
    second = [filt.step(y) for y in ys]
    _assert_same(first, second, atol=0.0)
    _, df = GSSFilter(pair, mode="cgo").run(N=60, seed=1)
    _, df_ref = GSSFilter(pair, mode="gpb2").run(N=60, seed=1)
    assert np.allclose(df.to_numpy(), df_ref.to_numpy(), atol=1e-9)
    assert np.all(df["V_x_0"] > 0.0)


def test_posterior_variance_is_positive_definite():
    pair = _random_pair_params(900, K=3, q=3, s=2)
    _, _, ys = _simulate(pair, 150, seed=9)
    for r in _filter(pair, ys, mode="cgo"):
        assert np.linalg.eigvalsh(r.Var_x).min() > 0.0
