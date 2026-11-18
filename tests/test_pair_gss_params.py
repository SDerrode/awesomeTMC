#!/usr/bin/env python3
"""
tests/test_pair_gss_params.py
=============================
Parameters indexed by the regime pair (r_n, r_{n+1}): ``PairGSSParams``, and its
support in ``GSSSimulator`` and in ``GSSFilter(mode="gpb2")``.

Three guarantees are checked:

* nothing changes for arrival-indexed models: parameters broadcast with
  ``as_pair()`` give the same simulated trajectory and the same filter output,
  bit for bit;
* on a genuinely pair-dependent model the filter equals two independent
  references, the enumeration of all regime paths (exact when C = 0) and the
  arrival-indexed filter run on the K^2 augmented regimes;
* invalid parameters are rejected.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest
from scipy.special import logsumexp

from awesometmc.classes import GSSParams, GSSSimulator, PairGSSParams
from awesometmc.experiments.models_paper import get_params
from awesometmc.experiments.run_simulations import _params_from_dict
from awesometmc.filter import GSSFilter
from awesometmc.models.model_gss_K2_q1_s1_consigne import ModelGssK2Q1S1Consigne
from awesometmc.utils.exceptions import CovarianceError, ParamError


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _spd(rng, n, scale=1.0):
    a = rng.standard_normal((n, n))
    return scale * (a @ a.T / n + 0.5 * np.eye(n))


def _random_pair_params(seed, K=2, q=1, s=1, cgo=True, p=0, bias=True) -> PairGSSParams:
    """A stable pair-dependent model; C(j, k) = 0 for every pair when ``cgo``."""
    rng = np.random.default_rng(seed)
    dim = q + s
    F = 0.35 * rng.standard_normal((K, K, dim, dim))
    if cgo:
        F[:, :, q:, :q] = 0.0
    SW = np.array([[_spd(rng, dim) for _ in range(K)] for _ in range(K)])
    P = rng.random((K, K)) + 0.2
    P /= P.sum(axis=1, keepdims=True)
    return PairGSSParams(
        K=K,
        q=q,
        s=s,
        P=P,
        F=F,
        Sigma_W=SW,
        pi0=None,
        mu_z0_list=[rng.standard_normal((dim, 1)) for _ in range(K)],
        Sigma_z0_list=[_spd(rng, dim) for _ in range(K)],
        b=rng.standard_normal((K, K, dim, 1)) if bias else None,
        G=rng.standard_normal((K, K, dim, p)) if p else None,
    )


def _simulate(params, N, seed, u=None):
    rs, xs, ys = [], [], []
    for _, r, x, y in GSSSimulator(params, N=N, seed=seed, u=u):
        rs.append(r)
        xs.append(x.copy())
        ys.append(y.copy())
    return np.array(rs), np.array(xs), np.array(ys)


def _filter(params, ys, us=None, mode="gpb2"):
    filt = GSSFilter(params, mode=mode)
    out = [filt.step(y, u=None if us is None else us[n]) for n, y in enumerate(ys)]
    return out


def _enumerate(params: PairGSSParams, ys):
    """Exact (E[X_N | y], P(r_N | y), log p(y)) by enumeration of the K^N regime paths."""
    K, q, s = params.K, params.q, params.s
    H = np.hstack([np.zeros((s, q)), np.eye(s)])

    def update(z, Pc, y):
        S = H @ Pc @ H.T
        nu = y - H @ z
        gain = Pc @ H.T @ np.linalg.inv(S)
        ll = (
            -0.5 * (nu.T @ np.linalg.solve(S, nu)).item()
            - 0.5 * np.linalg.slogdet(2 * np.pi * S)[1]
        )
        return z + gain @ nu, (np.eye(q + s) - gain @ H) @ Pc, ll

    log_w, means, last = [], [], []
    for path in itertools.product(range(K), repeat=len(ys)):
        k = path[0]
        z, Pc, ll = update(params.mu_z0(k), params.Sigma_z0(k), ys[0])
        lw = np.log(params.pi0[k]) + ll
        for n in range(1, len(ys)):
            j, k = path[n - 1], path[n]
            F = params.F(j, k)
            z, Pc, ll = update(F @ z + params.b(j, k), F @ Pc @ F.T + params.Sigma_W(j, k), ys[n])
            lw += np.log(params.P[j, k]) + ll
        log_w.append(lw)
        means.append(z[:q, 0])
        last.append(k)
    log_w = np.array(log_w)
    total = float(logsumexp(log_w))
    w = np.exp(log_w - total)
    return w @ np.array(means), np.bincount(last, weights=w, minlength=K), total


ARRIVAL_MODELS = ["M1", "M2", "M3"]


def _arrival(name) -> GSSParams:
    return _params_from_dict(get_params(name))


# --------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ARRIVAL_MODELS)
def test_as_pair_broadcasts_and_round_trips(name):
    params = _arrival(name)
    pair = params.as_pair()
    assert isinstance(pair, PairGSSParams) and pair.is_arrival_only()
    assert (pair.K, pair.q, pair.s, pair.p) == (params.K, params.q, params.s, params.p)
    for j, k in itertools.product(range(params.K), repeat=2):
        assert np.array_equal(pair.F(j, k), params.f_matrix.F(k))
        assert np.array_equal(pair.Sigma_W(j, k), params.noise_cov.Sigma_W(k))
        assert np.array_equal(pair.b(j, k), params.b(k))
    back = pair.to_arrival()
    for k in range(params.K):
        assert np.array_equal(back.f_matrix.F(k), params.f_matrix.F(k))
        assert np.array_equal(back.noise_cov.Sigma_W(k), params.noise_cov.Sigma_W(k))
        assert np.array_equal(back.b(k), params.b(k))
    assert np.array_equal(back.P, params.P) and np.array_equal(back.pi0, params.pi0)


def test_pair_dependent_params_have_no_arrival_form():
    pair = _random_pair_params(0)
    assert not pair.is_arrival_only()
    with pytest.raises(ParamError):
        pair.to_arrival()
    assert "pair-dependent" in repr(pair)


def test_is_cgo():
    assert _random_pair_params(1, cgo=True).is_cgo()
    assert not _random_pair_params(1, cgo=False).is_cgo()


def test_augmented_chain_is_consistent():
    pair = _random_pair_params(2, K=3, q=2, s=1)
    aug = pair.augmented()
    K = pair.K
    assert aug.K == K * K and np.allclose(aug.P.sum(axis=1), 1.0)
    for (i, j), (k, l) in itertools.product(itertools.product(range(K), repeat=2), repeat=2):
        expected = pair.P[j, l] if k == j else 0.0
        assert aug.P[i * K + j, k * K + l] == expected
    assert np.allclose(aug.pi0.reshape(K, K).sum(axis=0), pair.pi0)
    assert np.array_equal(aug.f_matrix.F(1 * K + 2), pair.F(1, 2))
    assert np.array_equal(aug.b(2 * K + 0), pair.b(2, 0))


# --------------------------------------------------------------------------
# Nothing changes for arrival-indexed models
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ARRIVAL_MODELS)
def test_simulator_is_bit_identical_on_broadcast_params(name):
    params = _arrival(name)
    ref, got = _simulate(params, 300, seed=7), _simulate(params.as_pair(), 300, seed=7)
    for a, b in zip(ref, got):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("name", ARRIVAL_MODELS)
def test_gpb2_is_bit_identical_on_broadcast_params(name):
    params = _arrival(name)
    _, _, ys = _simulate(params, 150, seed=3)
    ref, got = _filter(params, ys), _filter(params.as_pair(), ys)
    for a, b in zip(ref, got):
        assert np.array_equal(a.E_x, b.E_x) and np.array_equal(a.E_xx, b.E_xx)
        assert np.array_equal(a.pi, b.pi) and np.array_equal(a.innovation, b.innovation)
        assert a.log_lik == b.log_lik


def test_exogenous_input_is_bit_identical_on_broadcast_params():
    params = GSSParams.from_model(ModelGssK2Q1S1Consigne())
    assert params.p > 0
    N = 120
    u = np.random.default_rng(0).standard_normal((N, params.p))
    ref, got = _simulate(params, N, seed=5, u=u), _simulate(params.as_pair(), N, seed=5, u=u)
    for a, b in zip(ref, got):
        assert np.array_equal(a, b)
    fa, fb = _filter(params, ref[2], us=u), _filter(params.as_pair(), ref[2], us=u)
    for a, b in zip(fa, fb):
        assert np.array_equal(a.E_x, b.E_x) and a.log_lik == b.log_lik


# --------------------------------------------------------------------------
# Genuinely pair-dependent models
# --------------------------------------------------------------------------
def test_simulator_uses_the_parameters_of_the_pair():
    """Deterministic alternation 0 -> 1 -> 0 and negligible noise: each step must follow
    z_n = F(r_{n-1}, r_n) z_{n-1} + b(r_{n-1}, r_n)."""
    rng = np.random.default_rng(4)
    F = 0.5 * rng.standard_normal((2, 2, 2, 2))
    b = rng.standard_normal((2, 2, 2, 1))
    pair = PairGSSParams(
        K=2,
        q=1,
        s=1,
        P=np.array([[0.0, 1.0], [1.0, 0.0]]),
        F=F,
        Sigma_W=np.broadcast_to(1e-18 * np.eye(2), (2, 2, 2, 2)),
        pi0=np.array([1.0, 0.0]),
        mu_z0_list=[np.ones((2, 1)), np.ones((2, 1))],
        Sigma_z0_list=[np.eye(2), np.eye(2)],
        b=b,
    )
    rs, xs, ys = _simulate(pair, 12, seed=0)
    assert list(rs) == [0, 1] * 6
    z = np.concatenate([xs, ys], axis=1)
    for n in range(1, 12):
        j, k = rs[n - 1], rs[n]
        assert np.allclose(z[n], F[j, k] @ z[n - 1] + b[j, k], atol=1e-7)
    assert not np.allclose(F[0, 1], F[1, 0])


@pytest.mark.parametrize("K,q,s,N", [(2, 1, 1, 8), (3, 1, 1, 5), (2, 2, 1, 7), (2, 1, 2, 7)])
def test_gpb2_is_exact_on_pair_dependent_cgo(K, q, s, N):
    """C(j, k) = 0 for every pair: the order-2 collapse equals the path enumeration."""
    pair = _random_pair_params(10 + K + q + s, K=K, q=q, s=s, cgo=True)
    _, _, ys = _simulate(pair, N, seed=1)
    out = _filter(pair, ys)
    e_x, pi, log_lik = _enumerate(pair, ys)
    assert np.allclose(out[-1].E_x.ravel(), e_x, atol=1e-10)
    assert np.allclose(out[-1].pi, pi, atol=1e-10)
    assert sum(r.log_lik for r in out) == pytest.approx(log_lik, abs=1e-9)


def test_gpb2_is_only_approximate_when_c_is_not_zero():
    pair = _random_pair_params(21, cgo=False)
    _, _, ys = _simulate(pair, 8, seed=1)
    out = _filter(pair, ys)
    e_x, _, log_lik = _enumerate(pair, ys)
    assert np.all(np.isfinite(out[-1].E_x))
    assert abs(sum(r.log_lik for r in out) - log_lik) > 1e-8


@pytest.mark.parametrize("cgo", [True, False])
@pytest.mark.parametrize("p", [0, 1])
def test_gpb2_equals_the_filter_on_the_augmented_regimes(cgo, p):
    """Oracle: arrival-indexed GPB2 on S_n = (r_{n-1}, r_n). It collapses one step later
    than the pair filter, so the two coincide when the pair filter is exact (C = 0)."""
    pair = _random_pair_params(30 + p, K=2, q=2, s=1, cgo=cgo, p=p)
    N, K = 40, pair.K
    u = np.random.default_rng(1).standard_normal((N, p)) if p else None
    _, _, ys = _simulate(pair, N, seed=2, u=u)
    got = _filter(pair, ys, us=u)
    ref = _filter(pair.augmented(), ys, us=u)
    gap = max(float(np.abs(a.E_x - b.E_x).max()) for a, b in zip(got, ref))
    if cgo:
        assert gap < 1e-9
        for a, b in zip(got, ref):
            assert np.allclose(a.pi, b.pi.reshape(K, K).sum(axis=0), atol=1e-10)
            assert a.log_lik == pytest.approx(b.log_lik, abs=1e-9)
    else:
        assert 1e-9 < gap < 1.0


def test_filter_run_accepts_pair_params():
    pair = _random_pair_params(40, K=2, q=1, s=1)
    _, df = GSSFilter(pair).run(N=50, seed=0)
    assert len(df) == 50 and np.all(np.isfinite(df["E_x_0"]))
    assert np.allclose(df[["p_r_0", "p_r_1"]].sum(axis=1), 1.0)


# --------------------------------------------------------------------------
# Mode dispatch and validation
# --------------------------------------------------------------------------
def test_default_mode_and_ngh_kf_is_rejected():
    pair = _random_pair_params(50)
    # C = 0 for every pair: the exact fast mode; otherwise the general one
    assert GSSFilter(pair).mode == "cgo"
    assert GSSFilter(_random_pair_params(50, cgo=False)).mode == "gpb2"
    with pytest.raises(ValueError, match="ngh_kf"):
        GSSFilter(pair, mode="ngh_kf")
    # arrival-only pair parameters can go back to the constant-gain filter
    params = _arrival("M1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert GSSFilter(params.as_pair().to_arrival(), mode="ngh_kf").mode == "ngh_kf"


def test_invalid_parameters_are_rejected():
    good = _random_pair_params(60)
    K, dim = good.K, good.dim_z
    base = dict(
        K=K,
        q=good.q,
        s=good.s,
        P=good.P,
        F=np.array([[good.F(j, k) for k in range(K)] for j in range(K)]),
        Sigma_W=np.array([[good.Sigma_W(j, k) for k in range(K)] for j in range(K)]),
        pi0=None,
        mu_z0_list=[good.mu_z0(k) for k in range(K)],
        Sigma_z0_list=[good.Sigma_z0(k) for k in range(K)],
    )
    PairGSSParams(**base)  # sanity: the base dictionary is valid
    with pytest.raises(ParamError):
        PairGSSParams(**{**base, "F": base["F"][0]})  # arrival-shaped F
    with pytest.raises(ParamError):
        PairGSSParams(**{**base, "P": np.array([[0.5, 0.6], [0.5, 0.5]])})
    with pytest.raises(ParamError):
        PairGSSParams(**{**base, "b": np.zeros((K, K, dim + 1, 1))})
    with pytest.raises(ParamError):
        PairGSSParams(**{**base, "G": np.zeros((K, dim, 1))})
    bad = base["Sigma_W"].copy()
    bad[0, 1] = -np.eye(dim)
    with pytest.raises(CovarianceError):
        PairGSSParams(**{**base, "Sigma_W": bad})
    nan = base["F"].copy()
    nan[1, 0, 0, 0] = np.nan
    with pytest.raises(ParamError):
        PairGSSParams(**{**base, "F": nan})


def test_bias_accepts_vectors_and_columns():
    good = _random_pair_params(61)
    K, dim = good.K, good.dim_z
    b = np.arange(K * K * dim, dtype=float).reshape(K, K, dim)
    kw = dict(
        K=K,
        q=good.q,
        s=good.s,
        P=good.P,
        F=np.array([[good.F(j, k) for k in range(K)] for j in range(K)]),
        Sigma_W=np.array([[good.Sigma_W(j, k) for k in range(K)] for j in range(K)]),
        pi0=good.pi0,
        mu_z0_list=[good.mu_z0(k) for k in range(K)],
        Sigma_z0_list=[good.Sigma_z0(k) for k in range(K)],
    )
    a, c = PairGSSParams(**kw, b=b), PairGSSParams(**kw, b=b[..., None])
    assert a.b(1, 0).shape == (dim, 1) and np.array_equal(a.b(1, 0), c.b(1, 0))
