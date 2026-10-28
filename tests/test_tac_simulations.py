#!/usr/bin/env python3
"""
tests/test_tac_simulations.py
=============================
Checks of the self-contained filters of ``prg/experiments/tac_simulations.py`` (Section V of the
IEEE TAC paper) against brute-force enumeration of the regime paths:

* exact filter of the CGOMSM1 and of the CGOMSM2 (means and variances), scalar case of Section V-B;
* the vectorized enumeration ``exact_filter`` against a naive path-by-path Kalman bank;
* the exact CGOMSM1 / CGOMSM2 filters with a known input (Section V-C), including the
  log-likelihood, for an arbitrary (non-slaved) initial law;
* ``rbpf_c`` with K = 1 (Kalman filter) and the family checks of ``cgo_filter_c``;
* the closed forms of Section V-A against a Monte Carlo simulation of the dynamic form;
* the generated LaTeX tables and the integrity check of the dataset download.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from prg.experiments import tac_simulations as T

# --------------------------------------------------------------------------- brute force


def _lognorm(y, mu, var):
    return -0.5 * (np.log(2 * np.pi * var) + (y - mu) ** 2 / var)


def _brute_force(reg, P, p1, y, u=None):
    """E[X_n | y_{1:n}], Var[X_n | y_{1:n}] and log p(y_{1:N}) by a Kalman filter per regime path.

    ``reg`` is a list of per-regime dicts with a, b, c, d, su, rho, sv, S1 (2x2 initial covariance)
    and, optionally, input gains lx, ly (u[n] drives the transition from n - 1 to n).
    """
    K, N = len(reg), len(y)
    E, V = np.empty(N), np.empty(N)
    LL = None
    for n in range(N):
        logw, mus, Ps = [], [], []
        for p in itertools.product(range(K), repeat=n + 1):
            S1 = reg[p[0]]["S1"]
            lw = np.log(p1[p[0]]) + _lognorm(y[0], 0.0, S1[1, 1])
            m = S1[0, 1] / S1[1, 1] * y[0]
            Pv = S1[0, 0] - S1[0, 1] ** 2 / S1[1, 1]
            for t in range(1, n + 1):
                R = reg[p[t]]
                gx = R.get("lx", np.zeros(2)) @ u[t] if u is not None else 0.0
                gy = R.get("ly", np.zeros(2)) @ u[t] if u is not None else 0.0
                mx = R["a"] * m + R["b"] * y[t - 1] + gx
                my = R["c"] * m + R["d"] * y[t - 1] + gy
                sxx = R["a"] ** 2 * Pv + R["su"]
                sxy = R["a"] * R["c"] * Pv + R["rho"]
                syy = R["c"] ** 2 * Pv + R["sv"]
                lw += np.log(P[p[t - 1], p[t]]) + _lognorm(y[t], my, syy)
                m, Pv = mx + sxy / syy * (y[t] - my), sxx - sxy**2 / syy
            logw.append(lw)
            mus.append(m)
            Ps.append(Pv)
        logw, mus, Ps = map(np.array, (logw, mus, Ps))
        a = logw.max()
        w = np.exp(logw - a)
        LL = a + np.log(w.sum())
        w /= w.sum()
        E[n] = w @ mus
        V[n] = w @ (Ps + mus**2) - E[n] ** 2
    return E, V, LL


def _reg_from_b(m):
    """Section V-B model (dict of arrays) -> list of per-regime dicts for _brute_force."""
    return [
        dict(
            a=m["a"][j],
            b=m["b"][j],
            c=m["c"][j],
            d=m["d"][j],
            su=m["su"][j],
            rho=m["rho"][j],
            sv=m["sv"][j],
            S1=T.SZ1(m, j),
        )
        for j in range(2)
    ]


def _random_c_model(rng, K, family):
    """Random model in the format of Section V-C (input u_n = (u1, 1)), arbitrary initial law."""
    reg = []
    for _ in range(K):
        su, sv = rng.uniform(0.3, 1.0, 2)
        rho = rng.uniform(-0.6, 0.6) * np.sqrt(su * sv)
        c, d = rng.uniform(-0.8, 0.8, 2)
        a, b = rng.uniform(-0.8, 0.8, 2)
        M = rho / sv
        if family == "cgomsm1":
            c = 0.0
        elif family == "cgomsm2":
            a, b = M * c, M * d
        L = rng.uniform(-1, 1, (2, 2))
        reg.append(
            dict(
                a=a,
                b=b,
                c=c,
                d=d,
                su=su,
                rho=rho,
                sv=sv,
                lx=rng.uniform(-1, 1, 2),
                ly=rng.uniform(-1, 1, 2),
                S1=L @ L.T + 0.3 * np.eye(2),
            )
        )
    P = rng.dirichlet(np.ones(K) * 3, size=K)
    p1 = rng.dirichlet(np.ones(K) * 3)
    return {"K": K, "P": P, "p1": p1, "reg": reg}


def _simulate_c(m, N, rng):
    u = np.c_[rng.normal(size=N), np.ones(N)]
    r = rng.choice(m["K"], p=m["p1"])
    z = rng.multivariate_normal(np.zeros(2), m["reg"][r]["S1"])
    y = np.empty(N)
    y[0] = z[1]
    for n in range(1, N):
        r = rng.choice(m["K"], p=m["P"][r])
        R = m["reg"][r]
        w = rng.multivariate_normal(np.zeros(2), [[R["su"], R["rho"]], [R["rho"], R["sv"]]])
        z = np.array(
            [
                R["a"] * z[0] + R["b"] * z[1] + R["lx"] @ u[n] + w[0],
                R["c"] * z[0] + R["d"] * z[1] + R["ly"] @ u[n] + w[1],
            ]
        )
        y[n] = z[1]
    return y, u


# --------------------------------------------------------------------------- Section V-B filters


def test_exact_filter_matches_path_by_path_kalman():
    m = T.path(T.model_b(0.15, "B"), 0.4)
    _, _, y = T.simulate(m, 7, np.random.default_rng(3))
    e, v = T.exact_filter(m, y)
    eb, vb, _ = _brute_force(_reg_from_b(m), m["P"], m["p1"], y)
    np.testing.assert_allclose(e, eb, atol=1e-12)
    np.testing.assert_allclose(v, vb, atol=1e-12)


@pytest.mark.parametrize("ab", ["A", "B"])
@pytest.mark.parametrize("family", ["cgomsm1", "cgomsm2"])
def test_cgomsm_filters_are_exact(ab, family):
    m = T.model_b(0.05, ab)
    mf = T.cgomsm1(m) if family == "cgomsm1" else T.cgomsm2(m)
    filt = T.cgomsm1_filter if family == "cgomsm1" else T.cgomsm2_filter
    for s in range(5):
        _, _, y = T.simulate(mf, 10, np.random.default_rng(s))
        e, v = T.exact_filter(mf, y)
        f, fv = filt(mf, y, return_var=True)
        np.testing.assert_allclose(f, e, atol=1e-12)
        np.testing.assert_allclose(fv, v, atol=1e-12)
        np.testing.assert_array_equal(filt(mf, y), f)


def test_path_endpoints_are_the_two_families():
    m = T.model_b(0.15, "A")
    for key in ("a", "b", "c", "d"):
        np.testing.assert_array_equal(T.path(m, 0.0)[key], T.cgomsm1(m)[key])
        np.testing.assert_array_equal(T.path(m, 1.0)[key], T.cgomsm2(m)[key])


def test_rbpf_is_close_to_exact_filter():
    m = T.path(T.model_b(0.05, "A"), 0.5)
    assert T.validate_rbpf(m, n=8, seeds=5, n_part=500) < 5e-3


def test_crossing_and_ratio():
    e = np.linspace(0, 1, 5)
    assert T._crossing(e, np.array([-2.0, -1.0, 1.0, 2.0, 3.0])) == pytest.approx(0.375)
    assert np.isnan(T._crossing(e, np.ones(5)))
    num, den = np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0])
    R, hw = T._ratio_hw(num, den)
    assert R == pytest.approx(2.0)
    assert hw == pytest.approx(1.96 * np.std(num - 2.0 * den, ddof=1) / np.sqrt(3))


# --------------------------------------------------------------------------- Section V-C filters


@pytest.mark.parametrize("family", ["cgomsm1", "cgomsm2"])
@pytest.mark.parametrize("K", [2, 3])
def test_cgo_filter_c_matches_enumeration(family, K):
    rng = np.random.default_rng(10 * K + (family == "cgomsm2"))
    m = _random_c_model(rng, K, family)
    y, u = _simulate_c(m, 6, rng)
    E, LL = T.cgo_filter_c(m, y, u, family)
    Eb, _, LLb = _brute_force(m["reg"], m["P"], m["p1"], y, u)
    np.testing.assert_allclose(E, Eb, atol=1e-11)
    assert LL == pytest.approx(LLb, abs=1e-10)


def test_rbpf_c_with_one_regime_is_the_kalman_filter():
    rng = np.random.default_rng(7)
    m = _random_c_model(rng, 1, "thgmsm")
    y, u = _simulate_c(m, 30, rng)
    E, LL = T.rbpf_c(m, y, u, 1, np.random.default_rng(0))
    Eb, _, LLb = _brute_force(m["reg"], m["P"], m["p1"], y, u)
    np.testing.assert_allclose(E, Eb, atol=1e-11)
    assert LL == pytest.approx(LLb, abs=1e-9)


def test_cgo_filter_c_rejects_wrong_family_or_model():
    rng = np.random.default_rng(0)
    m2 = _random_c_model(rng, 2, "cgomsm2")
    y, u = _simulate_c(m2, 4, rng)
    with pytest.raises(ValueError, match="unknown family"):
        T.cgo_filter_c(m2, y, u, "imm")
    with pytest.raises(ValueError, match="CGOMSM1"):
        T.cgo_filter_c(m2, y, u, "cgomsm1")  # c != 0
    m1 = _random_c_model(rng, 2, "cgomsm1")
    with pytest.raises(ValueError, match="CGOMSM2"):
        T.cgo_filter_c(m1, y, u, "cgomsm2")


def test_fit_c_produces_constrained_models():
    rng = np.random.default_rng(1)
    seqs = [np.c_[rng.normal(size=(80, 3)), rng.uniform(0, 60, 80)] for _ in range(4)]
    edges = np.array([30.0])
    sc = np.ones(3)
    mods = T.fit_c(seqs, 2, edges, sc)
    T._check_family(mods["cgomsm1"], "cgomsm1")
    T._check_family(mods["cgomsm2"], "cgomsm2")
    np.testing.assert_allclose(mods["thgmsm"]["P"].sum(axis=1), 1.0)


# --------------------------------------------------------------------------- Section V-A


def test_closed_forms_of_section_va_match_monte_carlo():
    m = T.design_a(1.0, 0.05, 0.7)  # experiment 6
    mc = T.exp_a_mc(m, n=200_000, seed=0)
    assert abs(mc["tau1_z"]) < 4 and abs(mc["tau2_z"]) < 4
    closed = T.exp_a_closed(m)
    assert closed["tau1"] == pytest.approx(0.33636, abs=1e-5)
    assert closed["tau2"] == pytest.approx(0.0032424, abs=1e-6)


# --------------------------------------------------------------------------- tables and data


def test_tables_have_true_minus_signs_and_no_negative_zero(tmp_path):
    assert T._num(-0.001, ".2f") == "$0.00$"
    assert T._num(-0.5, "g") == "$-0.5$"
    rows = [
        dict(
            exp=k,
            a=[0.1, -0.2],
            b=[0.3, -0.4],
            c=[0.5, 0.0],
            d=[-0.5, -0.3],
            rho=[0.1, -0.2],
            tau1=1e-4 * k,
            tau2=2e-4,
        )
        for k in (1, 2)
    ]
    T._table_a(rows, tmp_path / "a.tex")
    txt = (tmp_path / "a.tex").read_text()
    assert "$-0.20$" in txt and " -0." not in txt and "$0.0002$" in txt
    assert "exp. & 1 & 2 \\\\" in txt  # same corner cell as Table II
    keys = ("cgomsm1", "cgomsm2", "thgmsm_rbpf", "kalman_K1")
    fit = {key: dict(rmse=1.0, loglik=-0.5) for key in keys}
    T._table_c({"main": {"input": fit, "no_input": fit}}, tmp_path / "c.tex")
    lines_c = (tmp_path / "c.tex").read_text().splitlines()
    assert lines_c[2].startswith(r"filter & \multicolumn{2}")
    assert lines_c[3] == r"\cmidrule(lr){2-3}"  # rule under the two-column heading
    assert r"CGOMSM1 (exact) & $1.00$ & $1.00$ & $-0.50$ \\" in lines_c
    res = {
        "1": dict(
            p12=0.05,
            a=[0.7, 0.4],
            b=[0.2, -0.1],
            max_excess_long_cgomsm1=dict(value=1.2438),
            max_excess_long_cgomsm2=dict(value=0.05243),
            eps_star_long=0.2019,
        )
    }
    T._table_b(res, tmp_path / "b.tex")
    assert (
        r"1 & $0.05$ & $0.7$ & $0.4$ & $0.2$ & $-0.1$ & $1.24$ & $0.052$ & $0.20$ \\"
        in (tmp_path / "b.tex").read_text()
    )
    json.dumps(res)  # the inputs of the tables are JSON-serializable


def test_dataset_download_is_checked(tmp_path, monkeypatch):
    from prg.experiments import make_vehicle_consigne_fig as mv

    class _Resp:
        def read(self):
            return b"not the archive"

    monkeypatch.setattr(mv.urllib.request, "urlopen", lambda *a, **k: _Resp())
    with pytest.raises(RuntimeError, match="MD5"):
        mv._ensure_data(tmp_path / "cache")
