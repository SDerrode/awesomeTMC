#!/usr/bin/env python3
"""
tests/test_tac_stationary.py
============================
Checks of ``awesometmc/experiments/tac_stationary.py`` (stationary THGMSMs, Section V of the IEEE TAC
paper):

* construction of the data-generating THGMSM: stationarity, prescribed conditional autoregression,
  coefficients of the two families, positive definite noise covariances;
* the four-path estimate of X_3 against the enumeration of all the regime paths of
  ``tac_simulations.exact_filter`` and against the library (``exact_mixture_filter``,
  ``GSSFilter(mode="gpb2")``, ``apply_AB_constraint``), for the THGMSM and its three approximations;
* the properties used in the paper: the law of Y_{1:3} does not depend on gamma in the family M, the
  two CGOMSM1s have the same regime posteriors, the regression of a CGOMSM2 reduces to M_{r_3} y_3;
* the quadrature: identities between the two computations of each error, published values at the
  ends of the two families, agreement with a Monte Carlo simulation;
* the sensitivity plan and the outputs of case ``e``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from awesometmc.experiments import tac_simulations as T
from awesometmc.experiments import tac_stationary as S

POINTS = [("M", 0.0), ("M", 0.5), ("N", 0.4), ("N", 1.0)]


# --------------------------------------------------------------------------- construction
@pytest.mark.parametrize("family,t", [(f, t) for f in S.FAMILIES for t in S.FAMILIES[f]["grid"]])
def test_generator_is_stationary_with_prescribed_coefficients(family, t):
    f = S.FAMILIES[family]
    m = S.family_model(family, t)
    Sigma = T.SZ1(m, 0)
    for j in range(S.K):
        np.testing.assert_allclose(T.F(m, j) @ Sigma @ T.F(m, j).T + T.SW(m, j), Sigma, atol=1e-14)
        assert np.linalg.eigvalsh(T.SW(m, j)).min() > 0
    np.testing.assert_allclose(m["p1"] @ m["P"], m["p1"], atol=1e-15)
    np.testing.assert_allclose(np.diag(m["P"]), (1 + S.CHI) / 2)
    ga, be = S.conditional_coefficients(m)
    s = np.sqrt(1 - f["lam"] ** 2)
    np.testing.assert_allclose(m["c"], np.array(f["w"]) / s, atol=1e-14)
    if family == "M":
        np.testing.assert_allclose(ga, t, atol=1e-14)
    else:
        np.testing.assert_allclose(ga, 0.0, atol=1e-14)
        np.testing.assert_allclose(be, s * t * np.array(f["beta0"]), atol=1e-14)
    assert np.linalg.norm(S.family_Ft(family, t), 2, axis=(1, 2)).max() < 0.91


def test_inadmissible_generator_is_rejected():
    with pytest.raises(ValueError):
        S.make_thgmsm(0.5, np.array([[[0.9, 0.9], [0.0, 0.5]]] * 2))
    with pytest.raises(ValueError):
        S.make_thgmsm(1.0, S.family_Ft("M", 0.0))


def test_approximations_change_the_prescribed_entries_only():
    m = S.family_model("M", 0.3)
    ap = S.approximations(m)
    M = m["rho"] / m["sv"]
    assert np.all(ap["1"]["c"] == 0) and np.all(ap["1p"]["c"] == 0) and np.all(ap["1p"]["a"] == 0)
    np.testing.assert_allclose(ap["2"]["a"], M * m["c"])
    np.testing.assert_allclose(ap["2"]["b"], M * m["d"])
    for name, keys in (("1", "abd"), ("2", "cd"), ("1p", "bd")):
        for key in list(keys) + ["su", "rho", "sv", "sx1", "sy1", "lam", "P", "p1"]:
            np.testing.assert_array_equal(ap[name][key], m[key])


# --------------------------------------------------------------------------- exact estimates
@pytest.mark.parametrize("family,t", POINTS)
def test_estimate_matches_enumeration_of_all_regime_paths(family, t):
    rng = np.random.default_rng(0)
    ys = rng.normal(size=(20, 3)) * 1.5
    for name, m in S.approximations(S.family_model(family, t)).items():
        ref = np.array([T.exact_filter(m, y)[0][-1] for y in ys])
        np.testing.assert_allclose(S.estimate(m, ys), ref, atol=1e-12, err_msg=name)


def _library_params(m):
    from awesometmc.classes.FMatrix import FMatrix
    from awesometmc.classes.GSSParams import GSSParams
    from awesometmc.classes.NoiseCovariance import GSSNoiseCovariance

    def block(key):
        return [np.array([[m[key][j]]]) for j in range(S.K)]

    return GSSParams(
        K=S.K,
        q=1,
        s=1,
        P=m["P"],
        f_matrix=FMatrix(
            K=S.K,
            q=1,
            s=1,
            A_list=block("a"),
            B_list=block("b"),
            C_list=block("c"),
            D_list=block("d"),
        ),
        noise_cov=GSSNoiseCovariance(
            K=S.K,
            q=1,
            s=1,
            Sigma_U_list=block("su"),
            Delta_list=block("rho"),
            Sigma_V_list=block("sv"),
        ),
        pi0=m["p1"],
        mu_z0_list=[np.zeros((2, 1)) for _ in range(S.K)],
        Sigma_z0_list=[T.SZ1(m, i) for i in range(S.K)],
    )


@pytest.mark.parametrize("family,t", [("M", 0.5), ("N", 1.0)])
def test_estimate_matches_the_library_filters(family, t):
    from awesometmc.experiments.reference_filters import exact_mixture_filter
    from awesometmc.filter.gss_filter import GSSFilter
    from awesometmc.utils.ab_constraint import apply_AB_constraint

    m = S.family_model(family, t)
    ap = S.approximations(m)
    params = {name: _library_params(v) for name, v in ap.items()}
    # the CGOMSM2 is the AB projection of the library
    proj = apply_AB_constraint(params["G"])
    for j in range(S.K):
        np.testing.assert_allclose(proj.f_matrix.F(j), T.F(ap["2"], j), atol=1e-13)
    rng = np.random.default_rng(1)
    for y in rng.normal(size=(5, 3)):
        for name, p in params.items():
            mine = S.estimate(ap[name], y)[0]
            E, _, _ = exact_mixture_filter(p, y[:, None])
            assert E[-1, 0] == pytest.approx(mine, abs=1e-10), name
            flt = GSSFilter(p, mode="gpb2")  # exact at n = 3: Sigma_{Z_1,i} does not depend on i
            for obs in y:
                out = flt.step(np.array([[obs]]))
            assert float(out.E_x[0, 0]) == pytest.approx(mine, abs=1e-10), name


def test_observation_law_does_not_depend_on_gamma_in_family_M():
    ref = S.components(S.family_model("M", 0.0))
    for t in S.FAMILIES["M"]["grid"][1:]:
        comp = S.components(S.family_model("M", t))
        np.testing.assert_allclose(comp["chol"], ref["chol"], atol=1e-14)
        np.testing.assert_allclose(comp["prior"], ref["prior"], atol=1e-15)


@pytest.mark.parametrize("family,t", POINTS)
def test_regime_posteriors_and_regressions_of_the_approximations(family, t):
    m = S.family_model(family, t)
    ap = S.approximations(m)
    comps = {name: S.components(v) for name, v in ap.items()}
    y = np.random.default_rng(2).normal(size=(50, 3))
    # the law of (R, Y) of a CGOMSM1 does not depend on a_j
    np.testing.assert_allclose(
        S.posterior(comps["1"], y)[0], S.posterior(comps["1p"], y)[0], atol=1e-13
    )
    # CGOMSM2: E[X_3 | r, y_{1:3}] = M_{r_3} y_3
    M = m["rho"] / m["sv"]
    expected = np.array([[0.0, 0.0, M[k]] for _, k in S.PATHS])
    np.testing.assert_allclose(comps["2"]["reg"], expected, atol=1e-13)
    # family N_beta: gamma = 0, so that the CGOMSM1 without autoregression has the regressions of G
    if family == "N":
        np.testing.assert_allclose(comps["1p"]["reg"], comps["G"]["reg"], atol=1e-13)


# --------------------------------------------------------------------------- quadrature
PUBLISHED = {  # ER, R1, R1p, R2, R0, 100 Delta, contributions (regression, weight, interaction)
    ("M", 0.0): (0.567, 0.596, 0.579, 0.628, 0.64, -5.65, (-9.6, 3.4, 0.6)),
    ("M", 0.5): (0.539, 0.560, 0.653, 0.634, 0.64, -13.82, (-16.8, 2.9, 0.1)),
    ("N", 0.4): (0.740, 0.770, 0.754, 0.749, 0.75, 2.94, (0.9, 6.0, -4.0)),
    ("N", 1.0): (0.688, 0.797, 0.780, 0.740, 0.75, 8.35, (1.2, 23.4, -16.2)),
}


@pytest.mark.parametrize("family,t", POINTS)
def test_quadrature_reproduces_the_published_values(family, t):
    r = S.integrate(S.family_model(family, t), 32)
    ER, R1, R1p, R2, R0, delta, terms = PUBLISHED[(family, t)]
    assert r["check"] < 1e-5
    for key, ref in (("ER", ER), ("R1", R1), ("R1p", R1p), ("R2", R2), ("R0", R0)):
        assert r[key] == pytest.approx(ref, abs=6e-4), key
    assert 100 * r["delta"] == pytest.approx(delta, abs=6e-3)
    for key, ref in zip(("reg", "wgt", "int"), terms):
        assert 100 * (r[f"{key}1"] - r[f"{key}2"]) == pytest.approx(ref, abs=6e-2), key
    for l in S.MODELS:
        assert r[f"tau{l}"] == pytest.approx(r[f"reg{l}"] + r[f"wgt{l}"] + r[f"int{l}"], abs=1e-9)


def test_converge_stops_after_two_stable_refinements():
    m = S.family_model("N", 0.4)
    r = S.converge(m, orders=(8, 12, 16, 24, 32, 48), tol=1e-5)
    assert r["converged"] and r["change"] < 1e-5 and r["order"] <= 48
    coarse = S.converge(m, orders=(4, 6), tol=1e-12)
    assert not coarse["converged"] and coarse["order"] == 6


def test_errors_agree_with_a_monte_carlo_simulation():
    m = S.family_model("N", 1.0)
    ap = S.approximations(m)
    rng = np.random.default_rng(3)
    n = 400_000
    r = np.empty((n, 3), dtype=int)
    r[:, 0] = rng.random(n) < m["p1"][1]
    Z = rng.multivariate_normal(np.zeros(2), T.SZ1(m, 0), size=n)
    ys = [Z[:, 1].copy()]
    for step in (1, 2):
        r[:, step] = rng.random(n) < m["P"][r[:, step - 1], 1]
        W = np.empty((n, 2))
        Zn = np.empty((n, 2))
        for j in range(S.K):
            sel = r[:, step] == j
            W[sel] = rng.multivariate_normal(np.zeros(2), T.SW(m, j), size=sel.sum())
            Zn[sel] = Z[sel] @ T.F(m, j).T + W[sel]
        Z = Zn
        ys.append(Z[:, 1].copy())
    y, x3 = np.column_stack(ys), Z[:, 0]
    ref = S.integrate(m, 32)
    for name, key in (("G", "ER"), ("1", "R1"), ("2", "R2"), ("1p", "R1p")):
        err = (x3 - S.estimate(ap[name], y)) ** 2
        assert err.mean() == pytest.approx(ref[key], abs=4 * err.std() / np.sqrt(n)), name
    err0 = (x3 - m["lam"][0] * y[:, 2]) ** 2
    assert err0.mean() == pytest.approx(ref["R0"], abs=4 * err0.std() / np.sqrt(n))


# --------------------------------------------------------------------------- sensitivity, outputs
def test_sensitivity_plan_has_52_admissible_perturbations_per_family():
    plan = S.sensitivity_plan()
    for family in S.FAMILIES:
        cases = [c for c in plan if c["family"] == family]
        assert len(cases) == 52
        assert len({(c["t"], c["label"]) for c in cases}) == 52
        for c in cases:
            m = S._perturbed_model(c)
            ga, _ = S.conditional_coefficients(m)
            if "direct" not in c:  # structural perturbations keep the prescribed gamma
                np.testing.assert_allclose(ga, c["t"] if family == "M" else 0.0, atol=1e-13)


def _small_grids(monkeypatch):
    small = {
        "M": {**S.FAMILIES["M"], "grid": (0.0, 0.5)},
        "N": {**S.FAMILIES["N"], "grid": (0.4, 1.0)},
    }
    monkeypatch.setattr(S, "FAMILIES", small)


def test_case_e_writes_results(tmp_path, monkeypatch):
    _small_grids(monkeypatch)
    monkeypatch.setattr(S, "_plot_e", lambda res, path: None)  # matplotlib is optional
    res = S.exp_e(tmp_path, orders=(8, 12, 16, 24), tol=1e-3)
    stored = json.loads((tmp_path / "exp_e.json").read_text())
    assert [(r["family"], r["t"]) for r in stored["rows"]] == POINTS
    assert all(r["converged"] for r in res["rows"])
    assert stored["rows"] == res["rows"]


def test_case_e_figure(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    _small_grids(monkeypatch)
    S.exp_e(tmp_path, orders=(8, 12, 16, 24), tol=1e-3)
    assert (tmp_path / "fig_e_stationary.pdf").stat().st_size > 5000
    (tmp_path / "fig_e_stationary.pdf").unlink()
    S.replot(tmp_path)
    assert (tmp_path / "fig_e_stationary.pdf").exists()
