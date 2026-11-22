#!/usr/bin/env python3
"""Sections V-A to V-C of the TAC paper (case ``e`` of ``tac_simulations``): stationary THGMSMs.

CGOMSM1s and CGOMSM2s are compared as approximate models of a THGMSM G that belongs to neither
family. The data follow G, all the parameters are known, the regimes are hidden, and X_3 is estimated
from y_{1:3}. Scalar case q = s = 1, K = 2 regimes. Nothing is simulated: the estimates are exact
Gaussian-mixture computations and their mean square errors are computed by quadrature.

Rewriting, with the notation of the paper, of the study of U. Hubert-Almuzara (package
``stationary_cgomsm``, 2026-11-17); the families, the criteria and the stopping rule are his.

Data-generating THGMSM G (dynamic form of ``tac_simulations``, regime j = R_{n+1}):
    [X_{n+1}; Y_{n+1}] = F_j [X_n; Y_n] + W_{n+1},  F_j = [[a_j, b_j], [c_j, d_j]],
    Sigma_{Z_1,i} = Sigma = [[1, lam], [lam, 1]] = L L^T,  L = [[sqrt(1 - lam^2), lam], [0, 1]],
    F_j = L Ft_j L^{-1},  Sigma_{W,j} = Sigma - F_j Sigma F_j^T = L (I - Ft_j Ft_j^T) L^T,
    Ft_j = [[((1 - w_j^2 - z_j^2) gamma - v_j w_j z_j) / (1 - z_j^2), v_j], [w_j, z_j]],
    p(r_1) = (1/2, 1/2),  p(r_{n+1} = j | r_n = i) = (1 + chi)/2 if j = i, (1 - chi)/2 otherwise.
Ft_j ("F tilde") is the transition of the normalized variable L^{-1} Z_n, whose covariance is I;
Sigma_{W,j} is positive definite iff the spectral norm of Ft_j is below 1, and (R_n, Z_n) is then
stationary. With M_j = rho_j / sv_j,
    X_{n+1} = M_j Y_{n+1} + (a_j - M_j c_j) X_n + (b_j - M_j d_j) Y_n + xi_{n+1},
and the first entry of Ft_j is such that a_j - M_j c_j = gamma in both regimes: gamma is the
conditional autoregression of the state. Moreover c_j = w_j / sqrt(1 - lam^2).

Two families (``FAMILIES``), chi = 0.8:
    M      : gamma in {0, 0.05, ..., 0.5}, the other parameters fixed. The law of (R_{1:3}, Y_{1:3})
             does not depend on gamma.
    N_beta : gamma = 0 and v_j = r beta0_j (1 - z_j^2), so that b_j - M_j d_j = sqrt(1 - lam^2) r
             beta0_j, with the scale factor r in {0.4, 0.5, ..., 1}.

Approximate models (same regime law, Sigma_{Z_1,i} and Sigma_{W,j} as G):
    CGOMSM1  ("1") : c_j := 0;
    CGOMSM2  ("2") : a_j := M_j c_j, b_j := M_j d_j;
    CGOMSM1' ("1p"): c_j := 0 and a_j := 0 (CGOMSM1 without autoregression).
Benchmark ("0"): the memoryless estimate E_G[X_3 | y_3] = lam y_3, of error 1 - lam^2.

Estimates. Sigma_{Z_1,i} does not depend on i, so that, in every model l, (X_3, Y_{1:3}) given the
regimes depends on (r_2, r_3) only, and
    xhat_l = E_l[X_3 | y_{1:3}] = sum_{r_2, r_3} p_l(r_2, r_3 | y_{1:3}) m_l(r_2, r_3)^T y_{1:3},
where m_l(r_2, r_3)^T y_{1:3} is the Gaussian regression of X_3 on Y_{1:3} given the regimes.

Criteria, all under G:
    ER = E_G[(X_3 - xhat_G)^2]  (optimal error),   tau_l = E_G[(xhat_l - xhat_G)^2] / ER,
    R_l = ER (1 + tau_l) = E_G[(X_3 - xhat_l)^2],  Delta = tau_1 - tau_2
(Delta < 0: the CGOMSM1 is the better approximation; Delta > 0: the CGOMSM2 is). Decomposition of
xhat_l - xhat_G = e_reg + e_w into a regression error, computed with the regime posteriors of G, and
a weight error:
    e_reg = sum p_G(r_2, r_3 | y) (m_l - m_G)(r_2, r_3)^T y,
    e_w   = sum (p_l - p_G)(r_2, r_3 | y) m_l(r_2, r_3)^T y,
    tau_l = (E_G[e_reg^2] + E_G[e_w^2] + 2 E_G[e_reg e_w]) / ER = reg_l + wgt_l + int_l.

Quadrature. Under G, Y_{1:3} is a mixture of four centered Gaussian laws; every expectation is
computed by a tensor Gauss-Hermite rule on each of them, with ``ORDERS`` nodes per dimension in
turn, until all the reported values change by less than ``TOL`` on ``N_STABLE`` successive
refinements. Each R_l is also computed directly, as E_G[Var_G(X_3 | r, y) + (E_G[X_3 | r, y] -
xhat_l)^2]; the largest gap between the two computations is stored as ``check``.

Case ``e`` (exp_e.json, fig_e_stationary.pdf): the two families and the decomposition at their ends.
Case ``sensitivity`` (exp_e_sensitivity.json): one-at-a-time perturbations around two points of each
    family: v_j (or beta0_j), w_j, z_j by +-0.02 and lam, p(r_1 = 1), chi by +-0.05, which keep
    the prescribed gamma; and, at the last point, each entry of Ft_j by +-0.01, which does not. 52
    perturbations per family. The summary gives the ranges of 100 Delta and of the errors, and the
    number of perturbations that reverse a ranking of the unperturbed point.
Case ``replot``: figure from the stored exp_e.json.

Run:  python -m awesometmc.experiments.tac_stationary [e|sensitivity|all|replot] [--out DIR]
      (also ``python -m awesometmc.experiments.tac_simulations e``)
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.special import roots_hermitenorm

from awesometmc.experiments.tac_simulations import (
    OUT,
    SW,
    SZ1,
    F,
    _mpl,
    _savefig,
    cgomsm1,
    cgomsm2,
    make_model,
)

K = 2
CHI = 0.8  # persistence of the regimes
P1 = 0.5  # p(r_1 = 1)
ORDERS = (16, 24, 36, 48, 64, 96, 128)  # Gauss-Hermite nodes per dimension
TOL = 1e-7
N_STABLE = 2  # successive refinements below TOL
MODELS = ("1", "2", "1p")
PATHS = tuple(itertools.product(range(K), repeat=2))  # (r_2, r_3), 0-based

FAMILIES = {
    "M": dict(
        lam=0.6,
        w=(-0.25, 0.25),
        z=(-0.40, -0.70),
        v=(-0.70, 0.25),
        grid=tuple(round(0.05 * k, 2) for k in range(11)),  # gamma
    ),
    "N": dict(
        lam=0.5,
        w=(0.55, -0.50),
        z=(-0.30, -0.60),
        beta0=(0.80, -0.55),
        grid=tuple(round(0.4 + 0.1 * k, 1) for k in range(7)),  # r
    ),
}

# sensitivity: points, steps and tolerances
ANCHORS = {"M": (0.2, 0.5), "N": (0.4, 1.0)}  # direct perturbations of Ft at the last one
STEP_COEF, STEP_SCALAR, STEP_FT = 0.02, 0.05, 0.01
TOL_SENS = {"M": 1e-5, "N": 1e-6}


# ----------------------------------------------------------------------------- models
def chain(chi=CHI, p1=P1):
    """Stationary chain: p(r_1) = (p1, 1 - p1), P = chi I + (1 - chi) 1 p(r_1)^T.

    For p1 = 1/2, p(r_{n+1} = j | r_n = i) = (1 + chi)/2 if j = i and (1 - chi)/2 otherwise.
    """
    pi = np.array([p1, 1.0 - p1])
    return chi * np.eye(K) + (1.0 - chi) * np.tile(pi, (K, 1)), pi


def normalized_transition(gamma, v, w, z):
    """Ft_j = [[u_j, v_j], [w_j, z_j]], shape (K, 2, 2), with u_j such that a_j - M_j c_j = gamma."""
    v, w, z = (np.asarray(x, dtype=float) for x in (v, w, z))
    u = ((1.0 - w**2 - z**2) * gamma - v * w * z) / (1.0 - z**2)
    return np.stack([np.array([[u[j], v[j]], [w[j], z[j]]]) for j in range(K)])


def make_thgmsm(lam, Ft, chi=CHI, p1=P1):
    """Stationary THGMSM G with Sigma = [[1, lam], [lam, 1]] and normalized transitions Ft_j."""
    Ft = np.asarray(Ft, dtype=float)
    if not abs(lam) < 1:
        raise ValueError("|lam| must be below 1")
    if np.linalg.norm(Ft, 2, axis=(1, 2)).max() >= 1:
        raise ValueError("Sigma_W is not positive definite: the spectral norm of Ft_j must be < 1")
    L = np.array([[np.sqrt(1.0 - lam**2), lam], [0.0, 1.0]])
    Fj = L @ Ft @ np.linalg.inv(L)
    SWj = L @ (np.eye(2) - Ft @ Ft.transpose(0, 2, 1)) @ L.T
    P, pi = chain(chi, p1)
    return make_model(
        a=Fj[:, 0, 0],
        b=Fj[:, 0, 1],
        c=Fj[:, 1, 0],
        d=Fj[:, 1, 1],
        su=SWj[:, 0, 0],
        rho=SWj[:, 0, 1],
        sv=SWj[:, 1, 1],
        sx1=[1.0, 1.0],
        sy1=[1.0, 1.0],
        lam=[lam, lam],
        P=P,
        p1=pi,
    )


def family_Ft(family, t, **over):
    """Ft_j of the family ``M`` at gamma = t, or of the family ``N`` (N_beta) at r = t."""
    f = {**FAMILIES[family], **over}
    w, z = np.asarray(f["w"], dtype=float), np.asarray(f["z"], dtype=float)
    if family == "M":
        return normalized_transition(t, f["v"], w, z)
    return normalized_transition(0.0, t * np.asarray(f["beta0"], dtype=float) * (1.0 - z**2), w, z)


def family_model(family, t, chi=CHI, p1=P1, **over):
    """THGMSM of the family at gamma = t (``M``) or r = t (``N``); ``over`` replaces lam, v, w, ..."""
    lam = over.get("lam", FAMILIES[family]["lam"])
    return make_thgmsm(lam, family_Ft(family, t, **over), chi, p1)


def cgomsm1p(m):
    """CGOMSM1 without autoregression: c_j := 0 and a_j := 0."""
    m1 = cgomsm1(m)
    m1["a"] = np.zeros_like(m["a"])
    return m1


def approximations(m):
    return {"G": m, "1": cgomsm1(m), "2": cgomsm2(m), "1p": cgomsm1p(m)}


def conditional_coefficients(m):
    """(a_j - M_j c_j, b_j - M_j d_j): coefficients of X_n and Y_n in X_{n+1} given Y_{n+1}."""
    M = m["rho"] / m["sv"]
    return m["a"] - M * m["c"], m["b"] - M * m["d"]


# ----------------------------------------------------------------------------- exact estimates
def components(m):
    """Gaussian law of (X_3, Y_{1:3}) given (r_2, r_3) in the model m, for the four paths of PATHS.

    Returns the prior p(r_2, r_3), the Cholesky factor of Cov(Y_{1:3}), its inverse, the constant
    log p(r_2, r_3) - log det(chol), the regression vector m(r_2, r_3) and Var(X_3 | r, Y_{1:3}).
    """
    S1 = SZ1(m, 0)
    if not all(np.allclose(SZ1(m, i), S1, rtol=0, atol=1e-14) for i in range(K)):
        raise ValueError("Sigma_{Z_1,i} must not depend on i")
    p2 = m["p1"] @ m["P"]
    prior, chol, reg, var = [], [], [], []
    iy = [1, 3, 5]
    for j, k in PATHS:
        F2, F3 = F(m, j), F(m, k)
        C = np.zeros((6, 6))  # covariance of (Z_1, Z_2, Z_3), lower blocks
        C[:2, :2] = S1
        C[2:4, :2] = F2 @ S1
        C[2:4, 2:4] = F2 @ S1 @ F2.T + SW(m, j)
        C[4:, :4] = F3 @ C[2:4, :4]
        C[4:, 4:] = F3 @ C[2:4, 2:4] @ F3.T + SW(m, k)
        C = np.tril(C) + np.tril(C, -1).T
        Cy, cxy = C[np.ix_(iy, iy)], C[4, iy]
        coef = np.linalg.solve(Cy, cxy)
        prior.append(p2[j] * m["P"][j, k])
        chol.append(np.linalg.cholesky(Cy))
        reg.append(coef)
        var.append(C[4, 4] - cxy @ coef)
    prior, chol = np.array(prior), np.array(chol)
    return dict(
        prior=prior,
        chol=chol,
        white=np.linalg.inv(chol),
        logc=np.log(prior) - np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1),
        reg=np.array(reg),
        var=np.array(var),
    )


def posterior(comp, y):
    """p(r_2, r_3 | y_{1:3}) and E[X_3 | r_2, r_3, y_{1:3}], both (n, 4), for observations y (n, 3)."""
    white = np.einsum("pij,nj->npi", comp["white"], y)
    logw = comp["logc"] - 0.5 * np.sum(white**2, axis=2)
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    w /= w.sum(axis=1, keepdims=True)
    return w, y @ comp["reg"].T


def estimate(m, y):
    """E_m[X_3 | y_{1:3}] in the model m, regimes hidden; y of shape (3,) or (n, 3)."""
    w, g = posterior(components(m), np.atleast_2d(np.asarray(y, dtype=float)))
    return np.sum(w * g, axis=1)


# ----------------------------------------------------------------------------- quadrature under G
_RAW = ["ER", "ER_direct", "EI0", "R0_direct"] + [
    f"{name}{l}" for l in MODELS for name in ("EI", "reg", "wgt", "int", "R_direct")
]


def integrate(m, order, block=1 << 16):
    """Errors of the estimates under G with ``order`` Gauss-Hermite nodes per dimension."""
    comps = {name: components(v) for name, v in approximations(m).items()}
    G = comps["G"]
    lam = m["lam"][0] / m["sy1"][0]
    x, wq = roots_hermitenorm(order)
    wq = wq / np.sqrt(2.0 * np.pi)
    tail = np.stack(np.meshgrid(x, x, indexing="ij"), axis=-1).reshape(-1, 2)
    wtail = np.outer(wq, wq).ravel()
    step = max(1, block // order**2)
    acc = np.zeros(len(_RAW))
    for t in range(
        len(PATHS)
    ):  # true path: Y_{1:3} ~ N(0, chol_t chol_t^T) with probability prior_t
        for i0 in range(0, order, step):
            head, whead = x[i0 : i0 + step], wq[i0 : i0 + step]
            nodes = np.column_stack([np.repeat(head, order**2), np.tile(tail, (len(head), 1))])
            wt = G["prior"][t] * np.repeat(whead, order**2) * np.tile(wtail, len(head))
            y = nodes @ G["chol"][t].T
            wG, gG = posterior(G, y)
            fG = np.sum(wG * gG, axis=1)
            x_t, var_t = gG[:, t], G["var"][t]  # X_3 | r, y ~ N(x_t, var_t) under G
            f0 = lam * y[:, 2]
            cols = [
                np.sum(wG * (G["var"] + (gG - fG[:, None]) ** 2), axis=1),
                var_t + (x_t - fG) ** 2,
                (f0 - fG) ** 2,
                var_t + (x_t - f0) ** 2,
            ]
            for l in MODELS:
                wl, gl = posterior(comps[l], y)
                fl = np.sum(wl * gl, axis=1)
                e_reg = np.sum(wG * gl, axis=1) - fG
                e_w = fl - fG - e_reg
                cols += [(fl - fG) ** 2, e_reg**2, e_w**2, 2 * e_reg * e_w, var_t + (x_t - fl) ** 2]
            acc += wt @ np.column_stack(cols)
    raw = dict(zip(_RAW, acc))
    ER = raw["ER"]
    res = dict(ER=ER, R0=m["sx1"][0] - lam**2 * m["sy1"][0], tau0=raw["EI0"] / ER)
    check = [ER - raw["ER_direct"], ER + raw["EI0"] - raw["R0_direct"], ER + raw["EI0"] - res["R0"]]
    for l in MODELS:
        res[f"tau{l}"] = raw[f"EI{l}"] / ER
        res[f"R{l}"] = ER + raw[f"EI{l}"]
        for name in ("reg", "wgt", "int"):
            res[f"{name}{l}"] = raw[f"{name}{l}"] / ER
        check += [
            res[f"R{l}"] - raw[f"R_direct{l}"],
            raw[f"EI{l}"] - sum(raw[f"{name}{l}"] for name in ("reg", "wgt", "int")),
        ]
    res["delta"] = res["tau1"] - res["tau2"]
    res = {k: float(v) for k, v in res.items()}
    res["check"] = float(np.max(np.abs(check)))
    return res


def converge(m, orders=ORDERS, tol=TOL, n_stable=N_STABLE):
    """Refine until every value changes by less than ``tol`` on ``n_stable`` successive refinements."""
    prev, streak, change = None, 0, None
    for order in orders:
        cur = integrate(m, order)
        if prev is not None:
            change = max(abs(cur[k] - prev[k]) for k in cur if k != "check")
            streak = streak + 1 if change < tol else 0
        prev = cur
        if streak >= n_stable and cur["check"] < tol:
            break
    converged = streak >= n_stable and cur["check"] < tol
    return dict(**cur, order=order, change=change, converged=bool(converged))


# ----------------------------------------------------------------------------- case e
def _describe(m):
    ga, be = conditional_coefficients(m)
    return dict(
        a_minus_Mc=ga.tolist(),
        b_minus_Md=be.tolist(),
        c=m["c"].tolist(),
        M=(m["rho"] / m["sv"]).tolist(),
    )


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))


def exp_e(out: Path, orders=ORDERS, tol=TOL):
    rows = []
    for family, f in FAMILIES.items():
        for t in f["grid"]:
            m = family_model(family, t)
            norm = float(np.linalg.norm(family_Ft(family, t), 2, axis=(1, 2)).max())
            rows.append(
                dict(family=family, t=t, norm_Ft=norm, **_describe(m), **converge(m, orders, tol))
            )
    res = dict(
        chi=CHI,
        p1=P1,
        orders=list(orders),
        tol=tol,
        n_stable=N_STABLE,
        families={
            k: {a: list(b) if isinstance(b, tuple) else b for a, b in f.items()}
            for k, f in FAMILIES.items()
        },
        rows=rows,
    )
    _write(out / "exp_e.json", res)
    _plot_e(res, out / "fig_e_stationary.pdf")
    return res


def _ends(res):
    """Rows at the two ends of each family, in the order M, N."""
    return [
        rows[i]
        for fam in FAMILIES
        for rows in [[r for r in res["rows"] if r["family"] == fam]]
        for i in (0, -1)
    ]


# ----------------------------------------------------------------------------- sensitivity
def sensitivity_plan():
    """One-at-a-time perturbations: 18 per point, and 16 direct ones at the last point of each family."""
    plan = []
    for family, f in FAMILIES.items():
        coefs = [("v" if family == "M" else "beta0"), "w", "z"]
        for t in ANCHORS[family]:
            for name in coefs:
                for j in range(K):
                    for sign in (-1, 1):
                        val = list(f[name])
                        val[j] += sign * STEP_COEF
                        label = f"{name}_{j + 1} {sign * STEP_COEF:+.2f}"
                        plan.append(dict(family=family, t=t, label=label, over={name: val}))
            for name, base in (("lam", f["lam"]), ("p1", P1), ("chi", CHI)):
                for sign in (-1, 1):
                    label = f"{name} {sign * STEP_SCALAR:+.2f}"
                    plan.append(
                        dict(
                            family=family, t=t, label=label, over={name: base + sign * STEP_SCALAR}
                        )
                    )
        t = ANCHORS[family][-1]
        for j, row, col, sign in itertools.product(range(K), range(2), range(2), (-1, 1)):
            label = f"Ft_{j + 1}[{row + 1},{col + 1}] {sign * STEP_FT:+.2f}"
            plan.append(dict(family=family, t=t, label=label, direct=(j, row, col, sign * STEP_FT)))
    return plan


def _perturbed_model(case):
    family, t = case["family"], case["t"]
    if "direct" in case:
        j, row, col, eps = case["direct"]
        Ft = family_Ft(family, t)
        Ft[j, row, col] += eps
        return make_thgmsm(FAMILIES[family]["lam"], Ft)
    over = dict(case["over"])
    return family_model(family, t, chi=over.pop("chi", CHI), p1=over.pop("p1", P1), **over)


_RANKINGS = (("R1", "R2"), ("R1", "R1p"), ("R1p", "R2"), ("R2", "R0"))


def exp_e_sensitivity(out: Path, orders=ORDERS):
    rows, summary = [], []
    for family in FAMILIES:
        tol = TOL_SENS[family]
        base = {t: converge(family_model(family, t), orders, tol) for t in ANCHORS[family]}
        cases = [c for c in sensitivity_plan() if c["family"] == family]
        for c in cases:
            rows.append(
                dict(
                    family=family,
                    t=c["t"],
                    label=c["label"],
                    **converge(_perturbed_model(c), orders, tol),
                )
            )
        for t in ANCHORS[family]:
            sub = [r for r in rows if r["family"] == family and r["t"] == t]
            s = dict(
                family=family,
                t=t,
                n=len(sub),
                base={k: base[t][k] for k in ("ER", "R0", "R1", "R2", "R1p", "delta")},
            )
            for key in ("delta", "R1", "R2", "R1p"):
                s[f"{key}_range"] = [min(r[key] for r in sub), max(r[key] for r in sub)]
            s["reversed"] = {
                f"{u}<{v}": [r["label"] for r in sub if (r[u] < r[v]) != (base[t][u] < base[t][v])]
                for u, v in _RANKINGS
            }
            s["not_converged"] = [r["label"] for r in sub if not r["converged"]]
            summary.append(s)
    res = dict(
        steps=dict(coef=STEP_COEF, scalar=STEP_SCALAR, Ft=STEP_FT),
        tol=TOL_SENS,
        summary=summary,
        rows=rows,
    )
    _write(out / "exp_e_sensitivity.json", res)
    return res


# ----------------------------------------------------------------------------- figure
_CURVES = (  # key, label, color, line style, marker, marker face
    ("R1", "CGOMSM1", "C0", "-", "o", "C0"),
    ("R1p", "CGOMSM1 without autoregression", "C0", "--", "o", "white"),
    ("R2", "CGOMSM2", "C3", "-", "s", "C3"),
    ("R0", r"benchmark $\lambda y_3$", "0.45", ":", None, None),
    ("ER", "optimal (THGMSM)", "k", "-", None, None),
)
_TERMS = (  # key, label, face color, hatch
    ("reg", "regression", "0.25", None),
    ("wgt", "weight", "0.65", None),
    ("int", "interaction", "white", "////"),
)


def _plot_e(res, path):
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))
    panels = (
        ("M", r"(a) family $\mathcal{M}$", r"conditional autoregression $\gamma$"),
        ("N", r"(b) family $\mathcal{N}_{\beta}$", r"scale factor $r$"),
    )
    for ax, (family, title, xlabel) in zip(axes, panels):
        rows = [r for r in res["rows"] if r["family"] == family]
        t = [r["t"] for r in rows]
        for key, label, color, ls, marker, mfc in _CURVES:
            ax.plot(
                t,
                [r[key] for r in rows],
                ls,
                color=color,
                marker=marker,
                ms=3,
                mfc=mfc,
                lw=0.8 if key == "ER" else 1.0,
                label=label,
            )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean square error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.03, 0.0),
        ncol=3,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.2,
    )

    ax = axes[2]
    ends = _ends(res)
    pos = np.arange(len(ends))
    width = 0.2  # four slots per group: the three terms, then their sum
    for i, (key, label, face, hatch) in enumerate(_TERMS):
        ax.bar(
            pos + (i - 1.5) * width,
            [100 * (r[f"{key}1"] - r[f"{key}2"]) for r in ends],
            width * 0.9,
            color=face,
            edgecolor="k",
            lw=0.5,
            hatch=hatch,
            label=label,
            zorder=2,
        )
    total = [100 * r["delta"] for r in ends]
    ax.plot(
        pos + 1.5 * width,
        total,
        "D",
        color="C3",
        mec="k",
        mew=0.5,
        ms=4,
        label=r"sum $100\,\Delta$",
        zorder=3,
    )
    for p, v in zip(pos + 1.5 * width, total):
        ax.annotate(
            f"{v:+.2f}".replace("-", "\N{MINUS SIGN}"),
            (p, v),
            xytext=(0, 4 if v > 0 else -4),
            textcoords="offset points",
            ha="center",
            va="bottom" if v > 0 else "top",
            fontsize=7,
        )
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlim(-0.55, len(ends) - 0.35)
    names = {"M": (r"$\mathcal{M}$", r"$\gamma$"), "N": (r"$\mathcal{N}_{\beta}$", r"$r$")}
    ax.set_xticks(
        pos, [f"{names[r['family']][0]}\n{names[r['family']][1]}$\\,=\\,{r['t']:g}$" for r in ends]
    )
    ax.set_title(r"(c) contributions to $100\,\Delta$")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    h, lab = ax.get_legend_handles_labels()
    order = [1, 2, 3, 0]  # regression, weight, interaction, sum
    fig.legend(
        [h[i] for i in order],
        [lab[i] for i in order],
        loc="lower right",
        bbox_to_anchor=(0.995, 0.0),
        ncol=2,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.2)
    _savefig(fig, path)
    plt.close(fig)


def replot(out: Path):
    if (out / "exp_e.json").exists():
        _plot_e(json.loads((out / "exp_e.json").read_text()), out / "fig_e_stationary.pdf")


# ----------------------------------------------------------------------------- CLI
def _print_e(res):
    print("family     t     ER      R1      R1p     R2      R0    100*Delta  order  check")
    for r in res["rows"]:
        print(
            f"{r['family']:6s} {r['t']:5.2f}  {r['ER']:.4f}  {r['R1']:.4f}  {r['R1p']:.4f}  {r['R2']:.4f}  "
            f"{r['R0']:.4f}  {100 * r['delta']:+8.3f}  {r['order']:5d}  {r['check']:.0e}"
            + ("" if r["converged"] else "  NOT CONVERGED")
        )
    print("ends: contributions to 100*Delta (regression, weight, interaction)")
    for r in _ends(res):
        terms = "  ".join(f"{100 * (r[k + '1'] - r[k + '2']):+7.2f}" for k in ("reg", "wgt", "int"))
        print(f"{r['family']:6s} {r['t']:5.2f}  {terms}")


def _print_sensitivity(res):
    for s in res["summary"]:
        lo, hi = s["delta_range"]
        rev = {k: len(v) for k, v in s["reversed"].items() if v}
        print(
            f"{s['family']:2s} t={s['t']:.1f}: {s['n']} perturbations, 100*Delta in [{100 * lo:+.2f}, {100 * hi:+.2f}]"
            f" (unperturbed {100 * s['base']['delta']:+.2f}); reversed rankings: {rev or 'none'}"
            + (f"; not converged: {s['not_converged']}" if s["not_converged"] else "")
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("which", nargs="?", default="e", choices=("e", "sensitivity", "all", "replot"))
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    if a.which == "replot":
        replot(a.out)
        return
    if a.which in ("e", "all"):
        _print_e(exp_e(a.out))
    if a.which in ("sensitivity", "all"):
        _print_sensitivity(exp_e_sensitivity(a.out))


if __name__ == "__main__":
    main()
