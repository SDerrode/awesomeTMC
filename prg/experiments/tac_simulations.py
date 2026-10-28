#!/usr/bin/env python3
"""Reproduction of Section V of the IEEE TAC paper (CGOMSM1 vs CGOMSM2).

Paper: S. Derrode, C. Fernandes, U. Hubert-Almuzara, F. Lehmann, W. Pieczynski, "On a New Family of
Models for Exact Fast Filtering in Gaussian Markov Switching Systems". CGOMSM1 = THGMSM with C = 0;
CGOMSM2 = THGMSM with A = Delta Sigma_V^{-1} C and B = Delta Sigma_V^{-1} D.

The filters used here are self-contained numpy implementations for the scalar case q = s = 1; they do
not call the library's ``GSSFilter``. They are: the exact filter of the CGOMSM1 (K conditional means
propagated and mixed over r_n), the exact filter of the CGOMSM2 (Propositions 3-4 of the paper, with
the first transition for an arbitrary initial law), the exact Bayes filter by enumeration of the K^N
regime paths, a Rao-Blackwellized particle filter (RBPF, optimal proposal), and their versions with a
known input (Section V-C). ``tests/test_tac_simulations.py`` checks them against the enumeration
(agreement to about 1e-12).

Model (dynamic form, regime j = R_{n+1}):
    [X_{n+1}; Y_{n+1}] = F_j [X_n; Y_n] + W_{n+1},  F_j = [[a_j, b_j], [c_j, d_j]],
    W ~ N(0, [[su_j, rho_j], [rho_j, sv_j]]),   Z_1 | r_1 = i ~ N(0, [[sx1_i, lam_i], [lam_i, sy1_i]]).
CGOMSM1 of a THGMSM: same parameters with c_j := 0.
CGOMSM2 of a THGMSM: same parameters with a_j := rho_j/sv_j c_j, b_j := rho_j/sv_j d_j.

Case ``a`` -- Section V-A (Table I -> table_a.tex, lambda figure -> fig_a_lambda.pdf):
    regime-known estimators at n = 2, closed forms (no simulation); tau_k = EI_k / ER, checked by a
    Monte Carlo simulation of the dynamic form.
Case ``b`` -- Section V-B (Table II -> table_b.tex, path figure -> fig_b_homotopy.pdf):
    path F^eps = (1 - eps) F^1 + eps F^2 from the CGOMSM1 (eps = 0) to the CGOMSM2 (eps = 1). Data
    simulated with THGMSM(eps); the exact filters of the CGOMSM1 and of the CGOMSM2 (frozen parameters)
    are compared with two references:
      * exact Bayes filter by enumeration of the 2^n_short regime paths, on short sequences;
        excess = sum_n (f - e)^2 / sum_n Var[X_n | y_{1:n}] (orthogonality: same expectation ratio as
        MSE_f / MSE_opt - 1);
      * RBPF (optimal proposal) on long sequences; excess = MSE_f / MSE_rbpf - 1.
    eps* = crossing point of the two excess curves. 95% Monte Carlo half-widths by the delta method
    (ratio of means), bootstrap over sequences for eps* and for the maxima over eps. The per-sequence
    errors are stored in exp_b_seq.npz.
Case ``c`` -- Section V-C (Table III -> table_c.tex, trace -> fig_c_vehicle.pdf): real driving data
    (Figshare 10.6084/m9.figshare.28078274.v1, aggressive driver, 100 Hz, recordings longer than 5 s,
    subsampled to 10 Hz). X = yaw rate, Y = lateral acceleration, u_n = (steering angle at n + 1, 1),
    regime = speed band (labelled for training only, hidden in the filters). The variables are divided
    by their standard deviations on the training recordings. THGMSM, CGOMSM1 and CGOMSM2 transition
    densities are fitted per regime by maximum likelihood; each fit reduces to two least-squares
    regressions:
      THGMSM : (x', y') on (x, y, u);
      CGOMSM1: y' on (y, u), then x' on (x, y, y', u);
      CGOMSM2: y' on (x, y, u), then x' on (y', u).
    Held-out files: yaw RMSE and log-likelihood of Y of the exact CGOMSM1 / CGOMSM2 filters, of an RBPF
    for the THGMSM, and of the Kalman filter of a single-regime THGMSM; with and without the steering
    angle in u (the constant being kept).
Case ``transient`` (also run by ``b``): at the two ends of the path, where the optimal filter is exact,
    excess of the other family over the first n_short steps and over the following ones (exp_b.json).
Case ``timing``: computing time per time step of the filters of case b (timing.json).
Case ``d`` (Section V-C of the current version, ``exp_d``): river discharge with a measured tributary
    inflow as known input (USGS NWIS, three reaches); see ``prg/experiments/tac_rivers.py``. Case ``c``
    (vehicle data) is the real-data study of an earlier version of the paper.
Case ``replot``: regenerates the tables and figures from the stored JSON files, without simulating.

Run:  python -m prg.experiments.tac_simulations [a|b|c|d|all|transient|timing|replot] [--out DIR]
                                                [--data DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np

OUT = Path("results/tac_simulations")


# ----------------------------------------------------------------------------- figures
def _mpl():
    """Matplotlib with the paper style: 8 pt serif text, STIX fonts embedded as TrueType (no Type 3)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "font.family": "serif",
            # STIX is a Times-like font shipped with matplotlib: identical output on every machine.
            "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    return plt


def _savefig(fig, path):
    fig.savefig(path, metadata={"CreationDate": None})  # reproducible PDF bytes


# ----------------------------------------------------------------------------- models
def make_model(a, b, c, d, su, rho, sv, sx1, sy1, lam, P, p1):
    m = {
        k: np.asarray(v, dtype=float)
        for k, v in dict(
            a=a, b=b, c=c, d=d, su=su, rho=rho, sv=sv, sx1=sx1, sy1=sy1, lam=lam
        ).items()
    }
    m["P"] = np.asarray(P, dtype=float)
    m["p1"] = np.asarray(p1, dtype=float)
    assert np.all(m["rho"] ** 2 < m["su"] * m["sv"]), "Sigma_W not positive definite"
    assert np.all(m["lam"] ** 2 < m["sx1"] * m["sy1"]), "Sigma_Z1 not positive definite"
    return m


def cgomsm1(m):
    m1 = dict(m)
    m1["c"] = np.zeros_like(m["c"])
    return m1


def cgomsm2(m):
    m2 = dict(m)
    g = m["rho"] / m["sv"]
    m2["a"], m2["b"] = g * m["c"], g * m["d"]
    return m2


def path(m, eps):
    m1, m2 = cgomsm1(m), cgomsm2(m)
    me = dict(m)
    for key in ("a", "b", "c", "d"):
        me[key] = (1 - eps) * m1[key] + eps * m2[key]
    return me


def F(m, j):
    return np.array([[m["a"][j], m["b"][j]], [m["c"][j], m["d"][j]]])


def SW(m, j):
    return np.array([[m["su"][j], m["rho"][j]], [m["rho"][j], m["sv"][j]]])


def SZ1(m, i):
    return np.array([[m["sx1"][i], m["lam"][i]], [m["lam"][i], m["sy1"][i]]])


def spectral_radius(m):
    return max(max(abs(np.linalg.eigvals(F(m, j)))) for j in range(2))


def stationary_P(p12):
    """Symmetric stationary chain with p(r1=1)=0.5 and p(r1=1, r2=2)=p12 (jump probability 2 p12)."""
    s = 2 * p12
    return np.array([[1 - s, s], [s, 1 - s]]), np.array([0.5, 0.5])


# ----------------------------------------------------------------------------- V-A (closed form)
def _cov12(m, i, j):
    """Cov of (X1, Y1, X2, Y2) given r1 = i, r2 = j."""
    S1, Fj = SZ1(m, i), F(m, j)
    return np.block([[S1, S1 @ Fj.T], [Fj @ S1, Fj @ S1 @ Fj.T + SW(m, j)]])


def coeffs(m, j, i=0):
    """[alpha_j, beta_j]: E[X2 | Y1, Y2, r2 = j] (Sigma_Z1 common to both regimes, so i is irrelevant)."""
    S = _cov12(m, i, j)
    iy = [1, 3]
    return np.linalg.solve(S[np.ix_(iy, iy)], S[iy, 2])


def exp_a_closed(m):
    """ER, EI1, EI2, tau1, tau2 for the regime-known estimators at n = 2."""
    assert all(np.allclose(m[k], m[k][0]) for k in ("sx1", "sy1", "lam")), "Sigma_Z1 must be common"
    pj = m["p1"] @ m["P"]
    m1, m2 = cgomsm1(m), cgomsm2(m)
    ER = EI1 = EI2 = 0.0
    for j in range(2):
        S = _cov12(m, 0, j)
        Syy = S[np.ix_([1, 3], [1, 3])]
        sxy = S[[1, 3], 2]
        c0, c1, c2 = coeffs(m, j), coeffs(m1, j), coeffs(m2, j)
        ER += pj[j] * (S[2, 2] - c0 @ sxy)
        EI1 += pj[j] * (c0 - c1) @ Syy @ (c0 - c1)
        EI2 += pj[j] * (c0 - c2) @ Syy @ (c0 - c2)
    return dict(
        ER=float(ER), EI1=float(EI1), EI2=float(EI2), tau1=float(EI1 / ER), tau2=float(EI2 / ER)
    )


def exp_a_mc(m, n=2_000_000, seed=0):
    """Monte Carlo check of exp_a_closed, simulated with the dynamic form (independent of _cov12).

    Returns the estimated ER, tau1, tau2, their standard errors (delta method for the ratios), and the
    z-scores of tau1, tau2 against the closed forms.
    """
    rng = np.random.default_rng(seed)
    r1 = rng.choice(2, size=n, p=m["p1"])
    r2 = (rng.random(n) < m["P"][r1, 1]).astype(int)
    Z1 = np.empty((n, 2))
    Z2 = np.empty((n, 2))
    for i in range(2):
        idx = r1 == i
        Z1[idx] = rng.standard_normal((idx.sum(), 2)) @ np.linalg.cholesky(SZ1(m, i)).T
    for j in range(2):
        idx = r2 == j
        W = rng.standard_normal((idx.sum(), 2)) @ np.linalg.cholesky(SW(m, j)).T
        Z2[idx] = Z1[idx] @ F(m, j).T + W
    Y = np.c_[Z1[:, 1], Z2[:, 1]]
    sq = []
    for mm in (m, cgomsm1(m), cgomsm2(m)):
        coef = np.array([coeffs(mm, j) for j in range(2)])
        sq.append((Z2[:, 0] - np.sum(Y * coef[r2], axis=1)) ** 2)
    ER = sq[0].mean()
    closed = exp_a_closed(m)
    res = dict(ER=float(ER), ER_se=float(sq[0].std(ddof=1) / np.sqrt(n)), n=n)
    for k in (1, 2):
        D = sq[k] - sq[0]
        tau = D.mean() / ER
        se = (D - tau * sq[0]).std(ddof=1) / np.sqrt(n) / ER
        res[f"tau{k}"] = float(tau)
        res[f"tau{k}_se"] = float(se)
        res[f"tau{k}_z"] = float((tau - closed[f"tau{k}"]) / se)
    return res


# V-A design: coupling x gap to CGOMSM2 x noise correlation (2 x 2 x 2)
BASE_A = dict(
    d=[-0.5, -0.3], su=[1.0, 0.8], sv=[0.5, 1.0], sx1=[1.0, 1.0], sy1=[1.0, 1.0], lam=[0.5, 0.5]
)
# c = gamma (1, -0.8); gaps a_j - rho_j/sv_j c_j = +-g, b_j - rho_j/sv_j d_j = -+g (upper sign j = 1);
# rho_j = +-kappa sqrt(su_j sv_j) (kappa = correlation coefficient of the noises); experiments 1-8
# enumerate (gamma, g, kappa) in this order, kappa varying fastest.
LEVELS_A = dict(c=(0.2, 1.0), g=(0.05, 0.5), rf=(0.2, 0.7))


def design_a(c_lvl, g_lvl, rf, lam=0.5, p12=0.1):
    """THGMSM of Section V-A for coupling gamma = c_lvl, gap g = g_lvl and noise correlation kappa = rf."""
    su, sv, d = np.array(BASE_A["su"]), np.array(BASE_A["sv"]), np.array(BASE_A["d"])
    rho = rf * np.sqrt(su * sv) * np.array([1.0, -1.0])
    c = c_lvl * np.array([1.0, -0.8])
    a = rho / sv * c + g_lvl * np.array([1.0, -1.0])
    b = rho / sv * d + g_lvl * np.array([-1.0, 1.0])
    P, p1 = stationary_P(p12)
    return make_model(a, b, c, d, su, rho, sv, BASE_A["sx1"], BASE_A["sy1"], [lam, lam], P, p1)


def exp_a(out: Path):
    rows = []
    for k, (c_lvl, g_lvl, rf) in enumerate(itertools.product(*LEVELS_A.values()), start=1):
        m = design_a(c_lvl, g_lvl, rf)
        res = exp_a_closed(m)
        mc = exp_a_mc(m, seed=k)
        rows.append(
            dict(
                exp=k,
                c_level=c_lvl,
                gap=g_lvl,
                rho_factor=rf,
                a=m["a"].tolist(),
                b=m["b"].tolist(),
                c=m["c"].tolist(),
                d=m["d"].tolist(),
                rho=m["rho"].tolist(),
                **res,
                mc=mc,
            )
        )
    lam_grid = np.linspace(0.0, 0.95, 39)
    curves = {}
    for k in (2, 7):  # two contrasted configurations
        c_lvl, g_lvl, rf = list(itertools.product(*LEVELS_A.values()))[k - 1]
        curves[k] = [exp_a_closed(design_a(c_lvl, g_lvl, rf, lam=l)) for l in lam_grid]
    out.mkdir(parents=True, exist_ok=True)
    (out / "exp_a.json").write_text(
        json.dumps(
            dict(
                design=dict(BASE_A, levels=LEVELS_A, mc_samples=rows[0]["mc"]["n"]),
                rows=rows,
                lam=lam_grid.tolist(),
                curves={str(k): v for k, v in curves.items()},
            ),
            indent=1,
        )
    )
    _table_a(rows, out / "table_a.tex")
    _plot_a(lam_grid, curves, out / "fig_a_lambda.pdf")
    return rows


def _num(v, fmt):
    """Number in math mode (true minus sign), without a negative zero."""
    s = format(v, fmt)
    if float(s) == 0.0:
        s = s.lstrip("-")
    return f"${s}$"


def _table_a(rows, path):
    lines = [
        # 4-decimal tau, true minus signs and the "exp." corner: 250 pt, fits \columnwidth (252 pt)
        r"\setlength{\tabcolsep}{2.8pt}%",
        r"\begin{tabular}{@{}l" + "c" * len(rows) + r"@{}}",
        r"\toprule",
        "exp. & " + " & ".join(str(r["exp"]) for r in rows) + r" \\",  # same corner as Table II
        r"\midrule",
    ]
    for key, lab in (("a", "a"), ("b", "b"), ("c", "c"), ("d", "d"), ("rho", r"\rho")):
        for j in range(2):
            lines.append(
                f"${lab}_{{{j + 1}}}$ & "
                + " & ".join(_num(r[key][j], ".2f") for r in rows)
                + r" \\"
            )
    lines.append(r"\midrule")
    for k in (1, 2):  # 4 decimals: the rankings of Table I can be read even for experiment 1
        lines.append(
            rf"$\tau_{{{k}}}$ & " + " & ".join(_num(r[f"tau{k}"], ".4f") for r in rows) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def _plot_a(lam, curves, path):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    styles = {2: "-", 7: "--"}
    for k, cur in curves.items():
        k = int(k)
        for t, col, mk in (("tau1", "C0", "o"), ("tau2", "C3", "s")):
            ax.plot(
                lam,
                [c[t] for c in cur],
                styles[k],
                color=col,
                marker=mk,
                ms=3.5,
                markevery=(k % 2, 4),
                mfc="white" if k == 7 else col,
                lw=1.0,
                label=rf"$\tau_{t[-1]}$ (exp. {k})",
            )
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("relative error increase")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(ncol=2, frameon=False, handlelength=2.6, columnspacing=1.0)
    fig.tight_layout()
    _savefig(fig, path)
    plt.close(fig)


# ----------------------------------------------------------------------------- simulation and filters (V-B)
def simulate(m, N, rng):
    # multivariate_normal (SVD factorization) is kept so that the published numbers are reproduced;
    # the factorization comes from LAPACK, so the last digits may differ on another platform.
    r = np.empty(N, dtype=int)
    Z = np.empty((N, 2))
    r[0] = rng.choice(2, p=m["p1"])
    Z[0] = rng.multivariate_normal(np.zeros(2), SZ1(m, r[0]))
    for n in range(1, N):
        r[n] = rng.choice(2, p=m["P"][r[n - 1]])
        Z[n] = F(m, r[n]) @ Z[n - 1] + rng.multivariate_normal(np.zeros(2), SW(m, r[n]))
    return r, Z[:, 0], Z[:, 1]


def _lognorm(y, mu, var):
    return -0.5 * (np.log(2 * np.pi * var) + (y - mu) ** 2 / var)


def _kalman_bank(m, y, mprev, Pprev, j, yprev):
    """One pairwise-Kalman step for arrays of hypotheses arriving in regime j (vectorized)."""
    a, b, c, d = m["a"][j], m["b"][j], m["c"][j], m["d"][j]
    mx = a * mprev + b * yprev
    my = c * mprev + d * yprev
    sxx = a * a * Pprev + m["su"][j]
    sxy = a * c * Pprev + m["rho"][j]
    syy = c * c * Pprev + m["sv"][j]
    ll = _lognorm(y, my, syy)
    mnew = mx + sxy / syy * (y - my)
    Pnew = sxx - sxy**2 / syy
    return ll, mnew, Pnew


def _init_post(m, y1):
    """Per-regime log p(r1, y1), E[X1 | r1, y1], Var[X1 | r1, y1]."""
    ll = np.log(m["p1"]) + _lognorm(y1, 0.0, m["sy1"])
    mean = m["lam"] / m["sy1"] * y1
    var = m["sx1"] - m["lam"] ** 2 / m["sy1"]
    return ll, mean, var


def exact_filter(m, y):
    """Exact Bayes filter by enumeration of all regime paths (vectorized). Returns E[X_n|y], Var[X_n|y]."""
    N = len(y)
    ll0, m0, v0 = _init_post(m, y[0])
    logw, mu, P, last = ll0.copy(), m0.copy(), v0.copy(), np.array([0, 1])
    E, V = np.empty(N), np.empty(N)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    E[0] = w @ mu
    V[0] = w @ (P + mu**2) - E[0] ** 2
    for n in range(1, N):
        nl, nm, nP, nr = [], [], [], []
        for j in range(2):
            ll, mn, Pn = _kalman_bank(m, y[n], mu, P, j, y[n - 1])
            nl.append(logw + np.log(m["P"][last, j]) + ll)
            nm.append(mn)
            nP.append(Pn)
            nr.append(np.full(len(mu), j))
        logw, mu, P, last = map(np.concatenate, (nl, nm, nP, nr))
        w = np.exp(logw - logw.max())
        w /= w.sum()
        E[n] = w @ mu
        V[n] = w @ (P + mu**2) - E[n] ** 2
    return E, V


def rbpf(m, y, n_part, rng):
    """Rao-Blackwellized PF with the optimal (K = 2) proposal. Returns E[X_n | y]."""
    N = len(y)
    ll0, m0, v0 = _init_post(m, y[0])
    p0 = np.exp(ll0 - ll0.max())
    p0 /= p0.sum()
    r = rng.choice(2, size=n_part, p=p0)
    mu, P = m0[r], v0[r]
    w = np.full(n_part, 1.0 / n_part)
    E = np.empty(N)
    E[0] = w @ mu
    for n in range(1, N):
        cand = [_kalman_bank(m, y[n], mu, P, j, y[n - 1]) for j in range(2)]
        lj = np.stack([np.log(m["P"][r, j]) + cand[j][0] for j in range(2)])  # (2, n_part)
        lmax = lj.max(axis=0)
        pj = np.exp(lj - lmax)
        tot = pj.sum(axis=0)
        logw = np.log(w) + lmax + np.log(tot)
        choose = (rng.random(n_part) < pj[1] / tot).astype(int)
        mu = np.where(choose == 1, cand[1][1], cand[0][1])
        P = np.where(choose == 1, cand[1][2], cand[0][2])
        r = choose
        w = np.exp(logw - logw.max())
        w /= w.sum()
        E[n] = w @ mu
        if 1.0 / np.sum(w**2) < n_part / 2:  # systematic resampling
            u = (rng.random() + np.arange(n_part)) / n_part
            idx = np.minimum(np.searchsorted(np.cumsum(w), u), n_part - 1)
            mu, P, r = mu[idx], P[idx], r[idx]
            w = np.full(n_part, 1.0 / n_part)
    return E


def cgomsm1_filter(m, y, return_var=False):
    """Exact filter of a CGOMSM1 (c = 0): (R, Y) Markov, E[X_n | r_n, y_{1:n}] by an exact linear recursion.

    With ``return_var``, also returns Var[X_n | y_{1:n}], from the conditional second moments
    E[X_n^2 | r_n, y_{1:n}], which obey the same kind of recursion.
    """
    assert np.allclose(m["c"], 0.0)
    N = len(y)
    g = m["rho"] / m["sv"]
    gam = m["su"] - m["rho"] * g  # Var[X_{n+1} | r_{n+1}, X_n, y_n, y_{n+1}]
    ll0, mu, v0 = _init_post(m, y[0])
    S = v0 + mu**2  # E[X_1^2 | r_1, y_1]
    pi = np.exp(ll0 - ll0.max())
    pi /= pi.sum()
    E, V = np.empty(N), np.empty(N)
    E[0] = pi @ mu
    V[0] = pi @ S - E[0] ** 2
    for n in range(1, N):
        # transition of (R, Y): y_{n+1} | r_{n+1}=k, y_n ~ N(d_k y_n, sv_k)
        lk = _lognorm(y[n], m["d"] * y[n - 1], m["sv"])
        joint = pi[:, None] * m["P"] * np.exp(lk - lk.max())[None, :]  # (j, k)
        col = joint.sum(axis=0)
        pi_new = col / col.sum()
        cond = joint / np.maximum(col, 1e-300)[None, :]  # p(r_n = j | r_{n+1} = k, y_{1:n+1})
        # X_{n+1} | r_n, r_{n+1}, y = a_k X_n + h_k + noise of variance gam_k, X_n ~ p(x_n | r_n, y_{1:n})
        h = m["b"] * y[n - 1] + g * (y[n] - m["d"] * y[n - 1])
        mu_jk = m["a"][None, :] * mu[:, None] + h[None, :]
        S_jk = (
            m["a"][None, :] ** 2 * S[:, None]
            + 2 * m["a"][None, :] * mu[:, None] * h[None, :]
            + (h**2 + gam)[None, :]
        )
        mu = (cond * mu_jk).sum(axis=0)
        S = (cond * S_jk).sum(axis=0)
        pi = pi_new
        E[n] = pi @ mu
        V[n] = pi @ S - E[n] ** 2
    return (E, V) if return_var else E


def cgomsm2_filter(m, y, return_var=False):
    """Exact filter of a CGOMSM2 (Props. 3-4 of the paper), with the first transition for a general initial law.

    With ``return_var``, also returns Var[X_n | y_{1:n}] (Remark 4 of the paper for n >= 2).
    """
    g = m["rho"] / m["sv"]
    assert np.allclose(m["a"], g * m["c"]) and np.allclose(m["b"], g * m["d"])
    M, Gam = g, m["su"] - m["rho"] ** 2 / m["sv"]
    N = len(y)
    ll0, mu1, v1 = _init_post(m, y[0])
    pi = np.exp(ll0 - ll0.max())
    pi /= pi.sum()
    E, V = np.empty(N), np.empty(N)
    E[0] = pi @ mu1
    V[0] = pi @ (v1 + mu1**2) - E[0] ** 2
    for n in range(1, N):
        if n == 1:  # first transition: moments of X_1 | r_1, y_1 from the initial law
            mx, vx = mu1, v1
        else:  # slaved law p(x_n | r_n, y_{1:n}) = N(M y_n, Gamma) (eq:slaved-law of the paper)
            mx, vx = M * y[n - 1], Gam
        mean = m["c"][None, :] * mx[:, None] + m["d"][None, :] * y[n - 1]  # (j, k)
        var = m["sv"][None, :] + m["c"][None, :] ** 2 * vx[:, None]
        lj = np.log(pi[:, None] * m["P"] + 1e-300) + _lognorm(y[n], mean, var)
        a = np.exp(lj - lj.max()).sum(axis=0)
        pi = a / a.sum()
        E[n] = pi @ (M * y[n])
        V[n] = pi @ (Gam + (M * y[n]) ** 2) - E[n] ** 2
    return (E, V) if return_var else E


# V-B design (TAC paper, Sec. V-B, Table II): common Sigma_Z1, Sigma_W, c, d; four experiments
# (p12 x {a, b})
BASE_B = dict(
    c=[0.8, -0.6],
    d=[0.2, 0.3],
    su=[1.0, 0.8],
    sv=[0.5, 0.7],
    rho=[0.4, -0.3],
    sx1=[1.0, 1.0],
    sy1=[1.0, 1.0],
    lam=[0.5, -0.3],
)
AB_SETS = {"A": dict(a=[0.7, 0.4], b=[0.2, -0.1]), "B": dict(a=[-0.5, 0.8], b=[0.4, 0.3])}
EXPS_B = [(1, 0.05, "A"), (2, 0.15, "A"), (3, 0.05, "B"), (4, 0.15, "B")]
N_BOOT = 2000  # bootstrap resamples (over sequences) for eps* and for the maxima over eps
BOOT_SEED = 2026


def model_b(p12, ab):
    P, p1 = stationary_P(p12)
    return make_model(AB_SETS[ab]["a"], AB_SETS[ab]["b"], P=P, p1=p1, **BASE_B)


def run_b(m, eps_grid, n_short, n_long, seeds_short, seeds_long, n_part, seed0=0):
    """Per-sequence errors along the path THGMSM(eps).

    Returns a dict of arrays, the first axis being eps:
      short (n_eps, seeds_short, 3): sum_n (f1 - e)^2, sum_n (f2 - e)^2, sum_n Var[X_n | y_{1:n}],
          e = exact (enumeration) filter, on sequences of length n_short;
      long (n_eps, seeds_long, 3): mean_n (ref - x)^2, mean_n (f1 - x)^2, mean_n (f2 - x)^2,
          ref = RBPF, on sequences of length n_long;
      long_head (n_eps, seeds_long, 3): the same means over the first n_short time steps only;
      check (2, seeds_long, 2) at eps = 0 and eps = 1, where the exact filter of the CGOMSM1 (resp. of the
          CGOMSM2) is the optimal one: sum_n (ref - e)^2 and sum_n Var[X_n | y_{1:n}].
    Common random numbers across eps: sequence s always uses the same seeds.
    """
    m1, m2 = cgomsm1(m), cgomsm2(m)
    n_eps = len(eps_grid)
    short = np.empty((n_eps, seeds_short, 3))
    long = np.empty((n_eps, seeds_long, 3))
    head = np.empty((n_eps, seeds_long, 3))
    check = np.full((2, seeds_long, 2), np.nan)
    for i, eps in enumerate(eps_grid):
        me = path(m, eps)
        assert spectral_radius(me) < 1, (eps, spectral_radius(me))
        for s in range(seeds_short):
            rng = np.random.default_rng(seed0 + s)  # common random numbers across eps
            _, x, y = simulate(me, n_short, rng)
            e, v = exact_filter(me, y)
            f1, f2 = cgomsm1_filter(m1, y), cgomsm2_filter(m2, y)
            short[i, s] = np.sum((f1 - e) ** 2), np.sum((f2 - e) ** 2), np.sum(v)
        for s in range(seeds_long):
            rng = np.random.default_rng(10_000 + seed0 + s)
            _, x, y = simulate(me, n_long, rng)
            ref = rbpf(me, y, n_part, np.random.default_rng(50_000 + s))
            f1, f2 = cgomsm1_filter(m1, y), cgomsm2_filter(m2, y)
            long[i, s] = np.mean((ref - x) ** 2), np.mean((f1 - x) ** 2), np.mean((f2 - x) ** 2)
            h = slice(0, n_short)
            head[i, s] = (
                np.mean((ref[h] - x[h]) ** 2),
                np.mean((f1[h] - x[h]) ** 2),
                np.mean((f2[h] - x[h]) ** 2),
            )
            if eps in (0.0, 1.0):  # THGMSM(0) is the CGOMSM1 and THGMSM(1) the CGOMSM2
                e, v = (cgomsm1_filter if eps == 0.0 else cgomsm2_filter)(me, y, return_var=True)
                check[int(eps)][s] = np.sum((ref - e) ** 2), np.sum(v)
        print(
            f"eps={eps:.2f}  short: {_seqsum(short[i])[:2] / _seqsum(short[i])[2]}"
            f"  long: {_seqsum(long[i])[1:] / _seqsum(long[i])[0] - 1}",
            flush=True,
        )
    return dict(short=short, long=long, long_head=head, check=check)


def _seqsum(a):
    """Sum over the sequence axis (axis 0), accumulated in sequence order."""
    tot = np.zeros(a.shape[1:])
    for row in a:
        tot = tot + row
    return tot


def _ratio_hw(num, den):
    """Ratio of means sum(num)/sum(den) and its 95% half-width by the delta method."""
    R = _seqsum(num) / _seqsum(den)
    n = len(num)
    return R, 1.96 * np.std(num - R * den, ddof=1) / np.sqrt(n) / np.mean(den)


def _crossing(e, d):
    """First zero of the piecewise-linear interpolation of d(e); nan if none."""
    for i in range(len(e) - 1):
        if d[i] == 0:
            return float(e[i])
        if d[i] * d[i + 1] < 0:
            return float(e[i] - d[i] * (e[i + 1] - e[i]) / (d[i + 1] - d[i]))
    return float("nan")


def crossing(rows, field):
    e = np.array([r["eps"] for r in rows])
    d = np.array([r[field]["cgomsm1"] - r[field]["cgomsm2"] for r in rows])
    return _crossing(e, d)


def _boot_b(eps, short, long, rng, n_boot=N_BOOT):
    """Bootstrap over sequences (paired across eps): eps* (both references) and max_eps of the excesses."""
    ib_s = rng.integers(0, short.shape[1], size=(n_boot, short.shape[1]))
    ib_l = rng.integers(0, long.shape[1], size=(n_boot, long.shape[1]))
    S = short[:, ib_s].sum(axis=2)  # (n_eps, n_boot, 3)
    L = long[:, ib_l].sum(axis=2)
    ex_s = S[..., :2] / S[..., [2]]
    ex_l = L[..., 1:] / L[..., [0]] - 1.0
    out = {
        "eps_star_short": [_crossing(eps, ex_s[:, b, 0] - ex_s[:, b, 1]) for b in range(n_boot)],
        "eps_star_long": [_crossing(eps, ex_l[:, b, 0] - ex_l[:, b, 1]) for b in range(n_boot)],
        "max_excess_long_cgomsm1": ex_l[:, :, 0].max(axis=0),
        "max_excess_long_cgomsm2": ex_l[:, :, 1].max(axis=0),
    }
    res = {}
    for k, v in out.items():
        v = np.asarray(v, dtype=float)
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        res[k] = dict(ci95=[float(lo), float(hi)], sd=float(np.nanstd(v, ddof=1)))
    return res


def summarize_b(eps_grid, raw, n_short, n_long, seed=BOOT_SEED):
    """Rows of exp_b.json, crossing points and uncertainties from the per-sequence arrays of run_b.

    ``excess_long_head`` / ``excess_long_tail``: excess of the long-sequence (RBPF) reference restricted
    to the first n_short time steps / to the following ones (initial transient vs. stationary regime).
    """
    short, long, head, check = raw["short"], raw["long"], raw["long_head"], raw["check"]
    rows = []
    for i, eps in enumerate(eps_grid):
        es, hs = zip(*(_ratio_hw(short[i, :, k], short[i, :, 2]) for k in (0, 1)))
        el, hl = zip(*(_ratio_hw(long[i, :, k], long[i, :, 0]) for k in (1, 2)))
        eh, hh = zip(*(_ratio_hw(head[i, :, k], head[i, :, 0]) for k in (1, 2)))
        eh = np.array(eh) - 1.0
        mse = _seqsum(long[i]) / long.shape[1]
        tail = (n_long * _seqsum(long[i]) - n_short * _seqsum(head[i])) / (n_long - n_short)
        et = tail[1:] / tail[0] - 1.0
        rows.append(
            dict(
                eps=float(eps),
                rho=None,
                excess_short=dict(cgomsm1=float(es[0]), cgomsm2=float(es[1])),
                hw95_short=dict(cgomsm1=float(hs[0]), cgomsm2=float(hs[1])),
                mse_long=dict(rbpf=float(mse[0]), cgomsm1=float(mse[1]), cgomsm2=float(mse[2])),
                excess_long=dict(
                    cgomsm1=float(mse[1] / mse[0] - 1), cgomsm2=float(mse[2] / mse[0] - 1)
                ),
                hw95_long=dict(cgomsm1=float(hl[0]), cgomsm2=float(hl[1])),
                excess_long_head=dict(cgomsm1=float(eh[0]), cgomsm2=float(eh[1])),
                hw95_long_head=dict(cgomsm1=float(hh[0]), cgomsm2=float(hh[1])),
                excess_long_tail=dict(cgomsm1=float(et[0]), cgomsm2=float(et[1])),
            )
        )
    res = dict(
        rows=rows,
        eps_star_short=crossing(rows, "excess_short"),
        eps_star_long=crossing(rows, "excess_long"),
    )
    for key in ("cgomsm1", "cgomsm2"):
        i = int(np.argmax([r["excess_long"][key] for r in rows]))
        res[f"max_excess_long_{key}"] = dict(
            value=rows[i]["excess_long"][key], eps=rows[i]["eps"], hw95=rows[i]["hw95_long"][key]
        )
    res["bootstrap"] = _boot_b(np.asarray(eps_grid), short, long, np.random.default_rng(seed))
    res["rbpf_check_long"] = {
        f"eps={e:g}": float(np.sum(check[k, :, 0]) / np.sum(check[k, :, 1]))
        for k, e in enumerate((0.0, 1.0))
    }
    return res


def transient_b(m, n_short=10, n_long=200, n_seq=1000, seed0=0):
    """Initial transient of the excess, at the two ends of the path, where the optimal filter is exact.

    THGMSM(0) is the CGOMSM1 and THGMSM(1) the CGOMSM2, so that the exact filter of the other family can
    be compared with the optimal one on long sequences without particle filter. For each end, returns
    the excess sum (f - e)^2 / sum Var[X_n | y_{1:n}] of the other filter over the first n_short steps
    ("head") and over the following ones ("tail"), with delta-method 95% half-widths, and per time step.
    """
    res = {}
    for eps, opt, oth in ((0.0, "cgomsm1", "cgomsm2"), (1.0, "cgomsm2", "cgomsm1")):
        me = path(m, eps)
        f_opt = cgomsm1_filter if opt == "cgomsm1" else cgomsm2_filter
        f_oth = cgomsm1_filter if oth == "cgomsm1" else cgomsm2_filter
        m_oth = cgomsm1(m) if oth == "cgomsm1" else cgomsm2(m)
        sq = np.empty((n_seq, n_long))
        var = np.empty((n_seq, n_long))
        for s in range(n_seq):
            _, _, y = simulate(me, n_long, np.random.default_rng(seed0 + 30_000 + s))
            e, var[s] = f_opt(me, y, return_var=True)
            sq[s] = (f_oth(m_oth, y) - e) ** 2
        out = dict(filter=oth, n_seq=n_seq)
        for name, sl in (("head", slice(0, n_short)), ("tail", slice(n_short, n_long))):
            R, hw = _ratio_hw(sq[:, sl].sum(axis=1), var[:, sl].sum(axis=1))
            out[name], out[f"{name}_hw95"] = float(R), float(hw)
        out["by_n"] = (sq.sum(axis=0) / var.sum(axis=0)).tolist()
        res[f"eps={eps:g}"] = out
    return res


def add_transient_b(out: Path, n_seq=1000):
    """Add the transient study (transient_b) to an existing exp_b.json."""
    full = json.loads((out / "exp_b.json").read_text())
    cfg = full["cfg"]
    for k, p12, ab in EXPS_B:
        full["exps"][str(k)]["transient"] = transient_b(
            model_b(p12, ab), cfg["n_short"], cfg["n_long"], n_seq, seed0=100 * k
        )
    cfg["transient"] = dict(
        n_seq=n_seq, seeds="experiment k, sequence s: default_rng(30000 + 100 k + s)"
    )
    (out / "exp_b.json").write_text(json.dumps(full, indent=1))
    return full


def validate_rbpf(m, n=10, seeds=50, n_part=1000):
    """Relative distance sum_n (rbpf - e)^2 / sum_n Var[X_n|y] to the exact (enumeration) filter."""
    num = den = 0.0
    for s in range(seeds):
        _, _, y = simulate(m, n, np.random.default_rng(900 + s))
        e, v = exact_filter(m, y)
        f = rbpf(m, y, n_part, np.random.default_rng(1900 + s))
        num += np.sum((f - e) ** 2)
        den += np.sum(v)
    return float(num / den)


def exp_b(
    out: Path, seeds_short=300, seeds_long=100, n_part=1000, n_short=10, n_long=200, n_eps=21
):
    eps_grid = np.linspace(0, 1, n_eps)
    cfg = dict(
        n_short=n_short,
        n_long=n_long,
        seeds_short=seeds_short,
        seeds_long=seeds_long,
        n_part=n_part,
        n_eps=n_eps,
        n_boot=N_BOOT,
        boot_seed=BOOT_SEED,
        rbpf_check_short=dict(eps=0.5, n=n_short, seeds=50),
        seeds=(
            "experiment k, sequence s: short sequences default_rng(100 k + s); long sequences "
            "default_rng(10000 + 100 k + s); RBPF default_rng(50000 + s); RBPF check at eps = 0.5: "
            "sequences default_rng(900 + s), RBPF default_rng(1900 + s)"
        ),
    )
    res, arrays = {}, {}
    out.mkdir(parents=True, exist_ok=True)
    for k, p12, ab in EXPS_B:
        m = model_b(p12, ab)
        chk = validate_rbpf(path(m, 0.5), n=n_short, n_part=n_part)
        print(f"--- experiment {k}: p12={p12}, set {ab}; rbpf check = {chk:.2e}", flush=True)
        raw = run_b(m, eps_grid, n_short, n_long, seeds_short, seeds_long, n_part, seed0=100 * k)
        summ = summarize_b(eps_grid, raw, n_short, n_long)
        for r in summ["rows"]:
            r["rho"] = float(spectral_radius(path(m, r["eps"])))
        check_long = summ.pop("rbpf_check_long")
        res[str(k)] = dict(
            p12=p12,
            ab=ab,
            a=AB_SETS[ab]["a"],
            b=AB_SETS[ab]["b"],
            **summ,
            rbpf_check={f"eps=0.5,N={n_short}": chk, **check_long},
        )
        arrays.update({f"exp{k}_{name}": v for name, v in raw.items()})
        (out / "exp_b.json").write_text(json.dumps(dict(base=BASE_B, cfg=cfg, exps=res), indent=1))
        np.savez_compressed(out / "exp_b_seq.npz", eps=eps_grid, **arrays)
    add_transient_b(out)
    res = json.loads((out / "exp_b.json").read_text())["exps"]
    _table_b(res, out / "table_b.tex")
    _plot_b(res, out / "fig_b_homotopy.pdf")
    return res


TABLE_B_SYMBOL = r"\eta"  # LaTeX symbol of the relative excess MSE in the header of Table II


def _table_b(res, path, sym=TABLE_B_SYMBOL):
    """Table II: maximum relative excess MSE over eps and crossing point (RBPF reference, n_long)."""
    lines = [
        r"\begin{tabular}{@{}ccccccccc@{}}",
        r"\toprule",
        r"exp. & $p_{12}$ & $a_{1}$ & $a_{2}$ & $b_{1}$ & $b_{2}$"
        rf" & $\max_{{\varepsilon}} {sym}_{{1}}$ & $\max_{{\varepsilon}} {sym}_{{2}}$"
        r" & $\varepsilon^{*}$ \\",
        r"\midrule",
    ]
    for k, r in res.items():
        vals = [
            _num(r["p12"], ".2f"),
            *(_num(v, "g") for v in (*r["a"], *r["b"])),
            _num(r["max_excess_long_cgomsm1"]["value"], ".2f"),
            _num(r["max_excess_long_cgomsm2"]["value"], ".3f"),
            _num(r["eps_star_long"], ".2f"),
        ]
        lines.append(f"{k} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def _plot_b(res, path):
    plt = _mpl()
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.3), sharex=True)
    for ax, (k, r) in zip(axes.ravel(), res.items()):
        e = np.array([x["eps"] for x in r["rows"]])
        for key, col, mk, lab in (
            ("cgomsm1", "C0", "o", "CGOMSM1"),
            ("cgomsm2", "C3", "s", "CGOMSM2"),
        ):
            for fld, hw, ls, mfc, alpha, suffix in (
                ("excess_long", "hw95_long", "-", col, 0.25, r", $N=200$"),
                ("excess_short", "hw95_short", ":", "white", 0.12, r", $N=10$"),
            ):
                y = np.array([x[fld][key] for x in r["rows"]])
                ci = np.array([x[hw][key] for x in r["rows"]])
                ax.fill_between(e, y - ci, y + ci, color=col, alpha=alpha, lw=0)
                ax.plot(
                    e,
                    y,
                    ls,
                    color=col,
                    marker=mk,
                    ms=3,
                    mfc=mfc,
                    markevery=4,
                    lw=1.0,
                    label=lab + suffix,
                )
        ax.set_title(rf"exp. {k}: $p_{{12}}={r['p12']}$, set {r['ab']}")
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel(r"$\varepsilon$")
    for ax in axes[:, 0]:
        ax.set_ylabel("relative excess MSE")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, handlelength=2.6)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    _savefig(fig, path)
    plt.close(fig)


# ----------------------------------------------------------------------------- Section V-C: real data
SUB_C = 10  # 100 Hz -> 10 Hz
TEST_EVERY = 4  # every 4th file (sorted order) is held out
MIN_RAW = 500  # recordings of at most 5 s (500 samples at 100 Hz) are discarded, at every rate


def _vehicle_files(data_dir: Path):
    from prg.experiments.make_vehicle_consigne_fig import _ensure_data

    return _ensure_data(Path(data_dir))


def load_vehicle(data_dir: Path, sub: int = SUB_C, info: dict | None = None):
    """List of per-file arrays [yaw (deg/s), a_lat (g), steering (deg), speed (km/h)], subsampled.

    The recordings of at most ``MIN_RAW`` samples at 100 Hz are discarded before subsampling, so that
    the same files are used at every sampling rate. ``info``, if given, receives the file counts.
    """
    import pandas as pd

    from prg.experiments.make_vehicle_consigne_fig import ALAT, STR, WS, YAW, _num

    files = _vehicle_files(data_dir)
    seqs, dropped, nan_rows = [], [], 0
    for f in files:
        d = pd.read_parquet(f)
        if len(d) <= MIN_RAW:
            dropped.append(dict(file=Path(f).name, samples_100Hz=len(d), seconds=len(d) / 100))
            continue
        for c in [YAW, ALAT, STR, *WS]:
            d[c] = _num(d[c])
        d["spd"] = d[WS].mean(axis=1)
        d = d[[YAW, ALAT, STR, "spd"]].iloc[::sub]  # subsample the regular 100 Hz grid first
        nan_rows += int(d.isna().any(axis=1).sum())
        seqs.append(d.dropna().to_numpy())
    if info is not None:
        info.update(
            n_files_archive=len(files),
            n_files_used=len(seqs),
            discarded=dropped,
            nan_rows_dropped=nan_rows,
            data_sha256=_fingerprint(files),
        )
    return seqs


def _fingerprint(files):
    """SHA-256 of the sorted (file name, content) pairs of the dataset."""
    h = hashlib.sha256()
    for f in sorted(files, key=lambda p: Path(p).name):
        h.update(Path(f).name.encode())
        h.update(Path(f).read_bytes())
    return h.hexdigest()


def _ols(target, design):
    b, *_ = np.linalg.lstsq(design, target, rcond=None)
    return b, float(np.mean((target - design @ b) ** 2))


def _xyu(seq, sc, use_input):
    """Scaled state, observation, and input u_n = (steering_{n+1}, 1) stored at index n + 1.

    Without input, the steering angle is set to 0 and only the constant term of u_n is kept.
    """
    x, y = seq[:, 0] / sc[0], seq[:, 1] / sc[1]
    st = seq[:, 2] / sc[2] if use_input else np.zeros(len(seq))
    return x, y, np.c_[st, np.ones(len(seq))]


def fit_c(train, K, edges, sc, use_input=True):
    """Per-regime fits of the THGMSM, CGOMSM1 and CGOMSM2 (regimes labelled by the speed band).

    The transition densities p(x', y' | x, y, u, r') are fitted by maximum likelihood (least squares).
    The other parameters are simple moment estimates: P from the regime-transition counts plus one
    pseudo-count per entry, and a centred initial law whose p(r_1) and Sigma_{Z_1, r_1} are the regime
    frequencies and the second moments of all the training samples of each regime.
    """
    P = np.ones((K, K))  # one pseudo-count per entry
    rows = {k: [] for k in range(K)}
    first = {k: [] for k in range(K)}
    for s in train:
        x, y, u = _xyu(s, sc, use_input)
        r = np.searchsorted(edges, s[:, 3])
        np.add.at(P, (r[:-1], r[1:]), 1)
        for k in range(K):
            first[k].append(np.c_[x[r == k], y[r == k]])
            i = np.where(r[1:] == k)[0]
            rows[k].append(np.c_[x[i], y[i], u[i + 1], x[i + 1], y[i + 1]])
    P /= P.sum(axis=1, keepdims=True)
    p1 = np.array([sum(len(a) for a in first[k]) for k in range(K)], float)
    mods = {"thgmsm": [], "cgomsm1": [], "cgomsm2": []}
    for k in range(K):
        D = np.vstack(rows[k])
        xn, yn, un, xp, yp = D[:, 0], D[:, 1], D[:, 2:4], D[:, 4], D[:, 5]
        Z1 = np.vstack(first[k])
        # Sigma_{Z_1, k}: second moments of all the training samples of regime k (centred initial law)
        base = {"S1": Z1.T @ Z1 / len(Z1)}
        des = np.c_[xn, yn, un]
        bx, _ = _ols(xp, des)
        by, _ = _ols(yp, des)
        SW = np.cov(np.c_[xp - des @ bx, yp - des @ by].T, bias=True)
        mods["thgmsm"].append(
            dict(
                base,
                a=bx[0],
                b=bx[1],
                lx=bx[2:],
                c=by[0],
                d=by[1],
                ly=by[2:],
                su=SW[0, 0],
                rho=SW[0, 1],
                sv=SW[1, 1],
            )
        )
        b1, sv = _ols(yp, np.c_[yn, un])
        d, ly = b1[0], b1[1:]
        b2, se = _ols(xp, np.c_[xn, yn, yp, un])
        g = b2[2]
        mods["cgomsm1"].append(
            dict(
                base,
                a=b2[0],
                b=b2[1] + g * d,
                lx=b2[3:] + g * ly,
                c=0.0,
                d=d,
                ly=ly,
                su=se + g * g * sv,
                rho=g * sv,
                sv=sv,
            )
        )
        b1, sv = _ols(yp, des)
        c, d, ly = b1[0], b1[1], b1[2:]
        b2, gam = _ols(xp, np.c_[yp, un])
        M = b2[0]
        mods["cgomsm2"].append(
            dict(
                base,
                a=M * c,
                b=M * d,
                lx=b2[1:] + M * ly,
                c=c,
                d=d,
                ly=ly,
                su=gam + M * M * sv,
                rho=M * sv,
                sv=sv,
            )
        )
    return {n: {"K": K, "P": P, "p1": p1 / p1.sum(), "reg": v} for n, v in mods.items()}


def _init_c(m, y1):
    """log p(r_1, y_1), E[X_1 | r_1, y_1], Var[X_1 | r_1, y_1] for the centred initial law."""
    S = np.array([R["S1"] for R in m["reg"]])
    ll = np.log(m["p1"]) + _lognorm(y1, 0.0, S[:, 1, 1])
    return ll, S[:, 0, 1] / S[:, 1, 1] * y1, S[:, 0, 0] - S[:, 0, 1] ** 2 / S[:, 1, 1]


def _lse(v):
    a = np.max(v)
    return a + np.log(np.sum(np.exp(v - a)))


def _check_family(m, family):
    if family not in ("cgomsm1", "cgomsm2"):
        raise ValueError(f"unknown family {family!r}: expected 'cgomsm1' or 'cgomsm2'")
    for k, R in enumerate(m["reg"]):
        if family == "cgomsm1":
            ok = np.isclose(R["c"], 0.0, rtol=0.0, atol=1e-12)
        else:
            M = R["rho"] / R["sv"]
            ok = np.isclose(R["a"], M * R["c"], rtol=1e-9, atol=1e-12) and np.isclose(
                R["b"], M * R["d"], rtol=1e-9, atol=1e-12
            )
        if not ok:
            raise ValueError(f"regime {k} violates the {family.upper()} constraints")


def cgo_filter_c(m, y, u, family):
    """Exact filter of a CGOMSM1 or CGOMSM2 with input; returns E[X_n | y_{1:n}] and log p(y_{1:N})."""
    _check_family(m, family)
    K, N, Rg = m["K"], len(y), m["reg"]
    ll0, mu, var = _init_c(m, y[0])
    LL = _lse(ll0)
    pi = np.exp(ll0 - LL)
    E = np.empty(N)
    E[0] = pi @ mu
    for n in range(1, N):
        lk = np.empty((K, K))
        mjk = np.empty((K, K))
        for k, R in enumerate(Rg):
            if family == "cgomsm1":  # y_{n+1} | r_{n+1}, y_n; X_{n+1} | r_n, r_{n+1}, y_{1:n+1}
                my = R["d"] * y[n - 1] + R["ly"] @ u[n]
                lk[:, k] = _lognorm(y[n], my, R["sv"])
                mjk[:, k] = (
                    R["a"] * mu
                    + R["b"] * y[n - 1]
                    + R["lx"] @ u[n]
                    + R["rho"] / R["sv"] * (y[n] - my)
                )
            else:  # Prop. 3 transition (first step from the initial law), Prop. 4 read-out
                my = R["c"] * mu + R["d"] * y[n - 1] + R["ly"] @ u[n]
                lk[:, k] = _lognorm(y[n], my, R["c"] ** 2 * var + R["sv"])
                M = R["rho"] / R["sv"]
                mjk[:, k] = M * y[n] + (R["lx"] - M * R["ly"]) @ u[n]
        L = np.log(np.maximum(pi, 1e-300))[:, None] + np.log(m["P"]) + lk
        a = L.max()
        J = np.exp(L - a)
        LL += a + np.log(J.sum())
        col = J.sum(axis=0)
        mu = (J / np.maximum(col, 1e-300)[None, :] * mjk).sum(
            axis=0
        )  # underflowed regimes: weight 0
        pi = col / col.sum()
        if family == "cgomsm2":
            var = np.array([R["su"] - R["rho"] ** 2 / R["sv"] for R in Rg])
        E[n] = pi @ mu
    return E, LL


def rbpf_c(m, y, u, n_part, rng):
    """RBPF (optimal proposal) for a THGMSM with input; exact Kalman filter when K = 1."""
    K, N, Rg = m["K"], len(y), m["reg"]
    ll0, m0, v0 = _init_c(m, y[0])
    LL = _lse(ll0)
    r = rng.choice(K, size=n_part, p=np.exp(ll0 - LL))
    mu, Pv = m0[r], v0[r]
    w = np.full(n_part, 1.0 / n_part)
    E = np.empty(N)
    E[0] = w @ mu
    for n in range(1, N):
        cand = []
        for R in Rg:
            my = R["c"] * mu + R["d"] * y[n - 1] + R["ly"] @ u[n]
            vy = R["c"] ** 2 * Pv + R["sv"]
            mx = R["a"] * mu + R["b"] * y[n - 1] + R["lx"] @ u[n]
            cxy = R["a"] * R["c"] * Pv + R["rho"]
            gain = cxy / vy
            cand.append(
                (
                    _lognorm(y[n], my, vy),
                    mx + gain * (y[n] - my),
                    R["a"] ** 2 * Pv + R["su"] - gain * cxy,
                )
            )
        lj = np.stack([np.log(m["P"][r, k]) + cand[k][0] for k in range(K)])
        lmax = lj.max(axis=0)
        pj = np.exp(lj - lmax)
        tot = pj.sum(axis=0)
        logw = np.log(w) + lmax + np.log(tot)
        LL += _lse(logw)
        ch = np.minimum(
            (rng.random(n_part)[None, :] > np.cumsum(pj / tot, axis=0)).sum(axis=0), K - 1
        )
        mu = np.choose(ch, [c[1] for c in cand])
        Pv = np.choose(ch, [c[2] for c in cand])
        r = ch
        w = np.exp(logw - logw.max())
        w /= w.sum()
        E[n] = w @ mu
        if 1.0 / np.sum(w**2) < n_part / 2:  # systematic resampling
            v = (rng.random() + np.arange(n_part)) / n_part
            idx = np.minimum(np.searchsorted(np.cumsum(w), v), n_part - 1)
            mu, Pv, r = mu[idx], Pv[idx], r[idx]
            w = np.full(n_part, 1.0 / n_part)
    return E, LL


def _split(seqs, K):
    test = seqs[TEST_EVERY - 1 :: TEST_EVERY]
    train = [s for i, s in enumerate(seqs) if i % TEST_EVERY != TEST_EVERY - 1]
    allv = np.vstack(train)
    edges = np.quantile(allv[:, 3], np.linspace(0, 1, K + 1)[1:-1])
    return train, test, edges, allv[:, :3].std(axis=0)


def _score(test, sc, use_input, run):
    se, ll, n = 0.0, 0.0, 0
    for s in test:
        x, y, u = _xyu(s, sc, use_input)
        E, L = run(y, u)
        se += float(np.sum((E * sc[0] - s[:, 0]) ** 2))
        ll += L - len(y) * np.log(sc[1])  # log-likelihood of the unscaled a_lat (in g)
        n += len(y)
    return {"rmse": float(np.sqrt(se / n)), "loglik": float(ll / n)}


def exp_c(out: Path, data_dir: Path, K: int = 2, n_part: int = 500):
    out.mkdir(parents=True, exist_ok=True)
    # one generator shared by the RBPF runs below, in this order: kept so that Table III is reproduced
    rng = np.random.default_rng(0)
    res = {"K": K, "sub": SUB_C, "min_raw_samples": MIN_RAW, "main": {}, "grid": []}
    info = {}
    seqs = load_vehicle(data_dir, info=info)
    res["data"] = info
    train, test, edges, sc = _split(seqs, K)
    allv = np.vstack(train)
    res.update(
        n_train=len(train),
        n_test=len(test),
        n_test_samples=int(sum(len(s) for s in test)),
        speed_edges_kmh=edges.tolist(),
        scales=dict(yaw_deg_s=float(sc[0]), alat_g=float(sc[1]), steering_deg=float(sc[2])),
        corr_train=dict(
            yaw_alat=float(np.corrcoef(allv[:, 0], allv[:, 1])[0, 1]),
            yaw_steering=float(np.corrcoef(allv[:, 0], allv[:, 2])[0, 1]),
            alat_steering=float(np.corrcoef(allv[:, 1], allv[:, 2])[0, 1]),
        ),
    )
    for use_input in (True, False):
        mods = fit_c(train, K, edges, sc, use_input)
        one = fit_c(train, 1, np.array([]), sc, use_input)
        runs = {
            "cgomsm1": lambda y, u, m=mods["cgomsm1"]: cgo_filter_c(m, y, u, "cgomsm1"),
            "cgomsm2": lambda y, u, m=mods["cgomsm2"]: cgo_filter_c(m, y, u, "cgomsm2"),
            "thgmsm_rbpf": lambda y, u, m=mods["thgmsm"]: rbpf_c(m, y, u, n_part, rng),
            "kalman_K1": lambda y, u, m=one["thgmsm"]: rbpf_c(m, y, u, 1, rng),
        }
        key = "input" if use_input else "no_input"
        res["main"][key] = {name: _score(test, sc, use_input, f) for name, f in runs.items()}
        if use_input:  # check: RBPF on each CGOMSM reproduces its exact filter
            res["rbpf_check"] = {
                fam: _score(test, sc, True, lambda y, u, m=mods[fam]: rbpf_c(m, y, u, n_part, rng))
                for fam in ("cgomsm1", "cgomsm2")
            }
            # fitted parameters, on the standardized variables (see "scales")
            res["params"] = {
                fam: [
                    {k: np.asarray(v).tolist() for k, v in R.items() if k != "S1"}
                    for R in mods[fam]["reg"]
                ]
                for fam in mods
            }
            res["P"] = mods["thgmsm"]["P"].tolist()
            # trace for the figure: longest test file
            s = max(test, key=len)
            x, y, u = _xyu(s, sc, True)
            res["trace"] = {
                "t": (np.arange(len(s)) * SUB_C / 100.0).tolist(),
                "yaw": s[:, 0].tolist(),
                "speed": s[:, 3].tolist(),
                **{
                    fam: (cgo_filter_c(mods[fam], y, u, fam)[0] * sc[0]).tolist()
                    for fam in ("cgomsm1", "cgomsm2")
                },
            }
    # robustness of the comparison: exact CGOMSM filters for other sampling rates and K
    for sub, k in itertools.product((5, 10, 20), (2, 3)):
        sq = seqs if sub == SUB_C else load_vehicle(data_dir, sub)
        tr, te, ed, s_ = _split(sq, k)
        mm = fit_c(tr, k, ed, s_, True)
        row = {"sub": sub, "rate_hz": 100 // sub, "K": k, "n_train": len(tr), "n_test": len(te)}
        for fam in ("cgomsm1", "cgomsm2"):
            row[fam] = _score(te, s_, True, lambda y, u, m=mm[fam], f=fam: cgo_filter_c(m, y, u, f))
        res["grid"].append(row)
    (out / "exp_c.json").write_text(json.dumps(res, indent=1))
    _table_c(res, out / "table_c.tex")
    _plot_c(res, out / "fig_c_vehicle.pdf")
    return res


def _table_c(res, path):
    m = res["main"]
    lines = [
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"filter & \multicolumn{2}{c}{yaw-rate RMSE (deg/s)} & log-lik. of $Y$ \\",
        r"\cmidrule(lr){2-3}",  # shows which columns the multicolumn header spans
        r" & with $\delta$ & without $\delta$ & with $\delta$ \\",
        r"\midrule",
    ]
    names = (
        ("cgomsm1", "CGOMSM1 (exact)"),
        ("cgomsm2", "CGOMSM2 (exact)"),
        ("thgmsm_rbpf", "THGMSM (RBPF)"),
        ("kalman_K1", "THGMSM, $K = 1$ (Kalman)"),
    )
    for key, lab in names:
        lines.append(
            f"{lab} & ${m['input'][key]['rmse']:.2f}$ & ${m['no_input'][key]['rmse']:.2f}$"
            f" & ${m['input'][key]['loglik']:.2f}$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def _plot_c(res, path):
    plt = _mpl()
    tr = res["trace"]
    t = np.array(tr["t"])
    sel = slice(0, min(len(t), 600))  # first 60 s
    fig, ax = plt.subplots(figsize=(3.45, 2.1))
    ax.plot(t[sel], np.array(tr["yaw"])[sel], color="k", lw=1.1, label="measured")
    ax.plot(t[sel], np.array(tr["cgomsm1"])[sel], color="C0", lw=0.9, ls="--", label="CGOMSM1")
    ax.plot(t[sel], np.array(tr["cgomsm2"])[sel], color="C3", lw=0.9, ls=":", label="CGOMSM2")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("yaw rate (deg/s)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    _savefig(fig, path)
    plt.close(fig)


# ----------------------------------------------------------------------------- computing times
def _time_per_call(fn, repeat):
    fn()  # warm-up
    t = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return float(np.median(t))


def _cpu_name():
    if platform.system() == "Darwin":
        import subprocess

        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def timing(out: Path, n_long=200, n_part=1000, repeat=20):
    """Median wall-clock time per time step of the filters of case b (single thread, scalar, K = 2)."""
    m = model_b(0.05, "A")
    m1, m2, mh = cgomsm1(m), cgomsm2(m), path(m, 0.5)
    _, _, y = simulate(mh, n_long, np.random.default_rng(0))
    res = dict(
        machine=dict(
            platform=platform.platform(),
            processor=_cpu_name(),
            python=platform.python_version(),
            numpy=np.__version__,
        ),
        n=n_long,
        n_part=n_part,
        ms_per_step={
            "cgomsm1_exact": _time_per_call(lambda: cgomsm1_filter(m1, y), repeat) / n_long * 1e3,
            "cgomsm2_exact": _time_per_call(lambda: cgomsm2_filter(m2, y), repeat) / n_long * 1e3,
            "rbpf": _time_per_call(
                lambda: rbpf(mh, y, n_part, np.random.default_rng(1)), max(repeat // 4, 3)
            )
            / n_long
            * 1e3,
        },
        enumeration_ms_per_sequence={},
    )
    for N in (8, 10, 12, 14, 16, 18, 20):  # cost of the exact filter grows as 2^N
        res["enumeration_ms_per_sequence"][str(N)] = (
            _time_per_call(lambda N=N: exact_filter(mh, y[:N]), 5) * 1e3
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "timing.json").write_text(json.dumps(res, indent=1))
    return res


# ----------------------------------------------------------------------------- replot
def resummarize_b(out: Path):
    """Recompute the summaries of exp_b.json (rows, eps*, uncertainties) from exp_b_seq.npz."""
    path_json, path_npz = out / "exp_b.json", out / "exp_b_seq.npz"
    full = json.loads(path_json.read_text())
    cfg = full["cfg"]
    with np.load(path_npz) as z:
        eps = z["eps"]
        for k, r in full["exps"].items():
            raw = {n: z[f"exp{k}_{n}"] for n in ("short", "long", "long_head", "check")}
            summ = summarize_b(eps, raw, cfg["n_short"], cfg["n_long"], seed=cfg["boot_seed"])
            for new, old in zip(summ["rows"], r["rows"]):
                new["rho"] = old["rho"]
            short_check = {c: v for c, v in r["rbpf_check"].items() if c.startswith("eps=0.5")}
            check_long = summ.pop("rbpf_check_long")
            r.update(summ)
            r["rbpf_check"] = {**short_check, **check_long}
    path_json.write_text(json.dumps(full, indent=1))


def replot(out: Path):
    """Regenerate the tables and figures from the stored results (no simulation)."""
    if (out / "exp_a.json").exists():
        a = json.loads((out / "exp_a.json").read_text())
        _table_a(a["rows"], out / "table_a.tex")
        _plot_a(np.array(a["lam"]), a["curves"], out / "fig_a_lambda.pdf")
    if (out / "exp_b.json").exists():
        if (out / "exp_b_seq.npz").exists():
            resummarize_b(out)
        b = json.loads((out / "exp_b.json").read_text())["exps"]
        _table_b(b, out / "table_b.tex")
        _plot_b(b, out / "fig_b_homotopy.pdf")
    if (out / "exp_c.json").exists():
        c = json.loads((out / "exp_c.json").read_text())
        _table_c(c, out / "table_c.tex")
        _plot_c(c, out / "fig_c_vehicle.pdf")
    if (out / "exp_d.json").exists():
        from prg.experiments.tac_rivers import _plot_d, _table_d

        d = json.loads((out / "exp_d.json").read_text())
        _table_d(d, out / "table_d.tex")
        _plot_d(d, out / "fig_d_rivers.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "which",
        nargs="?",
        default="all",
        choices=("a", "b", "c", "d", "all", "timing", "replot", "transient"),
    )
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seeds-short", type=int, default=300)
    ap.add_argument("--seeds-long", type=int, default=100)
    ap.add_argument("--particles", type=int, default=1000)
    ap.add_argument("--n-eps", type=int, default=21)
    ap.add_argument("--data", type=Path, default=Path("data/real/vehicle"))
    a = ap.parse_args()
    if a.which == "replot":
        replot(a.out)
        return
    if a.which == "transient":  # adds the transient study to an existing exp_b.json
        for k, r in add_transient_b(a.out)["exps"].items():
            print(k, {e: (v["filter"], v["head"], v["tail"]) for e, v in r["transient"].items()})
        return
    if a.which == "timing":
        print(json.dumps(timing(a.out, n_part=a.particles), indent=1))
        return
    if a.which in ("a", "all"):
        for r in exp_a(a.out):
            print(
                f"exp {r['exp']}: tau1={r['tau1']:.4f} (MC {r['mc']['tau1']:.4f}"
                f" +- {r['mc']['tau1_se']:.1e})  tau2={r['tau2']:.4f} (MC {r['mc']['tau2']:.4f}"
                f" +- {r['mc']['tau2_se']:.1e})"
            )
    if a.which in ("b", "all"):
        res = exp_b(a.out, a.seeds_short, a.seeds_long, a.particles, n_eps=a.n_eps)
        for k, r in res.items():
            bs = r["bootstrap"]
            print(
                f"exp {k}: eps* short = {r['eps_star_short']:.3f} {bs['eps_star_short']['ci95']},"
                f" long = {r['eps_star_long']:.3f} {bs['eps_star_long']['ci95']}; rbpf check"
                f" {r['rbpf_check']}"
            )
    if a.which in ("c", "all"):
        res = exp_c(a.out, a.data)
        for key, v in res["main"].items():
            for name, sc in v.items():
                print(f"{key:8s} {name:12s} RMSE={sc['rmse']:.3f}  loglik={sc['loglik']:.3f}")
        for row in res["grid"]:
            print(row)
    if a.which == "d":  # not in "all": it uses the cached USGS data of data/real/rivers
        from prg.experiments.tac_rivers import exp_d

        exp_d(a.out)


if __name__ == "__main__":
    main()
