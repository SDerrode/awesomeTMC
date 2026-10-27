#!/usr/bin/env python3
"""Simulations of Section V of the TAC paper (CGOMSM1 vs CGOMSM2), scalar case q = s = 1, K = 2.

Model (dynamic form, regime j = R_{n+1}):
    [X_{n+1}; Y_{n+1}] = F_j [X_n; Y_n] + W_{n+1},  F_j = [[a_j, b_j], [c_j, d_j]],
    W ~ N(0, [[su_j, rho_j], [rho_j, sv_j]]),   Z_1 | r_1 = i ~ N(0, [[sx1_i, lam_i], [lam_i, sy1_i]]).
CGOMSM1 of a THGMSM: same parameters with c_j := 0.
CGOMSM2 of a THGMSM: same parameters with a_j := rho_j/sv_j c_j, b_j := rho_j/sv_j d_j.

Section V-A (``exp_a``): regime-known estimators at n = 2, closed forms (no simulation);
    tau_k = EI_k / ER, checked by Monte Carlo.
Section V-B (``exp_b``): path F^eps = (1 - eps) F^1 + eps F^2 from the CGOMSM1 (eps = 0) to the CGOMSM2
    (eps = 1). Data simulated with THGMSM(eps); the exact filters of the CGOMSM1 and of the CGOMSM2
    (frozen parameters) are compared with two references:
      * exact Bayes filter by enumeration of the 2^N regime paths (short horizon N_SHORT);
        excess = E(f - e)^2 / E Var[X | y] (orthogonality: no state noise in the estimate);
      * Rao-Blackwellized particle filter (optimal proposal) on N_LONG; excess = MSE_f / MSE_rbpf - 1.
    eps* = crossing point of the two excess curves.
Section V-C (``exp_c``): real driving data (Figshare 10.6084/m9.figshare.28078274.v1, aggressive driver, 100 Hz,
    subsampled to 10 Hz). X = yaw rate, Y = lateral acceleration, u_n = (steering angle at n + 1, 1),
    regime = speed band (labelled for training only, hidden in the filters). THGMSM, CGOMSM1 and CGOMSM2
    are fitted per regime by maximum likelihood; each fit reduces to two least-squares regressions:
      THGMSM : (x', y') on (x, y, u);
      CGOMSM1: y' on (y, u), then x' on (x, y, y', u);
      CGOMSM2: y' on (x, y, u), then x' on (y', u).
    Held-out files: yaw RMSE and log-likelihood of Y of the exact CGOMSM1 / CGOMSM2 filters, of an RBPF
    for the THGMSM, and of the Kalman filter of a single-regime THGMSM; with and without the steering input.

Run:  python -m prg.experiments.tac_simulations [a|b|c|all] [--out DIR] [--data DIR]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

OUT = Path("results/tac_simulations")


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
    """Symmetric stationary chain with p(r1=1)=0.5 and p(r1=1, r2=2)=p12."""
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
    assert np.allclose(m["sx1"], m["sx1"][0]) and np.allclose(m["lam"], m["lam"][0])
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


def exp_a_mc(m, n=200_000, seed=0):
    """Monte-Carlo check of exp_a_closed."""
    rng = np.random.default_rng(seed)
    pj = m["p1"] @ m["P"]
    r2 = rng.choice(2, size=n, p=pj)
    m1, m2 = cgomsm1(m), cgomsm2(m)
    se = np.zeros(3)
    for j in range(2):
        idx = r2 == j
        Z = rng.multivariate_normal(np.zeros(4), _cov12(m, 0, j), size=idx.sum())
        Y = Z[:, [1, 3]]
        for k, mm in enumerate((m, m1, m2)):
            se[k] += np.sum((Z[:, 2] - Y @ coeffs(mm, j)) ** 2)
    se /= n
    return dict(
        ER=float(se[0]), tau1=float((se[1] - se[0]) / se[0]), tau2=float((se[2] - se[0]) / se[0])
    )


# V-A design: coupling x gap to CGOMSM2 x noise correlation (2 x 2 x 2)
BASE_A = dict(
    d=[-0.5, -0.3], su=[1.0, 0.8], sv=[0.5, 1.0], sx1=[1.0, 1.0], sy1=[1.0, 1.0], lam=[0.5, 0.5]
)
LEVELS_A = dict(c=(0.2, 1.0), g=(0.05, 0.5), rf=(0.2, 0.7))


def design_a(c_lvl, g_lvl, rf, lam=0.5, p12=0.1):
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
            dict(rows=rows, lam=lam_grid.tolist(), curves={str(k): v for k, v in curves.items()}),
            indent=1,
        )
    )
    _table_a(rows, out / "table_a.tex")
    _plot_a(lam_grid, curves, out / "fig_a_lambda.pdf")
    return rows


def _table_a(rows, path):
    f = lambda v: f"{v:.2f}"
    lines = [
        r"\begin{tabular}{@{}l" + "c" * len(rows) + r"@{}}",
        r"\toprule",
        " & " + " & ".join(str(r["exp"]) for r in rows) + r" \\",
        r"\midrule",
    ]
    for key, lab in (("a", "a"), ("b", "b"), ("c", "c"), ("d", "d"), ("rho", r"\rho")):
        for j in range(2):
            lines.append(
                f"${lab}_{{{j + 1}}}$ & " + " & ".join(f(r[key][j]) for r in rows) + r" \\"
            )
    lines.append(r"\midrule")
    lines.append(r"$\tau_{1}$ & " + " & ".join(f"{r['tau1']:.3f}" for r in rows) + r" \\")
    lines.append(r"$\tau_{2}$ & " + " & ".join(f"{r['tau2']:.3f}" for r in rows) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def _plot_a(lam, curves, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    styles = {2: "-", 7: "--"}
    for k, cur in curves.items():
        ax.plot(lam, [c["tau1"] for c in cur], styles[k], color="C0", label=rf"$\tau_1$ (exp. {k})")
        ax.plot(lam, [c["tau2"] for c in cur], styles[k], color="C3", label=rf"$\tau_2$ (exp. {k})")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("relative error increase")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.5, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- simulation and filters (V-B)
def simulate(m, N, rng):
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


def _kalman_bank(m, y, regimes_seq_step, mprev, Pprev, j, yprev):
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
            ll, mn, Pn = _kalman_bank(m, y[n], None, mu, P, j, y[n - 1])
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
        cand = [_kalman_bank(m, y[n], None, mu, P, j, y[n - 1]) for j in range(2)]
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


def cgomsm1_filter(m, y):
    """Exact filter of a CGOMSM1 (c = 0): (R, Y) Markov, E[X_n | r_n, y_{1:n}] by an exact linear recursion."""
    assert np.allclose(m["c"], 0.0)
    N = len(y)
    ll0, mu, _ = _init_post(m, y[0])
    pi = np.exp(ll0 - ll0.max())
    pi /= pi.sum()
    E = np.empty(N)
    E[0] = pi @ mu
    for n in range(1, N):
        # transition of (R, Y): y_{n+1} | r_{n+1}=k, y_n ~ N(d_k y_n, sv_k)
        lk = _lognorm(y[n], m["d"] * y[n - 1], m["sv"])
        joint = pi[:, None] * m["P"] * np.exp(lk - lk.max())[None, :]  # (j, k)
        pi_new = joint.sum(axis=0)
        pi_new /= pi_new.sum()
        cond = joint / joint.sum(axis=0, keepdims=True)  # p(r_n = j | r_{n+1} = k, y_{1:n+1})
        # E[X_{n+1} | r_n, r_{n+1}, y] = a_k E[X_n|r_n,y] + b_k y_n + rho_k/sv_k (y_{n+1} - d_k y_n)
        g = m["rho"] / m["sv"]
        mu_jk = (
            m["a"][None, :] * mu[:, None]
            + (m["b"] * y[n - 1] + g * (y[n] - m["d"] * y[n - 1]))[None, :]
        )
        mu = (cond * mu_jk).sum(axis=0)
        pi = pi_new
        E[n] = pi @ mu
    return E


def cgomsm2_filter(m, y):
    """Exact filter of a CGOMSM2 (Prop. 3-4), with the first transition for a general initial law."""
    g = m["rho"] / m["sv"]
    assert np.allclose(m["a"], g * m["c"]) and np.allclose(m["b"], g * m["d"])
    M, Gam = g, m["su"] - m["rho"] ** 2 / m["sv"]
    N = len(y)
    ll0, mu1, v1 = _init_post(m, y[0])
    pi = np.exp(ll0 - ll0.max())
    pi /= pi.sum()
    E = np.empty(N)
    E[0] = pi @ mu1
    for n in range(1, N):
        if n == 1:  # first transition: moments of X_1 | r_1, y_1 from the initial law
            mx, vx = mu1, v1
        else:  # slaved law (3.17)
            mx, vx = M * y[n - 1], Gam
        mean = m["c"][None, :] * mx[:, None] + m["d"][None, :] * y[n - 1]  # (j, k)
        var = m["sv"][None, :] + m["c"][None, :] ** 2 * vx[:, None]
        lj = np.log(pi[:, None] * m["P"] + 1e-300) + _lognorm(y[n], mean, var)
        a = np.exp(lj - lj.max()).sum(axis=0)
        pi = a / a.sum()
        E[n] = pi @ (M * y[n])
    return E


# V-B design (V7, Tab. 5.2): common Sigma_Z1, Sigma_W, c, d; four experiments (p12 x {a, b})
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


def model_b(p12, ab):
    P, p1 = stationary_P(p12)
    return make_model(AB_SETS[ab]["a"], AB_SETS[ab]["b"], P=P, p1=p1, **BASE_B)


def run_b(m, eps_grid, n_short, n_long, seeds_short, seeds_long, n_part, seed0=0):
    m1, m2 = cgomsm1(m), cgomsm2(m)
    rows = []
    for eps in eps_grid:
        me = path(m, eps)
        assert spectral_radius(me) < 1, (eps, spectral_radius(me))
        d1 = d2 = vsum = 0.0
        cnt = 0
        for s in range(seeds_short):
            rng = np.random.default_rng(seed0 + s)  # common random numbers across eps
            _, x, y = simulate(me, n_short, rng)
            e, v = exact_filter(me, y)
            f1, f2 = cgomsm1_filter(m1, y), cgomsm2_filter(m2, y)
            d1 += np.sum((f1 - e) ** 2)
            d2 += np.sum((f2 - e) ** 2)
            vsum += np.sum(v)
            cnt += len(y)
        se = np.zeros(3)
        per_seed = []
        for s in range(seeds_long):
            rng = np.random.default_rng(10_000 + seed0 + s)
            _, x, y = simulate(me, n_long, rng)
            ref = rbpf(me, y, n_part, np.random.default_rng(50_000 + s))
            f1, f2 = cgomsm1_filter(m1, y), cgomsm2_filter(m2, y)
            loc = np.array(
                [np.mean((ref - x) ** 2), np.mean((f1 - x) ** 2), np.mean((f2 - x) ** 2)]
            )
            se += loc
            per_seed.append(loc)
        se /= seeds_long
        per_seed = np.array(per_seed)
        rel = per_seed[:, 1:] / per_seed[:, [0]] - 1.0
        rows.append(
            dict(
                eps=float(eps),
                rho=float(spectral_radius(me)),
                excess_short=dict(cgomsm1=float(d1 / vsum), cgomsm2=float(d2 / vsum)),
                mse_long=dict(rbpf=float(se[0]), cgomsm1=float(se[1]), cgomsm2=float(se[2])),
                excess_long=dict(
                    cgomsm1=float(se[1] / se[0] - 1), cgomsm2=float(se[2] / se[0] - 1)
                ),
                ci95_long=dict(
                    cgomsm1=float(1.96 * rel[:, 0].std(ddof=1) / np.sqrt(seeds_long)),
                    cgomsm2=float(1.96 * rel[:, 1].std(ddof=1) / np.sqrt(seeds_long)),
                ),
            )
        )
        print(
            f"eps={eps:.2f}  short: {rows[-1]['excess_short']}  long: {rows[-1]['excess_long']}",
            flush=True,
        )
    return rows


def crossing(rows, field):
    e = np.array([r["eps"] for r in rows])
    d = np.array([r[field]["cgomsm1"] - r[field]["cgomsm2"] for r in rows])
    for i in range(len(e) - 1):
        if d[i] == 0:
            return float(e[i])
        if d[i] * d[i + 1] < 0:
            return float(e[i] - d[i] * (e[i + 1] - e[i]) / (d[i + 1] - d[i]))
    return float("nan")


def validate_rbpf(m, n=10, seeds=50, n_part=1000):
    """Mean relative gap E(rbpf - exact)^2 / E Var[X|y] on a short horizon."""
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
    res = {}
    for k, p12, ab in EXPS_B:
        m = model_b(p12, ab)
        print(
            f"--- experiment {k}: p12={p12}, set {ab}; rbpf check = {validate_rbpf(path(m, 0.5)):.2e}",
            flush=True,
        )
        rows = run_b(m, eps_grid, n_short, n_long, seeds_short, seeds_long, n_part, seed0=100 * k)
        res[str(k)] = dict(
            p12=p12,
            ab=ab,
            a=AB_SETS[ab]["a"],
            b=AB_SETS[ab]["b"],
            rows=rows,
            eps_star_short=crossing(rows, "excess_short"),
            eps_star_long=crossing(rows, "excess_long"),
        )
        out.mkdir(parents=True, exist_ok=True)
        (out / "exp_b.json").write_text(json.dumps(dict(base=BASE_B, exps=res), indent=1))
    _plot_b(res, out / "fig_b_homotopy.pdf")
    return res


def _plot_b(res, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 7, "font.family": "serif"})
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.0), sharex=True)
    for ax, (k, r) in zip(axes.ravel(), res.items()):
        e = [x["eps"] for x in r["rows"]]
        for key, col, lab in (("cgomsm1", "C0", "CGOMSM1"), ("cgomsm2", "C3", "CGOMSM2")):
            y = np.array([x["excess_long"][key] for x in r["rows"]])
            ci = np.array([x["ci95_long"][key] for x in r["rows"]])
            ax.plot(e, y, color=col, label=lab)
            ax.fill_between(e, y - ci, y + ci, color=col, alpha=0.2, lw=0)
            ax.plot(e, [x["excess_short"][key] for x in r["rows"]], ":", color=col)
        ax.set_title(rf"exp. {k}: $p_{{12}}={r['p12']}$, set {r['ab']}", fontsize=7)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel(r"$\varepsilon$")
    for ax in axes[:, 0]:
        ax.set_ylabel("relative excess MSE")
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- Section V-C: real data
SUB_C = 10  # 100 Hz -> 10 Hz
TEST_EVERY = 4  # every 4th file (sorted order) is held out


def load_vehicle(data_dir: Path, sub: int = SUB_C):
    """List of per-file arrays [yaw (deg/s), a_lat (g), steering (deg), speed (km/h)], subsampled."""
    import pandas as pd

    from prg.experiments import make_vehicle_consigne_fig as mv
    from prg.experiments.make_vehicle_consigne_fig import ALAT, STR, WS, YAW, _ensure_data, _num

    mv.CACHE = Path(data_dir)
    seqs = []
    for f in _ensure_data():
        d = pd.read_parquet(f)
        for c in [YAW, ALAT, STR, *WS]:
            d[c] = _num(d[c])
        d["spd"] = d[WS].mean(axis=1)
        a = d.dropna(subset=[YAW, ALAT, STR, "spd"])[[YAW, ALAT, STR, "spd"]].to_numpy()[::sub]
        if len(a) > 50:
            seqs.append(a)
    return seqs


def _ols(target, design):
    b, *_ = np.linalg.lstsq(design, target, rcond=None)
    return b, float(np.mean((target - design @ b) ** 2))


def _xyu(seq, sc, use_input):
    """Scaled state, observation, and input u_n = (steering_{n+1}, 1) stored at index n + 1."""
    x, y = seq[:, 0] / sc[0], seq[:, 1] / sc[1]
    st = seq[:, 2] / sc[2] if use_input else np.zeros(len(seq))
    return x, y, np.c_[st, np.ones(len(seq))]


def fit_c(train, K, edges, sc, use_input=True):
    """Per-regime ML fits of the THGMSM, CGOMSM1 and CGOMSM2 (regimes labelled by the speed band)."""
    P = np.ones((K, K))
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
        base = {"S1": Z1.T @ Z1 / len(Z1)}  # E[Z_1 Z_1^T | r_1] (centred model)
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


def cgo_filter_c(m, y, u, family):
    """Exact filter of a CGOMSM1 or CGOMSM2 with input; returns E[X_n | y_{1:n}] and log p(y_{1:N})."""
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
    rng = np.random.default_rng(0)
    res = {"K": K, "sub": SUB_C, "main": {}, "grid": []}
    seqs = load_vehicle(data_dir)
    train, test, edges, sc = _split(seqs, K)
    res.update(
        n_train=len(train),
        n_test=len(test),
        n_test_samples=int(sum(len(s) for s in test)),
        speed_edges_kmh=edges.tolist(),
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
        row = {"sub": sub, "K": k}
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
        r" & with $u$ & without $u$ & with $u$ \\",
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 7, "font.family": "serif"})
    tr = res["trace"]
    t = np.array(tr["t"])
    sel = slice(0, min(len(t), 600))  # first 60 s
    fig, ax = plt.subplots(figsize=(3.45, 1.9))
    ax.plot(t[sel], np.array(tr["yaw"])[sel], color="k", lw=1.0, label="true yaw rate")
    ax.plot(t[sel], np.array(tr["cgomsm1"])[sel], color="C0", lw=0.8, label="CGOMSM1 filter")
    ax.plot(
        t[sel], np.array(tr["cgomsm2"])[sel], color="C3", lw=0.8, ls="--", label="CGOMSM2 filter"
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("yaw rate (deg/s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all", choices=("a", "b", "c", "all"))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seeds-short", type=int, default=300)
    ap.add_argument("--seeds-long", type=int, default=100)
    ap.add_argument("--particles", type=int, default=1000)
    ap.add_argument("--n-eps", type=int, default=21)
    ap.add_argument("--data", type=Path, default=Path("data/real/vehicle"))
    a = ap.parse_args()
    if a.which in ("a", "all"):
        for r in exp_a(a.out):
            print(
                f"exp {r['exp']}: tau1={r['tau1']:.4f} (MC {r['mc']['tau1']:.4f})  "
                f"tau2={r['tau2']:.4f} (MC {r['mc']['tau2']:.4f})"
            )
    if a.which in ("b", "all"):
        res = exp_b(a.out, a.seeds_short, a.seeds_long, a.particles, n_eps=a.n_eps)
        for k, r in res.items():
            print(
                f"exp {k}: eps* short = {r['eps_star_short']:.3f}, long = {r['eps_star_long']:.3f}"
            )
    if a.which in ("c", "all"):
        res = exp_c(a.out, a.data)
        for key, v in res["main"].items():
            for name, sc in v.items():
                print(f"{key:8s} {name:12s} RMSE={sc['rmse']:.3f}  loglik={sc['loglik']:.3f}")
        for row in res["grid"]:
            print(row)


if __name__ == "__main__":
    main()
