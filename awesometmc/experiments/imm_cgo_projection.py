#!/usr/bin/env python3
"""Is the IMM a nearest-CGO filter?  (question of W. Pieczynski, 2026-09-29)

Along the CGO -> NGH homotopy of ``homotopy_cgo_ngh.py`` (same models, same
data, same metric), the pairwise IMM fed the TRUE model at epsilon is compared
with exact CGO filters fed the Kullback-Leibler-nearest CGO model, in three
nested ways of being "a CGO":

* ``kl_F``    -- homogeneous CGO (parameters indexed by r_{n+1}), only the pair
                 matrix F projected, Sigma_W kept (the ``cgo_kl`` of the homotopy
                 report);
* ``kl_hom``  -- homogeneous CGO, full KL projection (F AND Sigma_W);
* ``kl_pair`` -- CGO whose parameters depend on the PAIR (r_n, r_{n+1}), as in
                 Abbassi et al. (IEEE TAC 2015), full KL projection.

Full KL projection (closed form).  The true transition of arrival regime k is
N(F_k z, Sigma_W,k), and the departure state z_n ~ S (second moment E[z z^T])
given the departure regime.  Minimising E_z KL(N(F z, Sigma) || N(F' z, Sigma'))
over Sigma' and over F' with C' = 0 gives, with M = Delta Sigma_V^{-1},
L = S_xy S_yy^{-1} and P = S_xx - L S_yx,

    D'  = D + C L,            Sigma_V' = Sigma_V + C P C^T,
    A'  = A - M C,            B'       = B + M C L,
    Delta' = M Sigma_V',      Sigma_U' = Sigma_U + M C P C^T M^T,

i.e. the observation row absorbs the channel by regression on Y, and the state
row keeps its slaving to the observation noise (``M' = M``).  For ``kl_pair``,
S = S_j is the stationary second moment of Z_n given R_n = j (R_{n+1} is
independent of Z_n given R_n); for ``kl_hom`` S is the mixture sum_j w_j S_j,
w_j proportional to pi_j p_jk.  At eps = 0 (C = 0) every projection is the
identity.  At eps = 1 (NGH, L_j = M_j, P_j = Gamma_j under the stationary
law), ``kl_pair`` is EXACTLY the pair-dependent CGO whose filter coincides with
the NGH exact filter:  D'_jk = D_k + C_k M_j, Sigma_V'_jk = Sigma_V,k +
C_k Gamma_j C_k^T, A' = 0, B'_jk = M_k D'_jk.

The exact filter of a CGO with pair-dependent parameters is the CMSHLM
recursion (Pieczynski 2011; Derrode & Pieczynski 2013; Abbassi et al. 2015):

    w(j,k)      = pi_n(j) p_jk N(y_{n+1}; D'_jk y_n, Sigma_V'_jk)
    pi_{n+1}(k) = sum_j w(j,k) / sum_{j,k} w(j,k)
    x_{n+1}(k)  = sum_j w(j,k)/sum_j w(j,k) * [A'_jk x_n(j) + B'_jk y_n
                                               + M'_jk (y_{n+1} - D'_jk y_n)],

initialised with the TRUE initial law (the same first step as the exact filter).

Metric: relative excess MSE over the Bayes filter, E(f - e)^2 / E Var[X|y]
(orthogonality principle, see the homotopy report).

Run:  python -m awesometmc.experiments.imm_cgo_projection [--seeds 200] [--out DIR]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from awesometmc.experiments.homotopy_cgo_ngh import (
    N_STEPS,
    PATHS,
    SEED0,
    model,
    project_kl,
    spectral_radius,
)
from awesometmc.experiments.reference_filters import (
    exact_mixture_filter,
    gpb2_filter,
    imm_filter,
    stationary_moments,
)
from awesometmc.experiments.study import _simulate

KEYS = ("imm", "gpb2", "cgo_frozen", "kl_F", "kl_hom", "kl_pair")


# --------------------------------------------------------------------------
# CGO models with pair-dependent parameters
# --------------------------------------------------------------------------
def _blocks(p, k):
    q = p.q
    F = p.f_matrix.F(k)
    SW = p.noise_cov.Sigma_W(k)
    return F[:q, :q], F[:q, q:], F[q:, :q], F[q:, q:], SW[:q, :q], SW[:q, q:], SW[q:, q:]


def _second_moments(p):
    mu, Sig = stationary_moments(p)
    return [Sig[j] + mu[j] @ mu[j].T for j in range(p.K)]


def kl_cgo(p, S, k):
    """Full KL projection of the arrival-k transition onto CGO kernels, departure moment S."""
    q = p.q
    A, B, C, D, SU, Dl, SV = _blocks(p, k)
    M = Dl @ np.linalg.inv(SV)
    Sxx, Sxy, Syy = S[:q, :q], S[:q, q:], S[q:, q:]
    L = Sxy @ np.linalg.inv(Syy)
    P = Sxx - L @ Sxy.T
    Q = C @ P @ C.T
    return dict(
        A=A - M @ C, B=B + M @ C @ L, D=D + C @ L, SU=SU + M @ Q @ M.T, Dl=M @ (SV + Q), SV=SV + Q
    )


def pair_cgo(p, family):
    """{(j,k): blocks} of the KL-nearest CGO in the given family."""
    K = p.K
    S = _second_moments(p)
    pi, P = np.asarray(p.pi0).ravel(), np.asarray(p.P)
    out = {}
    for k in range(K):
        if family == "pair":
            for j in range(K):
                out[(j, k)] = kl_cgo(p, S[j], k)
        elif family == "hom":
            w = pi * P[:, k]
            w = w / w.sum()
            blk = kl_cgo(p, sum(w[j] * S[j] for j in range(K)), k)
            for j in range(K):
                out[(j, k)] = blk
        else:
            raise ValueError(family)
    return out


def blocks_from_params(p):
    """Homogeneous CGO blocks {(j,k)} read from a GSSParams whose C = 0."""
    out = {}
    for k in range(p.K):
        A, B, C, D, SU, Dl, SV = _blocks(p, k)
        assert np.allclose(C, 0.0), "not a CGO model"
        for j in range(p.K):
            out[(j, k)] = dict(A=A, B=B, D=D, SU=SU, Dl=Dl, SV=SV)
    return out


def _logn(y, m, S):
    r = y - m
    return float(
        -0.5 * ((r.T @ np.linalg.solve(S, r)).item() + np.linalg.slogdet(2 * np.pi * S)[1])
    )


def cgo_filter(blocks, p_true, ys):
    """Exact filter of the (pair-dependent) CGO ``blocks``, initialised with the true initial law."""
    K, q, s = p_true.K, p_true.q, p_true.s
    P = np.asarray(p_true.P)
    pi0 = np.asarray(p_true.pi0).ravel()
    ys = np.asarray(ys, dtype=float).reshape(-1, s)
    N = ys.shape[0]
    Ex = np.zeros((N, q))
    Pi = np.zeros((N, K))
    # first step: true initial law of Z_1 given r_1
    y1 = ys[0].reshape(s, 1)
    logw = np.zeros(K)
    xh = []
    for r in range(K):
        mu, Sig = p_true.mu_z0(r).reshape(q + s, 1), p_true.Sigma_z0(r)
        L = Sig[:q, q:] @ np.linalg.inv(Sig[q:, q:])
        xh.append(mu[:q] + L @ (y1 - mu[q:]))
        logw[r] = np.log(pi0[r] + 1e-300) + _logn(y1, mu[q:], Sig[q:, q:])
    pi = np.exp(logw - logw.max())
    pi /= pi.sum()
    Ex[0] = sum(pi[r] * xh[r] for r in range(K)).ravel()
    Pi[0] = pi
    for n in range(1, N):
        yp, yn = ys[n - 1].reshape(s, 1), ys[n].reshape(s, 1)
        lw = np.full((K, K), -np.inf)
        for j, k in itertools.product(range(K), range(K)):
            b = blocks[(j, k)]
            lw[j, k] = (
                np.log(pi[j] + 1e-300) + np.log(P[j, k] + 1e-300) + _logn(yn, b["D"] @ yp, b["SV"])
            )
        w = np.exp(lw - lw.max())
        pin = w.sum(axis=0)
        new_x = []
        for k in range(K):
            acc = np.zeros((q, 1))
            for j in range(K):
                b = blocks[(j, k)]
                Mjk = b["Dl"] @ np.linalg.inv(b["SV"])
                acc += w[j, k] * (b["A"] @ xh[j] + b["B"] @ yp + Mjk @ (yn - b["D"] @ yp))
            new_x.append(acc / pin[k])
        pi = pin / pin.sum()
        xh = new_x
        Ex[n] = sum(pi[k] * xh[k] for k in range(K)).ravel()
        Pi[n] = pi
    return Ex, Pi


# --------------------------------------------------------------------------
# Filter-optimal CGO (scalar case): parameters tuned on training sequences
# --------------------------------------------------------------------------
# Only (A', B', D', Sigma_V', M') enter the CGO filter mean; Sigma_U' does not.
_NPAR = 5


def _theta_from_blocks(blocks, K):
    th = []
    for j, k in itertools.product(range(K), range(K)):
        b = blocks[(j, k)]
        sv = float(np.asarray(b["SV"]).ravel()[0])
        m = float(np.asarray(b["Dl"]).ravel()[0]) / sv
        th += [float(np.asarray(b[n]).ravel()[0]) for n in ("A", "B", "D")] + [np.log(sv), m]
    return np.array(th)


def _cgo_filter_batch(theta, K, P, init, Y, pair=True):
    """Vectorised scalar CGO filter over a batch Y (n_seq, N); theta per pair (or per arrival)."""
    th = theta.reshape(-1, _NPAR)
    idx = (lambda j, k: j * K + k) if pair else (lambda j, k: k)
    A = np.array([[th[idx(j, k), 0] for k in range(K)] for j in range(K)])
    B = np.array([[th[idx(j, k), 1] for k in range(K)] for j in range(K)])
    D = np.array([[th[idx(j, k), 2] for k in range(K)] for j in range(K)])
    SV = np.exp(np.array([[th[idx(j, k), 3] for k in range(K)] for j in range(K)]))
    Mg = np.array([[th[idx(j, k), 4] for k in range(K)] for j in range(K)])
    pi0, L0, Syy0 = init
    ns, N = Y.shape
    lw = np.log(pi0)[None, :] - 0.5 * (
        Y[:, :1] ** 2 / Syy0[None, :] + np.log(2 * np.pi * Syy0)[None, :]
    )
    pi = np.exp(lw - lw.max(axis=1, keepdims=True))
    pi /= pi.sum(axis=1, keepdims=True)
    xh = Y[:, :1] * L0[None, :]  # (ns, K)
    out = np.zeros((ns, N))
    out[:, 0] = np.sum(pi * xh, axis=1)
    logP = np.log(P)
    for n in range(1, N):
        yp, yn = Y[:, n - 1][:, None, None], Y[:, n][:, None, None]
        res = yn - D[None] * yp  # (ns, K, K)
        lw = (
            np.log(pi + 1e-300)[:, :, None]
            + logP[None]
            - 0.5 * (res**2 / SV[None] + np.log(2 * np.pi * SV)[None])
        )
        w = np.exp(lw - lw.max(axis=(1, 2), keepdims=True))
        pin = w.sum(axis=1)  # (ns, K)
        cand = A[None] * xh[:, :, None] + B[None] * yp + Mg[None] * res
        xh = np.sum(w * cand, axis=1) / pin
        pi = pin / pin.sum(axis=1, keepdims=True)
        out[:, n] = np.sum(pi * xh, axis=1)
    return out


def optimise_cgo(p, start_blocks, Ytr, Etr, pair=True):
    """CGO parameters minimising sum (f - e)^2 on training sequences (start: KL projection)."""
    from scipy.optimize import minimize

    K = p.K
    assert p.q == 1 and p.s == 1, "the tuned CGO filter is written for the scalar case"
    assert all(np.allclose(p.mu_z0(r), 0.0) for r in range(K)), (
        "_cgo_filter_batch assumes a centred initial law"
    )
    P = np.asarray(p.P)
    pi0 = np.asarray(p.pi0).ravel()
    init = (
        pi0,
        np.array([p.Sigma_z0(r)[0, 1] / p.Sigma_z0(r)[1, 1] for r in range(K)]),
        np.array([p.Sigma_z0(r)[1, 1] for r in range(K)]),
    )
    th0 = _theta_from_blocks(start_blocks, K)
    if not pair:
        th0 = th0.reshape(K, K, _NPAR)[0].ravel()  # one block per arrival regime
    f = lambda th: float(np.mean((_cgo_filter_batch(th, K, P, init, Ytr, pair) - Etr) ** 2))
    r = minimize(f, th0, method="L-BFGS-B", options={"maxiter": 500})
    return r.x, init, float(r.fun)


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------
TRAIN_SEED0 = 90000


def run_eps(eps, p0, n_seeds, kind="straight", optimise=False, n_train=200):
    p = model(eps, kind)
    cgo = {
        "kl_F": blocks_from_params(project_kl(eps, kind, "cgo")),
        "kl_hom": pair_cgo(p, "hom"),
        "kl_pair": pair_cgo(p, "pair"),
    }
    opt = {}
    if optimise:
        Ytr, Etr = [], []
        for sd in range(n_train):
            _, _, ys = _simulate(p, N_STEPS, seed=TRAIN_SEED0 + sd)
            Ytr.append(ys[:, 0])
            Etr.append(exact_mixture_filter(p, ys)[0][:, 0])
        Ytr, Etr = np.array(Ytr), np.array(Etr)
        for name, pair, start in (
            ("opt_hom", False, cgo["kl_hom"]),
            ("opt_pair", True, cgo["kl_pair"]),
        ):
            opt[name] = optimise_cgo(p, start, Ytr, Etr, pair) + (pair,)
    keys = KEYS + tuple(opt)
    d2 = {k: 0.0 for k in keys}
    dpi = {k: 0.0 for k in keys}
    var_e = 0.0
    n_tot = 0
    for sd in range(n_seeds):
        _, xs, ys = _simulate(p, N_STEPS, seed=SEED0 + sd)
        ex_e, var_x, pi_e = exact_mixture_filter(p, ys)
        var_e += float(np.sum(np.asarray(var_x).reshape(len(ys), -1)[:, 0]))
        est = {}
        e, _, pi_i, _ = imm_filter(p, ys)
        est["imm"] = (e, pi_i)
        e, _, pi_g = gpb2_filter(p, ys)[:3]
        est["gpb2"] = (e, pi_g)
        e, _, pi_c, _ = imm_filter(p0, ys)
        est["cgo_frozen"] = (e, pi_c)
        for key, blk in cgo.items():
            est[key] = cgo_filter(blk, p, ys)
        for key, (th, init, _, pair) in opt.items():
            e = _cgo_filter_batch(th, p.K, np.asarray(p.P), init, ys[:, 0][None, :], pair)[0]
            est[key] = (e[:, None], None)
        for key, (e, pi) in est.items():
            d2[key] += float(np.sum((e[:, 0] - ex_e[:, 0]) ** 2))
            if pi is not None:
                dpi[key] += float(np.sum(np.abs(np.asarray(pi)[:, 0] - pi_e[:, 0])))
        n_tot += len(ys)
    post_var = var_e / n_tot
    return {
        "eps": eps,
        "path": kind,
        "rho": spectral_radius(eps, kind),
        "posterior_var": post_var,
        "excess": {k: (d2[k] / n_tot) / post_var for k in keys},
        "regime_gap": {k: dpi[k] / n_tot for k in keys},
        "opt_params": {k: v[0].tolist() for k, v in opt.items()},
        "kl_pair_blocks": {
            f"{j}{k}": {n: float(np.asarray(v).ravel()[0]) for n, v in b.items()}
            for (j, k), b in cgo["kl_pair"].items()
        },
    }


def self_checks():
    """Endpoint identities of the projections (printed, asserted)."""
    for kind in PATHS:
        p = model(0.0, kind)
        blk = pair_cgo(p, "pair")
        for (j, k), b in blk.items():
            A, B, C, D, SU, Dl, SV = _blocks(p, k)
            assert np.allclose(
                [b["A"], b["B"], b["D"], b["SU"], b["Dl"], b["SV"]], [A, B, D, SU, Dl, SV]
            ), "eps=0"
    p = model(1.0, "straight")
    S = _second_moments(p)
    for (j, k), b in pair_cgo(p, "pair").items():
        A, B, C, D, SU, Dl, SV = _blocks(p, k)
        Mj = _blocks(p, j)[5] @ np.linalg.inv(_blocks(p, j)[6])
        Gj = _blocks(p, j)[4] - Mj @ _blocks(p, j)[5].T
        assert np.allclose(b["D"], D + C @ Mj) and np.allclose(b["SV"], SV + C @ Gj @ C.T), (
            "eps=1 (D, SV)"
        )
        assert np.allclose(b["A"], 0.0) and np.allclose(
            b["B"], (Dl @ np.linalg.inv(SV)) @ b["D"]
        ), "eps=1 (A, B)"
    print("self-checks OK (identity at eps=0 on all paths; NGH-equivalent pair CGO at eps=1)")


STYLE = {  # key: (label, colour, style)
    "cgo_frozen": (r"CGO, gelé en $\varepsilon=0$", "#9aa7b3", "o-"),
    "kl_F": ("CGO homogène, KL sur $F$ seul", "#6b8fb3", "v:"),
    "kl_hom": ("CGO homogène, KL complet", "#1f5f8b", "^:"),
    "kl_pair": ("CGO apparié, KL complet", "#0b3050", "D:"),
    "opt_hom": ("CGO homogène, optimisé filtrage", "#b8451f", "^-"),
    "opt_pair": ("CGO apparié, optimisé filtrage", "#7a2410", "D-"),
    "imm": ("IMM, vrai modèle", "#8a6d1f", "s--"),
    "gpb2": ("GPB2, vrai modèle", "#5f7d3a", "o--"),
}
TITLE = {
    "straight": "(a) chemin droit",
    "c_first": "(b) canal d'abord",
    "ab_first": "(c) asservissement d'abord",
}


def plot(outdir: Path, paths=PATHS) -> None:
    """One log-scale panel per path: relative excess MSE of every filter."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(paths), figsize=(3.9 * len(paths), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    floor = 1e-12
    for ax, kind in zip(axes, paths):
        rows = json.loads((outdir / f"results_{kind}.json").read_text())
        eps = [r["eps"] for r in rows]
        for key, (lab, col, st) in STYLE.items():
            if key not in rows[0]["excess"]:
                continue
            ax.semilogy(
                eps,
                [max(r["excess"][key], floor / 3) for r in rows],
                st,
                ms=3.2,
                lw=1.1,
                color=col,
                label=lab,
            )
        ax.axhspan(1e-13, floor, color="0.88", zorder=0)
        ax.set_ylim(1e-13, 1.0)
        if kind != "straight":
            ax.axvline(0.5, color="gray", lw=0.8, alpha=0.6)
        ax.set_title(TITLE.get(kind, kind), fontsize=10)
        ax.set_xlabel(r"$\varepsilon$  (0 : CGO, 1 : NGH)")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel(r"excès relatif  $\mathbb{E}(f-e)^2/\mathbb{E}\,\mathrm{Var}[X\mid y]$")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    out = outdir / "figures" / "imm_cgo.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=160)
    print(f"figure -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--grid", type=int, default=21)
    ap.add_argument("--paths", nargs="+", default=list(PATHS))
    ap.add_argument("--out", type=Path, default=Path("docs/rapport_imm_cgo"))
    ap.add_argument(
        "--optimise", action="store_true", help="also tune CGO parameters for filtering (slow)"
    )
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if a.plot_only:
        plot(a.out, a.paths)
        return
    self_checks()
    a.out.mkdir(parents=True, exist_ok=True)
    p0 = model(0.0)
    for kind in a.paths:
        rows = []
        for eps in np.linspace(0.0, 1.0, a.grid):
            r = run_eps(float(eps), p0, a.seeds, kind, optimise=a.optimise)
            rows.append(r)
            print(
                f"[{kind}] eps={eps:.2f} "
                + " ".join(f"{k}={v:.2e}" for k, v in r["excess"].items()),
                flush=True,
            )
        (a.out / f"results_{kind}.json").write_text(json.dumps(rows, indent=1))
    plot(a.out, a.paths)


if __name__ == "__main__":
    main()
