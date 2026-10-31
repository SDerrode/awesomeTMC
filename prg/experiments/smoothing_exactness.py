"""Machine-precision checks of the exact fixed-interval smoothers (smoothing paper).

Thirteen experiments back the numbers quoted in the exact-smoothing paper
(docs/exact-smoothing):

S1  Smoothed regime posterior, exact and O(N K^2) on both families: the HMM
    forward-backward on (R,Y) -- kernel Q0 on {C==0}, kernel Q on AB --
    matches the exact smoothed regime posterior to ~1e-15.
S2  {C==0}: the reweighting smoother (pairwise-IMM filtered per-regime
    summaries recombined by the smoothed regime posterior) matches the exact
    smoothed mean AND variance to ~1e-15; the same formula under AB fails at
    O(1) -- the two branches need different smoothers.
S3  AB: the constant-gain smoother (affine read-out + one-step xi correction,
    precomputed gains W_jk, pair-smoothed weights) matches the exact smoothed
    mean and variance to ~1e-15, with no covariance recursion of any kind.
S4  The trap: the naive smoothed read-out sum_k gamma_n(k)(M_k y_n + c_k),
    even fed the EXACT smoothed regime posterior, misses the smoothed mean at
    the order of the entire smoothing gain |filtered - smoothed|.
S5  {A==MC} \\ AB: the lag-1 smoother (read-out M_j y_n + E_j y_{n-1} + c_j,
    same gains W_jk, forward-backward on the lag-1 (R,(Y_{n-1},Y_n)) chain)
    stays exact, mean and variance; on AB it reduces to S3.
S6  Off the union {C==0} u {A==MC}, the collapsed smoothed regime posterior
    (GPB2 forward + pair-likelihood backward) degrades at FIRST order in the
    violation eta (slope ~1.15) -- against third order for the filtered
    quantities (E3 of cns_exactness): smoothing pays structural
    misspecification at first order.  S6b extends the measurement to the
    DUAL approach direction (C = eta at both regimes, full state memory:
    toward the autonomy branch) and to the smoothed STATE MEAN (pair-RTS,
    reweighting): the first order is direction-independent for smoothing,
    whereas the filtered exponent depends on the approach direction.
S7  The pair-RTS form and its gain degeneracy: the mode-matched RTS smoother
    run on the pair (GPB2 forward bank, backward mixing over the arrival
    regime with smoothed pair weights) has gains G^{jk} whose x-rows satisfy,
    identically, G_x == 0 and G_y == W_jk on {A==MC} (the backward state
    input is multiplied by zero: RTS collapses to the one-step correction),
    and G_y == -G_x M_k on {C==0} (the correction against pair-conditional
    backward means annihilates: RTS collapses to the reweighting).  The
    naive mode-matched RTS -- which feeds the k-conditional backward mean
    where the pair-conditional one is required -- is therefore exact on the
    whole slaved branch but BIASED on {C==0} with state memory.
S8  Off the union, the three formulations (pair-RTS, one-step correction,
    reweighting) genuinely separate; their pairwise separations vanish with
    the violation eta and are computable online (no ground truth): a
    measurable diagnostic of the structural violation.
S9  The HBH-structure two-filter smoother (forward IMM + backward-time IMM
    on the inverted pair dynamics, pseudo-diffuse init, Fraser-Potter
    per-mode fusion), transcribed to the pair with exact observation.
    Structural finding first: on AB the pair dynamics is SINGULAR
    (det F_r = 0), so the backward-time filter -- hence the HBH scheme --
    is inapplicable on the very family where the one-step smoother is
    exact.  On {C==0} and on {A==MC}\\AB (where F_r is invertible), the
    measurement quantifies how far the scheme is from realizing the exact
    smoother: its backward pass substitutes a normalizable posterior for
    the non-normalizable likelihood (a known approximation, cf. Li-Liu-
    Yang-Mihaylova-Deng, FUSION 2021) and mixes modes before evaluating.
S10 Mixed families: on the E5 gauge of cns_exactness (regime 1: C=0 with
    memory A0; regime 2: C1!=0 slaved, B broken), every collapsed smoother
    is measurably biased as soon as A0 != 0, and the biases exceed the
    GPB2 FILTERED bias on the same gauge by orders of magnitude -- the
    filtering/smoothing asymmetry at a structural (non-small) violation.
    At A0 = 0 the model re-enters the uniform family {A==MC} (both
    conditions hold at regime 1) and the lag-1 smoother returns to
    machine precision: smoothing inherits the uniformity requirement.
S11 The matrix case (E10 gauge of cns_exactness, q=2): on the cross-
    annihilation family C_k (A_j - M_j C_j) = 0 outside the union -- where
    the GPB2 FILTER is exact -- the collapsed smoothed regime posterior
    (GPB2 forward + pair-likelihood backward) is exact too, but no collapsed
    STATE smoother of this paper is (reweighting, lag-1, pair-RTS all
    biased): the smoothed state genuinely needs the history that the
    filtered state does not.  For q>1 the smoothing state domains covered
    here ({C==0} u {A==MC}) are strictly smaller than the GPB2 filtering
    domain.
S12 Time reversal (random models, K=2, (q,s) in {(1,1),(2,1),(2,2)}): the
    two exact smoothers are mirror filters.  On CGOMSM2 (AB) the per-regime
    smoothed mean ignores the past and the pair-conditional one ignores
    everything beyond (y_n, y_{n+1}) -- the smoother is a reweighted one-step
    BACKWARD filter, equal to the constant-gain formula; on CGOMSM1 (C==0)
    the per-regime smoothed mean ignores the future (reweighted FORWARD
    filter), and the mean given (r_{n-1}, r_n) is one-step iff A==0, the
    exact mirror of AB.  Reversed transitions: the reversal of an AB model
    has C~ = A~ = 0 and a regime transition free of the hidden state; the
    reversal of a C==0 model is AB iff A==0, its regime transition involves
    the hidden state.  The moment-matched Gaussian reversal is exact for K=1
    only (mixture caveat).  The [Abb15] pair-dependent CGOMSM1 sharing the
    (R,Y) law and the filter of an AB model has the naive read-out as its
    smoother.  The stationary K=1 example with state memory realizes the
    mirror of memory for its stationary law only.

S13 Classical collapsed smoothers (exactness domains, the analogue for the
    smoothers of the IMM/GPB2 theorems of the exactness paper).  A collapsed
    RTS smoother is fixed by its forward bank (GPB2 = Kim filter, or IMM)
    and its backward regime weights (Kim's p(r_n | r_{n+1}, y_{1:n}), or the
    pair-likelihood recursion).  Kim's smoother and the IMM-RTS (Kim
    weights) are exact on {C==0, A==0} only -- the intersection of the two
    branches -- with exact regimes on all of {C==0} (the kernel ignores the
    departure regime) but a biased state as soon as A != 0, equal dynamics
    included; the IMM pair-RTS likewise; the GPB2 pair-RTS is exact on
    {A==MC} (S7), and HBH nowhere (inapplicable where F_r is singular).
    Random matrix models (K <= 3, q = 2, s <= 2) of each family confirm the
    map.  The K^2 embedding of an AB model (Section V-E of the exactness
    paper) transports the exact filter and the smoothed regimes to machine
    precision, and Kim / IMM-RTS with K^2 modes realize the exact smoother of
    the embedded model -- which misses the CGOMSM2 smoother by the order of
    the whole smoothing gain: the K^2 device that makes the IMM exact for
    filtering does not carry over to smoothing.  The bound is reachable,
    though: dropping the backward state recursion (one-step retrodiction of
    the filtered pair summary by y_{n+1}, pair-smoothed weights) gives a
    collapsed smoother exact -- mean, variance and regimes -- on the whole
    GPB2 domain with the GPB2 bank (cross-annihilation family, q > 1
    included, where S11 found no state smoother) and on {C==0} with the IMM
    bank: the domain of its forward filter, the terminal-point bound.

Ground truth is the exact K^N path smoother: one conditional pair Kalman
filter + RTS smoother per regime path, the paths weighted by their exact
posterior p(path | y_{1:N}).  Errors are normalized sup-norm deviations,
reported as median and max over seeds.  S12 uses its own exact path
enumeration for pair-dependent models (transition indexed by (r_n, r_{n+1})),
which also returns the filtered and the regime-conditional summaries.

Run:  python -m prg.experiments.smoothing_exactness
"""

from __future__ import annotations

from itertools import product

import numpy as np

from prg.classes.FMatrix import FMatrix
from prg.classes.GSSParams import GSSParams
from prg.classes.NoiseCovariance import GSSNoiseCovariance
from prg.experiments.cns_exactness import (
    N_STEPS,
    SEED0,
    _params,
    _rel,
    ab_model,
    cgo_memory_model,
    cross_annihilation_model,
    mixed_branch_model,
    off_union_model,
)
from prg.experiments.reference_filters import (
    _kalman_exact_y_update,
    _obs_matrix,
    exact_mixture_filter,
    gpb2_filter,
)
from prg.experiments.study import _simulate, with_stationary_init

N_SEEDS = 15  # exact path smoother costs K^N per seed -- keep moderate


# ---------------------------------------------------------------------------
# ground truth: exact K^N path smoother (mean, variance, regime posterior)
# ---------------------------------------------------------------------------
def exact_path_smoother(params, ys):
    """Exact fixed-interval smoother by full path enumeration.

    One conditional pair Kalman filter + RTS smoother per regime path,
    weighted by p(path | y_{1:N}).  Returns the smoothed state mean (N, q),
    state covariance (N, q, q) and regime posterior (N, K).
    """
    K, q, s = params.K, params.q, params.s
    dim = q + s
    ys = np.asarray(ys, float).reshape(-1, s)
    N = ys.shape[0]
    H = _obs_matrix(q, s)
    F = [params.f_matrix.F(k) for k in range(K)]
    b = [params.b(k).reshape(dim, 1) for k in range(K)]
    SW = [params.noise_cov.Sigma_W(k) for k in range(K)]
    mu0 = [params.mu_z0(k).reshape(dim, 1) for k in range(K)]
    S0 = [params.Sigma_z0(k) for k in range(K)]
    logws, store = [], []
    for path in product(range(K), repeat=N):
        k0 = path[0]
        z, P, ll = _kalman_exact_y_update(mu0[k0], S0[k0], ys[0].reshape(s, 1), H)
        lw = float(np.log(params.pi0[k0] + 1e-300) + ll)
        zf, Pf, zp, Pp = [z], [P], [None], [None]
        for n in range(1, N):
            k = path[n]
            zpr = F[k] @ zf[-1] + b[k]
            Ppr = F[k] @ Pf[-1] @ F[k].T + SW[k]
            z, P, ll = _kalman_exact_y_update(zpr, Ppr, ys[n].reshape(s, 1), H)
            lw += float(np.log(params.P[path[n - 1], k] + 1e-300)) + ll
            zf.append(z)
            Pf.append(P)
            zp.append(zpr)
            Pp.append(Ppr)
        # RTS backward pass along the (conditionally linear-Gaussian) pair
        zs = [None] * N
        Ps_ = [None] * N
        zs[N - 1], Ps_[N - 1] = zf[N - 1], Pf[N - 1]
        for n in range(N - 2, -1, -1):
            Fk = F[path[n + 1]]
            G = Pf[n] @ Fk.T @ np.linalg.inv(Pp[n + 1])
            zs[n] = zf[n] + G @ (zs[n + 1] - zp[n + 1])
            Ps_[n] = Pf[n] + G @ (Ps_[n + 1] - Pp[n + 1]) @ G.T
        logws.append(lw)
        store.append((path, zs, Ps_))
    logws = np.array(logws)
    w = np.exp(logws - logws.max())
    w /= w.sum()
    Ex = np.zeros((N, q))
    Vx = np.zeros((N, q, q))
    gam = np.zeros((N, K))
    for wi, (path, zs, _) in zip(w, store):
        for n in range(N):
            Ex[n] += wi * zs[n][:q, 0]
            gam[n, path[n]] += wi
    for wi, (_, zs, Ps_) in zip(w, store):
        for n in range(N):
            d = zs[n][:q, 0] - Ex[n]
            Vx[n] += wi * (Ps_[n][:q, :q] + np.outer(d, d))
    return Ex, Vx, gam


# ---------------------------------------------------------------------------
# shared machinery: slaving objects, kernels, forward-backward on (R,Y)
# ---------------------------------------------------------------------------
def _slaving(params):
    """Per-regime slaving objects (M_k, Gamma_k, c_k) from the blocks."""
    K, q, s = params.K, params.q, params.s
    M, Gam, c = [], [], []
    for k in range(K):
        SV = params.noise_cov.Sigma_V(k)
        Mk = params.noise_cov.Delta(k) @ np.linalg.inv(SV)
        bk = params.b(k).reshape(q + s, 1)
        M.append(Mk)
        Gam.append(params.noise_cov.Sigma_U(k) - Mk @ SV @ Mk.T)
        c.append(bk[:q] - Mk @ bk[q:])
    return M, Gam, c


def _gauss(x, m, S):
    r = np.atleast_1d(np.asarray(x, float).ravel() - np.asarray(m).ravel())
    S = np.atleast_2d(S)
    return float(np.exp(-0.5 * r @ np.linalg.solve(S, r)) / np.sqrt(np.linalg.det(2 * np.pi * S)))


def _alpha1(params, ys):
    """Initial regime posterior p(r_1 | y_1) from the per-regime laws of Y_1."""
    K, q, s = params.K, params.q, params.s
    H = _obs_matrix(q, s)
    y1 = np.asarray(ys, float).reshape(-1, s)[0]
    la = np.zeros(K)
    for k in range(K):
        mY = (H @ params.mu_z0(k).reshape(q + s, 1)).ravel()
        SY = H @ params.Sigma_z0(k) @ H.T
        r = y1 - mY
        la[k] = np.log(params.pi0[k] + 1e-300) - 0.5 * (
            r @ np.linalg.solve(SY, r) + np.log(np.linalg.det(2 * np.pi * SY))
        )
    a = np.exp(la - la.max())
    return a / a.sum()


def _forward_backward(alpha1, Ker):
    """HMM forward-backward for a time-varying kernel Ker (N-1, K, K).

    Returns the smoothed regime marginals gam (N, K) and the smoothed pair
    marginals gampair (N-1, K, K).
    """
    N, K = Ker.shape[0] + 1, alpha1.shape[0]
    alpha = np.zeros((N, K))
    alpha[0] = alpha1
    for n in range(N - 1):
        alpha[n + 1] = alpha[n] @ Ker[n]
        alpha[n + 1] /= alpha[n + 1].sum()
    beta = np.zeros((N, K))
    beta[N - 1] = 1.0
    for n in range(N - 2, -1, -1):
        beta[n] = Ker[n] @ beta[n + 1]
        beta[n] /= beta[n].max()
    gam = alpha * beta
    gam /= gam.sum(axis=1, keepdims=True)
    gampair = np.zeros((N - 1, K, K))
    for n in range(N - 1):
        x = (alpha[n][:, None] * Ker[n]) * beta[n + 1][None, :]
        gampair[n] = x / x.sum()
    return gam, gampair


def kernel_Q0(params, ys):
    """{C==0} kernel Q0(j->k) = p_jk N(y_{n+1}; D_k y_n + bY_k, SV_k)."""
    K, q, s = params.K, params.q, params.s
    ys = np.asarray(ys, float).reshape(-1, s)
    N = ys.shape[0]
    D = [params.f_matrix.D(k) for k in range(K)]
    SV = [params.noise_cov.Sigma_V(k) for k in range(K)]
    bY = [params.b(k).reshape(q + s, 1)[q:] for k in range(K)]
    Ker = np.zeros((N - 1, K, K))
    for n in range(N - 1):
        for k in range(K):
            m = D[k] @ ys[n].reshape(s, 1) + bY[k]
            g = _gauss(ys[n + 1], m, SV[k])
            for j in range(K):
                Ker[n, j, k] = params.P[j, k] * g
    return Ker


def kernel_Qab(params, ys):
    """AB kernel Q(j->k) with mean (C_k M_j + D_k) y_n + C_k c_j + bY_k and
    covariance S_jk = C_k Gam_j C_k^T + SV_k (eq. (kernAB) of the paper)."""
    K, q, s = params.K, params.q, params.s
    ys = np.asarray(ys, float).reshape(-1, s)
    N = ys.shape[0]
    M, Gam, c = _slaving(params)
    C = [params.f_matrix.C(k) for k in range(K)]
    D = [params.f_matrix.D(k) for k in range(K)]
    SV = [params.noise_cov.Sigma_V(k) for k in range(K)]
    bY = [params.b(k).reshape(q + s, 1)[q:] for k in range(K)]
    Ker = np.zeros((N - 1, K, K))
    for n in range(N - 1):
        y0 = ys[n].reshape(s, 1)
        for j in range(K):
            for k in range(K):
                m = (C[k] @ M[j] + D[k]) @ y0 + C[k] @ c[j] + bY[k]
                S = C[k] @ Gam[j] @ C[k].T + SV[k]
                Ker[n, j, k] = params.P[j, k] * _gauss(ys[n + 1], m, S)
    return Ker


# ---------------------------------------------------------------------------
# the three smoothers
# ---------------------------------------------------------------------------
def pairwise_imm_bank(params, ys):
    """Pairwise IMM keeping the per-regime filtered summaries (x-block).

    Returns xs (N, K, q) and Pxs (N, K, q, q) -- exact per-regime filtered
    moments on {C==0} by Theorem 1 of the exactness paper.
    """
    K, q, s = params.K, params.q, params.s
    dim = q + s
    ys = np.asarray(ys, float).reshape(-1, s)
    N = ys.shape[0]
    H = _obs_matrix(q, s)
    F = [params.f_matrix.F(k) for k in range(K)]
    b = [params.b(k).reshape(dim, 1) for k in range(K)]
    SW = [params.noise_cov.Sigma_W(k) for k in range(K)]
    zs, Ps, la = [], [], np.zeros(K)
    for k in range(K):
        z, P, ll = _kalman_exact_y_update(
            params.mu_z0(k).reshape(dim, 1), params.Sigma_z0(k), ys[0].reshape(s, 1), H
        )
        zs.append(z)
        Ps.append(P)
        la[k] = np.log(params.pi0[k] + 1e-300) + ll
    al = np.exp(la - la.max())
    al /= al.sum()
    xs = np.zeros((N, K, q))
    Pxs = np.zeros((N, K, q, q))
    for k in range(K):
        xs[0, k] = zs[k][:q, 0]
        Pxs[0, k] = Ps[k][:q, :q]
    for n in range(1, N):
        y = ys[n].reshape(s, 1)
        nzs, nPs, nla = [], [], np.zeros(K)
        for k in range(K):
            w = np.array([params.P[j, k] * al[j] for j in range(K)])
            cbar = w.sum()
            w = w / cbar
            z0 = sum(w[j] * zs[j] for j in range(K))
            P0 = sum(w[j] * (Ps[j] + (zs[j] - z0) @ (zs[j] - z0).T) for j in range(K))
            zpr = F[k] @ z0 + b[k]
            Ppr = F[k] @ P0 @ F[k].T + SW[k]
            z, P, ll = _kalman_exact_y_update(zpr, Ppr, y, H)
            nzs.append(z)
            nPs.append(P)
            nla[k] = np.log(cbar + 1e-300) + ll
        zs, Ps = nzs, nPs
        al = np.exp(nla - nla.max())
        al /= al.sum()
        for k in range(K):
            xs[n, k] = zs[k][:q, 0]
            Pxs[n, k] = Ps[k][:q, :q]
    return xs, Pxs


def reweight_smoother(params, ys, Ker):
    """{C==0} smoother: smoothed-regime reweighting of the FILTERED per-regime
    summaries (Theorem 1 of the smoothing paper).  Returns (Ex, Vx)."""
    xs, Pxs = pairwise_imm_bank(params, ys)
    gam, _ = _forward_backward(_alpha1(params, ys), Ker)
    N, K, q = xs.shape
    Ex = np.einsum("nk,nkq->nq", gam, xs)
    Vx = np.zeros((N, q, q))
    for n in range(N):
        for k in range(K):
            d = xs[n, k] - Ex[n]
            Vx[n] += gam[n, k] * (Pxs[n, k] + np.outer(d, d))
    return Ex, Vx


def cg_smoother(params, ys):
    """AB constant-gain smoother (Theorem 2 of the smoothing paper).

    X_{n|N} = sum_{j,k} gampair_n(j,k) [ M_j y_n + c_j
              + W_jk (y_{n+1} - m_jk(y_n)) ],   n < N,
    with precomputed gains W_jk = Gam_j C_k^T S_jk^{-1}, per-pair conditional
    covariance Gam_j - W_jk C_k Gam_j, and the filtered read-out at n = N.
    Returns (Ex, Vx); cost O(N K^2), no covariance recursion.
    """
    K, q, s = params.K, params.q, params.s
    ys_ = np.asarray(ys, float).reshape(-1, s)
    N = ys_.shape[0]
    M, Gam, c = _slaving(params)
    C = [params.f_matrix.C(k) for k in range(K)]
    D = [params.f_matrix.D(k) for k in range(K)]
    SV = [params.noise_cov.Sigma_V(k) for k in range(K)]
    bY = [params.b(k).reshape(q + s, 1)[q:] for k in range(K)]
    W = [
        [Gam[j] @ C[k].T @ np.linalg.inv(C[k] @ Gam[j] @ C[k].T + SV[k]) for k in range(K)]
        for j in range(K)
    ]
    Vpair = [[Gam[j] - W[j][k] @ C[k] @ Gam[j] for k in range(K)] for j in range(K)]
    gam, gampair = _forward_backward(_alpha1(params, ys_), kernel_Qab(params, ys_))
    Ex = np.zeros((N, q))
    Vx = np.zeros((N, q, q))
    for n in range(N - 1):
        y0 = ys_[n].reshape(s, 1)
        y1 = ys_[n + 1].reshape(s, 1)
        means = np.zeros((K, K, q))
        for j in range(K):
            for k in range(K):
                m_jk = (C[k] @ M[j] + D[k]) @ y0 + C[k] @ c[j] + bY[k]
                means[j, k] = (M[j] @ y0 + c[j] + W[j][k] @ (y1 - m_jk)).ravel()
                Ex[n] += gampair[n, j, k] * means[j, k]
        for j in range(K):
            for k in range(K):
                d = means[j, k] - Ex[n]
                Vx[n] += gampair[n, j, k] * (Vpair[j][k] + np.outer(d, d))
    yN = ys_[N - 1].reshape(s, 1)
    mN = np.array([(M[j] @ yN + c[j]).ravel() for j in range(K)])
    Ex[N - 1] = gam[N - 1] @ mN
    for j in range(K):
        d = mN[j] - Ex[N - 1]
        Vx[N - 1] += gam[N - 1, j] * (Gam[j] + np.outer(d, d))
    return Ex, Vx


def lag1_smoother(params, ys):
    """{A==MC} lag-1 smoother (Theorem 3 of the smoothing paper).

    Read-out M_j y_n + E_j y_{n-1} + c_j (E_j = B_j - M_j D_j), same gains
    W_jk, forward-backward on the lag-1 (R, (Y_{n-1}, Y_n)) chain; at n = 1
    the read-out comes from the initial law.  Reduces to cg_smoother on AB.
    Returns (Ex, Vx).
    """
    K, q, s = params.K, params.q, params.s
    ys_ = np.asarray(ys, float).reshape(-1, s)
    N = ys_.shape[0]
    dim = q + s
    M, Gam, c = _slaving(params)
    B = [params.f_matrix.B(k) for k in range(K)]
    C = [params.f_matrix.C(k) for k in range(K)]
    D = [params.f_matrix.D(k) for k in range(K)]
    SV = [params.noise_cov.Sigma_V(k) for k in range(K)]
    bY = [params.b(k).reshape(q + s, 1)[q:] for k in range(K)]
    H = _obs_matrix(q, s)

    # per-(n, j) read-out mean/cov: init-based at n=0, closed form after
    m_read = np.zeros((N, K, q, 1))
    P_read = np.zeros((N, K, q, q))
    for j in range(K):
        z, P, _ = _kalman_exact_y_update(
            params.mu_z0(j).reshape(dim, 1), params.Sigma_z0(j), ys_[0].reshape(s, 1), H
        )
        m_read[0, j] = z[:q]
        P_read[0, j] = P[:q, :q]
    for n in range(1, N):
        y0 = ys_[n - 1].reshape(s, 1)
        y1 = ys_[n].reshape(s, 1)
        for j in range(K):
            m_read[n, j] = M[j] @ y1 + (B[j] - M[j] @ D[j]) @ y0 + c[j]
            P_read[n, j] = Gam[j]
    # time-varying lag-1 kernel
    Ker = np.zeros((N - 1, K, K))
    for n in range(N - 1):
        y1 = ys_[n].reshape(s, 1)
        for j in range(K):
            for k in range(K):
                mm = C[k] @ m_read[n, j] + D[k] @ y1 + bY[k]
                SS = C[k] @ P_read[n, j] @ C[k].T + SV[k]
                Ker[n, j, k] = params.P[j, k] * _gauss(ys_[n + 1], mm, SS)
    gam, gampair = _forward_backward(_alpha1(params, ys_), Ker)
    Ex = np.zeros((N, q))
    Vx = np.zeros((N, q, q))
    for n in range(N - 1):
        y1 = ys_[n].reshape(s, 1)
        y2 = ys_[n + 1].reshape(s, 1)
        means = np.zeros((K, K, q))
        covs = np.zeros((K, K, q, q))
        for j in range(K):
            for k in range(K):
                mm = C[k] @ m_read[n, j] + D[k] @ y1 + bY[k]
                SS = C[k] @ P_read[n, j] @ C[k].T + SV[k]
                Wjk = P_read[n, j] @ C[k].T @ np.linalg.inv(SS)
                means[j, k] = (m_read[n, j] + Wjk @ (y2 - mm)).ravel()
                covs[j, k] = P_read[n, j] - Wjk @ C[k] @ P_read[n, j]
                Ex[n] += gampair[n, j, k] * means[j, k]
        for j in range(K):
            for k in range(K):
                d = means[j, k] - Ex[n]
                Vx[n] += gampair[n, j, k] * (covs[j, k] + np.outer(d, d))
    Ex[N - 1] = sum(gam[N - 1, j] * m_read[N - 1, j].ravel() for j in range(K))
    for j in range(K):
        d = m_read[N - 1, j].ravel() - Ex[N - 1]
        Vx[N - 1] += gam[N - 1, j] * (P_read[N - 1, j] + np.outer(d, d))
    return Ex, Vx


# ---------------------------------------------------------------------------
# S6: collapsed smoothed regime posterior off the union (GPB2 forward +
# pair-likelihood backward), and its light exact reference
# ---------------------------------------------------------------------------
def exact_regime_smoother(params, ys):
    """Exact smoothed regime posterior by path enumeration (no RTS needed)."""
    K, q, s = params.K, params.q, params.s
    dim = q + s
    ys = np.asarray(ys, float).reshape(-1, s)
    N = ys.shape[0]
    H = _obs_matrix(q, s)
    F = [params.f_matrix.F(k) for k in range(K)]
    b = [params.b(k).reshape(dim, 1) for k in range(K)]
    SW = [params.noise_cov.Sigma_W(k) for k in range(K)]
    lws, paths = [], []
    for path in product(range(K), repeat=N):
        k0 = path[0]
        z, P, ll = _kalman_exact_y_update(
            params.mu_z0(k0).reshape(dim, 1), params.Sigma_z0(k0), ys[0].reshape(s, 1), H
        )
        lw = float(np.log(params.pi0[k0] + 1e-300) + ll)
        for n in range(1, N):
            k = path[n]
            zpr = F[k] @ z + b[k]
            Ppr = F[k] @ P @ F[k].T + SW[k]
            z, P, ll = _kalman_exact_y_update(zpr, Ppr, ys[n].reshape(s, 1), H)
            lw += float(np.log(params.P[path[n - 1], k] + 1e-300)) + ll
        lws.append(lw)
        paths.append(path)
    lws = np.array(lws)
    w = np.exp(lws - lws.max())
    w /= w.sum()
    gam = np.zeros((N, K))
    for wi, path in zip(w, paths):
        for n in range(N):
            gam[n, path[n]] += wi
    return gam


def gpb2_bank(params, ys):
    """GPB2 forward pass keeping the per-regime merged PAIR summaries.

    Returns zs (N, K, dim, 1), Ps (N, K, dim, dim), alpha (N, K) and the
    backward-kernel matrices A_n (N-1, K, K) with a_n(j,k) = p_jk Lam^{jk}.
    """
    K, q, s = params.K, params.q, params.s
    dim = q + s
    ys = np.asarray(ys, float).reshape(-1, s)
    N = ys.shape[0]
    H = _obs_matrix(q, s)
    F = [params.f_matrix.F(k) for k in range(K)]
    b = [params.b(k).reshape(dim, 1) for k in range(K)]
    SW = [params.noise_cov.Sigma_W(k) for k in range(K)]
    zs = np.zeros((N, K, dim, 1))
    Ps = np.zeros((N, K, dim, dim))
    alpha = np.zeros((N, K))
    la = np.zeros(K)
    for k in range(K):
        z, P, ll = _kalman_exact_y_update(
            params.mu_z0(k).reshape(dim, 1), params.Sigma_z0(k), ys[0].reshape(s, 1), H
        )
        zs[0, k], Ps[0, k] = z, P
        la[k] = np.log(params.pi0[k] + 1e-300) + ll
    alpha[0] = np.exp(la - la.max())
    alpha[0] /= alpha[0].sum()
    A_n = np.zeros((N - 1, K, K))  # a_n(j,k) = p_jk Lam^{jk}
    for n in range(1, N):
        y = ys[n].reshape(s, 1)
        for k in range(K):
            upd = []
            for j in range(K):
                zpr = F[k] @ zs[n - 1, j] + b[k]
                Ppr = F[k] @ Ps[n - 1, j] @ F[k].T + SW[k]
                z, P, ll = _kalman_exact_y_update(zpr, Ppr, y, H)
                A_n[n - 1, j, k] = params.P[j, k] * np.exp(ll)
                upd.append((z, P))
            w = alpha[n - 1] * A_n[n - 1, :, k]
            wsum = w.sum()
            w = w / wsum if wsum > 0 else np.full(K, 1.0 / K)
            zbar = sum(w[j] * upd[j][0] for j in range(K))
            Pbar = sum(
                w[j] * (upd[j][1] + (upd[j][0] - zbar) @ (upd[j][0] - zbar).T) for j in range(K)
            )
            zs[n, k], Ps[n, k] = zbar, Pbar
        un = alpha[n - 1] @ A_n[n - 1]
        alpha[n] = un / un.sum()
    return zs, Ps, alpha, A_n


def gpb2_regime_smoother(params, ys):
    """Collapsed smoothed regime posterior: GPB2 forward pass (per-regime
    collapsed summaries, pair likelihoods Lam^{jk}) + backward recursion on
    a_n(j,k) = p_jk Lam^{jk}.  Exact on the union, biased off it (S6)."""
    _, _, alpha, A_n = gpb2_bank(params, ys)
    N = alpha.shape[0]
    beta = np.zeros_like(alpha)
    beta[N - 1] = 1.0
    for n in range(N - 2, -1, -1):
        beta[n] = A_n[n] @ beta[n + 1]
        beta[n] /= beta[n].max()
    gam = alpha * beta
    gam /= gam.sum(axis=1, keepdims=True)
    return gam


# ---------------------------------------------------------------------------
# S7/S8: the pair-RTS form -- gain degeneracy and off-domain separation
# ---------------------------------------------------------------------------
def pair_rts_smoother(params, ys, with_residuals=False):
    """Mode-matched pair-RTS smoother run on the pair (RTS-IMM style).

    Forward: GPB2 bank (per-regime merged pair summaries).  Backward: the
    per-regime smoothed pair means are mixed over the ARRIVAL regime with the
    smoothed pair weights, with the classical RTS pair gains
        G^{jk} = P_n(j) F_k^T (F_k P_n(j) F_k^T + SW_k)^{-1} .
    This is the naive collapse: the backward input is the k-conditional
    smoothed summary where the exact recursion calls for the pair-conditional
    one.  On {A==MC} the x-rows of G^{jk} are [0, W_jk] identically, so the
    mismatch is multiplied by zero and the smoother is exact; on {C==0} with
    state memory it is genuinely biased (S7).

    Returns the smoothed state mean (N, q); with_residuals=True also returns
    the max residuals of the two gain identities along the run:
    ('gx', |G_x| / |G_y|)   -- zero on {A==MC}, where G == [0, W_jk];
    ('w',  |G_y - W_jk| / |W_jk|)          -- same identity, gain part;
    ('c0', |G_y + G_x M_k| / |G_x M_k|)    -- zero on {C==0}.
    """
    K, q, s = params.K, params.q, params.s
    dim = q + s
    ys_ = np.asarray(ys, float).reshape(-1, s)
    N = ys_.shape[0]
    F = [params.f_matrix.F(k) for k in range(K)]
    b = [params.b(k).reshape(dim, 1) for k in range(K)]
    SW = [params.noise_cov.Sigma_W(k) for k in range(K)]
    M, Gam, _ = _slaving(params)
    C = [params.f_matrix.C(k) for k in range(K)]
    SV = [params.noise_cov.Sigma_V(k) for k in range(K)]
    W = [
        [Gam[j] @ C[k].T @ np.linalg.inv(C[k] @ Gam[j] @ C[k].T + SV[k]) for k in range(K)]
        for j in range(K)
    ]
    zs, Ps, alpha, A_n = gpb2_bank(params, ys_)
    beta = np.zeros((N, K))
    beta[N - 1] = 1.0
    for n in range(N - 2, -1, -1):
        beta[n] = A_n[n] @ beta[n + 1]
        beta[n] /= beta[n].max()
    gam = alpha * beta
    gam /= gam.sum(axis=1, keepdims=True)

    res = {"gx": 0.0, "w": 0.0, "c0": 0.0}
    Ex = np.zeros((N, q))
    zsm = zs[N - 1].copy()  # (K, dim, 1) smoothed per-regime pair means
    Ex[N - 1] = gam[N - 1] @ zsm[:, :q, 0]
    for n in range(N - 2, -1, -1):
        xi = (alpha[n][:, None] * A_n[n]) * beta[n + 1][None, :]
        xi /= xi.sum()
        new = np.zeros_like(zsm)
        for j in range(K):
            wkj = xi[j, :] / max(xi[j, :].sum(), 1e-300)
            corr = np.zeros((dim, 1))
            for k in range(K):
                Ppred = F[k] @ Ps[n, j] @ F[k].T + SW[k]
                G = Ps[n, j] @ F[k].T @ np.linalg.inv(Ppred)
                zpred = F[k] @ zs[n, j] + b[k]
                corr += wkj[k] * (G @ (zsm[k] - zpred))
                if with_residuals:
                    Gx, Gy = G[:q, :q], G[:q, q:]
                    nGy = np.max(np.abs(Gy)) + 1e-300
                    res["gx"] = max(res["gx"], np.max(np.abs(Gx)) / nGy)
                    if n >= 1:
                        # at n=0 the bank covariance is the (updated) initial
                        # law, not Gamma_j, so G_y = W_jk(P_0) != W_jk there
                        res["w"] = max(res["w"], np.max(np.abs(Gy - W[j][k])) / nGy)
                    GxM = Gx @ M[k]
                    nGxM = np.max(np.abs(GxM)) + 1e-300
                    res["c0"] = max(res["c0"], np.max(np.abs(Gy + GxM)) / nGxM)
            new[j] = zs[n, j] + corr
        zsm = new
        Ex[n] = gam[n] @ zsm[:, :q, 0]
    return (Ex, res) if with_residuals else Ex


# ---------------------------------------------------------------------------
# S9: HBH-structure two-filter smoother, transcribed to the pair
# ---------------------------------------------------------------------------
def hbh_smoother(params, ys, diffuse=1e8, variant="pair"):
    """The Helmick-Blair-Hoffman two-filter smoother on the pair.

    Implemented from the original paper (IEEE Trans. Inf. Theory 41(6):
    1845-1855, 1995; operation order and fusion formulas checked against
    eqs. (42)-(90) on 2026-08-03).

    Forward pass: pairwise IMM bank (per-regime summaries and posteriors).
    Backward pass (their Section II-B, verified order): per arrival mode,
    backward-time prediction through the inverted pair dynamics
    z_n = F_k^{-1}(z_{n+1} - b_k - w) (their (61)-(62)); THEN interaction
    over the arrival mode conditioned on the departure mode (their
    (63)-(66)); THEN the (exact-y) measurement update (their (48)-(53)).
    Pseudo-diffuse initialization at n = N (their (38)-(41): the backward
    ML estimates are Bayesian estimates under a diffuse prior).

    variant="pair" (their MAIN scheme, Section III, n^2 smoothers): per-pair
    Fraser-Potter fusion of the forward filtered mode-j summary with the
    UNMIXED backward predicted mode-k one (their (82)-(83)), pair matching
    likelihoods Lam^{jk} = N(z_bp(k) - z_f(j); 0, P_f(j) + P_bp(k)) (their
    (88)-(90)), pair weights oc p_jk Lam^{jk} (their (74)-(76)) and smoothed
    mode weights mu_s(j) oc alpha_n(j) d_j with d_j = sum_k p_jk Lam^{jk}
    (their (69)-(70)).
    variant="mixed" (their ALTERNATE n-smoother scheme, Section V, with the
    backward likelihood mass restored in the mode weights): fusion with the
    backward MIXED predicted per departure mode, mu_s(j) oc alpha_n(j) m_j
    N(z_f - z_bp; P_f + P_bp).

    Requires invertible F_r -- an assumption the original states explicitly
    ("attention is restricted to systems with invertible state transition
    matrices"); on AB, det F_r = 0 structurally and the scheme is
    inapplicable.  Returns (Ex (N, q), mu_s (N, K)).
    """
    K, q, s = params.K, params.q, params.s
    dim = q + s
    ys_ = np.asarray(ys, float).reshape(-1, s)
    N = ys_.shape[0]
    F = [params.f_matrix.F(k) for k in range(K)]
    Fi = [np.linalg.inv(Fk) for Fk in F]
    b = [params.b(k).reshape(dim, 1) for k in range(K)]
    SW = [params.noise_cov.Sigma_W(k) for k in range(K)]
    H = _obs_matrix(q, s)

    # ---- forward: pairwise IMM bank with per-regime PAIR summaries -------
    zf = np.zeros((N, K, dim, 1))
    Pf = np.zeros((N, K, dim, dim))
    alf = np.zeros((N, K))
    zs, Ps, la = [], [], np.zeros(K)
    for k in range(K):
        z, P, ll = _kalman_exact_y_update(
            params.mu_z0(k).reshape(dim, 1), params.Sigma_z0(k), ys_[0].reshape(s, 1), H
        )
        zs.append(z)
        Ps.append(P)
        la[k] = np.log(params.pi0[k] + 1e-300) + ll
    al = np.exp(la - la.max())
    al /= al.sum()
    zf[0], Pf[0], alf[0] = np.array(zs), np.array(Ps), al
    for n in range(1, N):
        y = ys_[n].reshape(s, 1)
        nzs, nPs, nla = [], [], np.zeros(K)
        for k in range(K):
            w = np.array([params.P[j, k] * al[j] for j in range(K)])
            cbar = w.sum()
            w = w / cbar
            z0 = sum(w[j] * zs[j] for j in range(K))
            P0 = sum(w[j] * (Ps[j] + (zs[j] - z0) @ (zs[j] - z0).T) for j in range(K))
            z, P, ll = _kalman_exact_y_update(F[k] @ z0 + b[k], F[k] @ P0 @ F[k].T + SW[k], y, H)
            nzs.append(z)
            nPs.append(P)
            nla[k] = np.log(cbar + 1e-300) + ll
        zs, Ps = nzs, nPs
        al = np.exp(nla - nla.max())
        al /= al.sum()
        zf[n], Pf[n], alf[n] = np.array(zs), np.array(Ps), al

    # ---- backward-time IMM (pseudo-diffuse init at N) --------------------
    # carrier per mode: summary (zb, Pb) of z_t given y_{t:N}, log-mass lm
    zb = np.zeros((K, dim, 1))
    Pb = np.zeros((K, dim, dim))
    lm = np.zeros(K)
    for k in range(K):
        zb[k, q:, 0] = ys_[N - 1]
        Pb[k][:q, :q] = diffuse * np.eye(q)
    zbp_all = np.zeros((N, K, dim, 1))  # mixed backward PREDICTED (per j)
    Pbp_all = np.zeros((N, K, dim, dim))
    lmp_all = np.full((N, K), -np.inf)
    zpr_all = np.zeros((N, K, dim, 1))  # UNMIXED backward predicted (per k)
    Ppr_all = np.zeros((N, K, dim, dim))
    for n in range(N - 2, -1, -1):
        # backward prediction per mode k (transition n->n+1 has mode r_{n+1}=k)
        zpr, Ppr = [], []
        for k in range(K):
            zpr.append(Fi[k] @ (zb[k] - b[k]))
            Ppr.append(Fi[k] @ (Pb[k] + SW[k]) @ Fi[k].T)
            zpr_all[n, k], Ppr_all[n, k] = zpr[k], Ppr[k]
        # interaction over the arrival mode k -> per departure mode j, THEN
        # the exact-y measurement update with y_n (HBH order: mix first)
        nzb, nPb, nlm = np.zeros_like(zb), np.zeros_like(Pb), np.zeros(K)
        for j in range(K):
            lw = np.array([np.log(params.P[j, k] + 1e-300) + lm[k] for k in range(K)])
            mx = lw.max()
            w = np.exp(lw - mx)
            tot = w.sum()
            w /= tot
            z0 = sum(w[k] * zpr[k] for k in range(K))
            P0 = sum(w[k] * (Ppr[k] + (zpr[k] - z0) @ (zpr[k] - z0).T) for k in range(K))
            zbp_all[n, j], Pbp_all[n, j] = z0, P0
            lmp = mx + np.log(tot)
            lmp_all[n, j] = lmp
            z, P, ll = _kalman_exact_y_update(z0, P0, ys_[n].reshape(s, 1), H)
            nzb[j], nPb[j], nlm[j] = z, P, lmp + ll
        zb, Pb, lm = nzb, nPb, nlm - nlm.max()

    # ---- fusion ----------------------------------------------------------
    Ex = np.zeros((N, q))
    mu_s = np.zeros((N, K))
    Ex[N - 1] = alf[N - 1] @ zf[N - 1, :, :q, 0]
    mu_s[N - 1] = alf[N - 1]
    for n in range(N - 1):
        lwt = np.zeros(K)
        zsm = np.zeros((K, dim, 1))
        if variant == "pair":
            # their main scheme: per-pair fusion (82)-(83), pair matching
            # likelihoods (88)-(90), weights (69)-(70) and (74)-(76)
            for j in range(K):
                lw = np.zeros(K)
                zpair = np.zeros((K, dim, 1))
                for k in range(K):
                    Djk = Pf[n, j] + Ppr_all[n, k]
                    d = (zpr_all[n, k] - zf[n, j]).ravel()
                    _, logdet = np.linalg.slogdet(2 * np.pi * Djk)
                    lw[k] = np.log(params.P[j, k] + 1e-300) - 0.5 * (
                        d @ np.linalg.solve(Djk, d) + logdet
                    )
                    G = Pf[n, j] @ np.linalg.inv(Djk)
                    zpair[k] = zf[n, j] + G @ (zpr_all[n, k] - zf[n, j])
                mx = lw.max()
                w = np.exp(lw - mx)
                tot = w.sum()
                zsm[j] = sum((w[k] / tot) * zpair[k] for k in range(K))
                lwt[j] = np.log(alf[n, j] + 1e-300) + mx + np.log(tot)
        else:
            # their alternate n-smoother scheme (Section V), with the
            # backward likelihood mass restored in the mode weights
            for j in range(K):
                Psum = Pf[n, j] + Pbp_all[n, j]
                d = (zbp_all[n, j] - zf[n, j]).ravel()
                _, logdet = np.linalg.slogdet(2 * np.pi * Psum)
                lwt[j] = (
                    np.log(alf[n, j] + 1e-300)
                    + lmp_all[n, j]
                    - 0.5 * (d @ np.linalg.solve(Psum, d) + logdet)
                )
                G = Pf[n, j] @ np.linalg.inv(Psum)
                zsm[j] = zf[n, j] + G @ (zbp_all[n, j] - zf[n, j])
        w = np.exp(lwt - lwt.max())
        w /= w.sum()
        mu_s[n] = w
        Ex[n] = w @ zsm[:, :q, 0]
    return Ex, mu_s


# ---------------------------------------------------------------------------
# S13: classical collapsed smoothers -- exactness domains
# ---------------------------------------------------------------------------
def imm_pair_bank(params, ys):
    """Pairwise IMM forward pass keeping the per-regime PAIR summaries.

    Same recursion as pairwise_imm_bank (mixing, per-regime prediction, exact
    y update), returned in the layout of gpb2_bank: zs (N, K, dim, 1), Ps
    (N, K, dim, dim), alpha (N, K) and the per-pair likelihood matrices A_n
    (N-1, K, K), a_n(j,k) = p_jk N(y_{n+1}; prediction of the mode-j summary
    through F_k) -- computed alongside, used only by the 'pair' weights.
    Zero-probability arrivals (pair chains) keep their previous summary.
    """
    K, q, s = params.K, params.q, params.s
    dim = q + s
    ys = np.asarray(ys, float).reshape(-1, s)
    N = ys.shape[0]
    H = _obs_matrix(q, s)
    F = [params.f_matrix.F(k) for k in range(K)]
    b = [params.b(k).reshape(dim, 1) for k in range(K)]
    SW = [params.noise_cov.Sigma_W(k) for k in range(K)]
    zs = np.zeros((N, K, dim, 1))
    Ps = np.zeros((N, K, dim, dim))
    alpha = np.zeros((N, K))
    la = np.zeros(K)
    for k in range(K):
        z, P, ll = _kalman_exact_y_update(
            params.mu_z0(k).reshape(dim, 1), params.Sigma_z0(k), ys[0].reshape(s, 1), H
        )
        zs[0, k], Ps[0, k] = z, P
        la[k] = np.log(params.pi0[k] + 1e-300) + ll
    alpha[0] = np.exp(la - la.max())
    alpha[0] /= alpha[0].sum()
    A_n = np.zeros((N - 1, K, K))
    for n in range(1, N):
        y = ys[n].reshape(s, 1)
        nla = np.full(K, -np.inf)
        for k in range(K):
            for j in range(K):
                zpr = F[k] @ zs[n - 1, j] + b[k]
                Ppr = F[k] @ Ps[n - 1, j] @ F[k].T + SW[k]
                _, _, ll = _kalman_exact_y_update(zpr, Ppr, y, H)
                A_n[n - 1, j, k] = params.P[j, k] * np.exp(ll)
            w = alpha[n - 1] * params.P[:, k]
            cbar = w.sum()
            if cbar <= 0.0:
                zs[n, k], Ps[n, k] = zs[n - 1, k], Ps[n - 1, k]
                continue
            w = w / cbar
            z0 = sum(w[j] * zs[n - 1, j] for j in range(K))
            P0 = sum(
                w[j] * (Ps[n - 1, j] + (zs[n - 1, j] - z0) @ (zs[n - 1, j] - z0).T)
                for j in range(K)
            )
            z, P, ll = _kalman_exact_y_update(F[k] @ z0 + b[k], F[k] @ P0 @ F[k].T + SW[k], y, H)
            zs[n, k], Ps[n, k] = z, P
            nla[k] = np.log(cbar) + ll
        a = np.exp(nla - nla.max())
        alpha[n] = a / a.sum()
    return zs, Ps, alpha, A_n


def collapsed_rts_smoother(params, ys, bank="gpb2", weights="kim"):
    """Classical collapsed RTS smoother, parameterized by its two choices.

    bank: the forward per-regime summaries -- 'gpb2' (the Kim filter: per-pair
    update, collapse over the departure regime) or 'imm' (interacting
    multiple model).  weights: the backward regime weights -- 'kim' (Kim,
    J. Econometrics 60, 1994: p(r_n | r_{n+1}, y_{1:N}) is replaced by
    p(r_n | r_{n+1}, y_{1:n}), so gam_n(j) = alpha_n(j) sum_k p_jk
    gam_{n+1}(k) / (alpha_n P)(k)) or 'pair' (backward recursion on the pair
    likelihoods a_n(j,k), as in S6/S7).  State: per pair (j, k) the RTS gain
    G^{jk} = P_n(j) F_k^T (F_k P_n(j) F_k^T + SW_k)^{-1} applied to the
    k-conditional smoothed summary, mixed over the arrival regime.
    ('gpb2', 'kim') is Kim's smoother; ('imm', 'kim') the IMM forward
    filtering / backward smoothing of the RTS type (cf. Nadarajah et al.,
    IEEE TAES 48(3), 2012 -- our transcription, not checked against the
    original); ('gpb2', 'pair') reproduces pair_rts_smoother (S7).
    Returns the smoothed state mean (N, q) and regime posterior (N, K).
    """
    K, q, s = params.K, params.q, params.s
    dim = q + s
    ys_ = np.asarray(ys, float).reshape(-1, s)
    N = ys_.shape[0]
    F = [params.f_matrix.F(k) for k in range(K)]
    b = [params.b(k).reshape(dim, 1) for k in range(K)]
    SW = [params.noise_cov.Sigma_W(k) for k in range(K)]
    zs, Ps, alpha, A_n = (gpb2_bank if bank == "gpb2" else imm_pair_bank)(params, ys_)
    gam = np.zeros((N, K))
    if weights == "pair":
        beta = np.zeros((N, K))
        beta[N - 1] = 1.0
        for n in range(N - 2, -1, -1):
            beta[n] = A_n[n] @ beta[n + 1]
            beta[n] /= beta[n].max()
        gam = alpha * beta
        gam /= gam.sum(axis=1, keepdims=True)
    else:
        gam[N - 1] = alpha[N - 1]
    Ex = np.zeros((N, q))
    zsm = zs[N - 1].copy()  # (K, dim, 1) smoothed per-regime pair means
    Ex[N - 1] = gam[N - 1] @ zsm[:, :q, 0]
    for n in range(N - 2, -1, -1):
        if weights == "pair":
            xi = (alpha[n][:, None] * A_n[n]) * beta[n + 1][None, :]
            xi /= xi.sum()
        else:
            pred = alpha[n] @ params.P
            ratio = np.where(pred > 0, gam[n + 1] / np.maximum(pred, 1e-300), 0.0)
            xi = alpha[n][:, None] * params.P * ratio[None, :]
            gam[n] = xi.sum(axis=1)
        new = np.zeros_like(zsm)
        for j in range(K):
            wj = xi[j].sum()
            new[j] = zs[n, j]
            if wj <= 1e-300:
                continue
            for k in range(K):
                if xi[j, k] <= 0.0:
                    continue
                Ppred = F[k] @ Ps[n, j] @ F[k].T + SW[k]
                G = Ps[n, j] @ F[k].T @ np.linalg.pinv(Ppred)
                new[j] = new[j] + xi[j, k] / wj * (G @ (zsm[k] - F[k] @ zs[n, j] - b[k]))
        zsm = new
        Ex[n] = gam[n] @ zsm[:, :q, 0]
    return Ex, gam


def onestep_collapsed_smoother(params, ys, bank="gpb2"):
    """Collapsed ONE-STEP smoother: no backward state recursion.

    Per pair (j, k), the mode-j filtered pair summary at n (forward bank
    'gpb2' or 'imm') is retrodicted by y_{n+1} through regime k --
    E[x_n | j, k, y_{1:n+1}] by one Kalman step on (z_n, y_{n+1}) -- and the
    pair summaries are mixed with the smoothed pair weights of the
    pair-likelihood backward recursion; the filtered summary at n = N.  On the
    cross-annihilation family C_k (A_j - M_j C_j) = 0 the observations beyond
    y_{n+1} carry nothing on x_n (its influence stays in the range of the
    memories, which every C_l annihilates), and the merged per-regime summary
    updates exactly (the components differ in ker C_k only): with the GPB2
    bank the smoother is exact on the whole GPB2 domain, with the IMM bank on
    {C==0} -- each time the domain of its forward filter, the bound set by the
    terminal point (S13d).  Returns (Ex, Vx, gam).
    """
    K, q, s = params.K, params.q, params.s
    dim = q + s
    ys_ = np.asarray(ys, float).reshape(-1, s)
    N = ys_.shape[0]
    H = _obs_matrix(q, s)
    F = [params.f_matrix.F(k) for k in range(K)]
    b = [params.b(k).reshape(dim, 1) for k in range(K)]
    SW = [params.noise_cov.Sigma_W(k) for k in range(K)]
    zs, Ps, alpha, A_n = (gpb2_bank if bank == "gpb2" else imm_pair_bank)(params, ys_)
    beta = np.zeros((N, K))
    beta[N - 1] = 1.0
    for n in range(N - 2, -1, -1):
        beta[n] = A_n[n] @ beta[n + 1]
        beta[n] /= beta[n].max()
    gam = alpha * beta
    gam /= gam.sum(axis=1, keepdims=True)
    Ex = np.zeros((N, q))
    Vx = np.zeros((N, q, q))
    for n in range(N - 1):
        xi = (alpha[n][:, None] * A_n[n]) * beta[n + 1][None, :]
        xi /= xi.sum()
        y1 = ys_[n + 1].reshape(s, 1)
        means = np.zeros((K, K, q))
        covs = np.zeros((K, K, q, q))
        for j in range(K):
            for k in range(K):
                Pzy = Ps[n, j] @ F[k].T @ H.T  # Cov(z_n, y_{n+1} | j, k)
                S = H @ (F[k] @ Ps[n, j] @ F[k].T + SW[k]) @ H.T
                G = Pzy @ np.linalg.inv(S)
                m = zs[n, j] + G @ (y1 - H @ (F[k] @ zs[n, j] + b[k]))
                means[j, k] = m[:q, 0]
                covs[j, k] = (Ps[n, j] - G @ Pzy.T)[:q, :q]
                Ex[n] += xi[j, k] * means[j, k]
        for j in range(K):
            for k in range(K):
                d = means[j, k] - Ex[n]
                Vx[n] += xi[j, k] * (covs[j, k] + np.outer(d, d))
    Ex[N - 1] = gam[N - 1] @ zs[N - 1, :, :q, 0]
    for j in range(K):
        d = zs[N - 1, j, :q, 0] - Ex[N - 1]
        Vx[N - 1] += gam[N - 1, j] * (Ps[N - 1, j, :q, :q] + np.outer(d, d))
    return Ex, Vx, gam


CLASSICAL_SMOOTHERS = (
    ("Kim (GPB2, Kim weights)", "gpb2", "kim"),
    ("IMM-RTS (IMM, Kim weights)", "imm", "kim"),
    ("GPB2 pair-RTS (S7)", "gpb2", "pair"),
    ("IMM pair-RTS", "imm", "pair"),
)


def memoryless_c0_model():
    """S13 gauge: the CGOMSM1 gauge without state memory (A = 0, C = 0), i.e.
    the intersection {C==0} n {A==MC} of the two branches."""
    return _params(
        A=[0.0, 0.0],
        B=[0.10, 0.10],
        C=[0.0, 0.0],
        D=[0.50, 0.50],
        SU=[0.40, 0.35],
        Dt=[0.15, -0.20],
        SV=[0.20, 0.60],
        p_switch=0.10,
    )


def _gss(P, A, B, C, D, SU, Dt, SV, pi0=None, mu0=None, S0=None, b=None):
    K, q, s = len(A), A[0].shape[0], D[0].shape[0]
    return GSSParams(
        K=K,
        q=q,
        s=s,
        P=P,
        f_matrix=FMatrix(K, q, s, A, B, C, D),
        noise_cov=GSSNoiseCovariance(K, q, s, SU, Dt, SV),
        pi0=pi0,
        mu_z0_list=mu0 if mu0 is not None else [np.zeros((q + s, 1))] * K,
        Sigma_z0_list=S0 if S0 is not None else [np.eye(q + s)] * K,
        b_list=b,
    )


def random_family_params(rng, K, q, s, family):
    """Random stable GSSParams (stationary init) of a family: 'C0A0' (C = A =
    0), 'C0mem' (C = 0, random memory A), 'AMC' (A = M C, B free) or 'AB'."""
    while True:
        A, B, C, D, SU, Dt, SV = [], [], [], [], [], [], []
        for _ in range(K):
            SVk = _rand_spd(rng, s, 0.3)
            Mk = rng.normal(size=(q, s)) * 0.6
            Gk = _rand_spd(rng, q, 0.3)
            Ck = np.zeros((s, q)) if family.startswith("C0") else rng.normal(size=(s, q)) * 0.5
            Dk = rng.normal(size=(s, s)) * 0.4
            if family == "C0A0":
                Ak = np.zeros((q, q))
            elif family == "C0mem":
                Ak = rng.normal(size=(q, q)) * 0.35
            else:
                Ak = Mk @ Ck
            Bk = Mk @ Dk if family == "AB" else rng.normal(size=(q, s)) * 0.4
            A.append(Ak)
            B.append(Bk)
            C.append(Ck)
            D.append(Dk)
            SU.append(Gk + Mk @ SVk @ Mk.T)
            Dt.append(Mk @ SVk)
            SV.append(SVk)
        if _contracting(A, B, C, D):
            return with_stationary_init(_gss(_sticky(K), A, B, C, D, SU, Dt, SV))


def random_cross_annihilation_params(rng, K, q=3, s=1):
    """Random stable model of the cross-annihilation family outside the union:
    rows C_k = [C0_k, 0] (s x q), memories N_k with zero first s rows (so
    C_l N_k = 0, N_k != 0), B free."""
    while True:
        A, B, C, D, SU, Dt, SV = [], [], [], [], [], [], []
        for _ in range(K):
            SVk = _rand_spd(rng, s, 0.3)
            Mk = rng.normal(size=(q, s)) * 0.6
            Gk = _rand_spd(rng, q, 0.3)
            Ck = np.zeros((s, q))
            Ck[:, :s] = rng.normal(size=(s, s)) * 0.5
            Nk = np.zeros((q, q))
            Nk[s:, :] = rng.normal(size=(q - s, q)) * 0.4
            A.append(Mk @ Ck + Nk)
            B.append(rng.normal(size=(q, s)) * 0.4)
            C.append(Ck)
            D.append(rng.normal(size=(s, s)) * 0.4)
            SU.append(Gk + Mk @ SVk @ Mk.T)
            Dt.append(Mk @ SVk)
            SV.append(SVk)
        if _contracting(A, B, C, D):
            return with_stationary_init(_gss(_sticky(K), A, B, C, D, SU, Dt, SV))


def _blocks(p):
    fm, nc, K = p.f_matrix, p.noise_cov, p.K
    return (
        [fm.A(k) for k in range(K)],
        [fm.B(k) for k in range(K)],
        [fm.C(k) for k in range(K)],
        [fm.D(k) for k in range(K)],
        [nc.Sigma_U(k) for k in range(K)],
        [nc.Delta(k) for k in range(K)],
        [nc.Sigma_V(k) for k in range(K)],
    )


def slaved_init(params):
    """The same model with a slaved initial law: Y_1 | r keeps its law and
    X_1 | r, Y_1 ~ N(M_r Y_1 + c_r, Gam_r) (the hypothesis of Section V-E of
    the exactness paper)."""
    K, q = params.K, params.q
    M, Gam, c = _slaving(params)
    mu0, S0 = [], []
    for k in range(K):
        Syy = params.Sigma_z0(k)[q:, q:]
        my = params.mu_z0(k).reshape(-1, 1)[q:]
        S0.append(np.block([[Gam[k] + M[k] @ Syy @ M[k].T, M[k] @ Syy], [Syy @ M[k].T, Syy]]))
        mu0.append(np.vstack([M[k] @ my + c[k], my]))
    b = [params.b(k) for k in range(K)]
    return _gss(params.P, *_blocks(params), pi0=params.pi0, mu0=mu0, S0=S0, b=b)


def k2_embedding(params):
    """Section V-E of the exactness paper: the white-noise CGOMSM1 on the pair
    chain (r_{n-1}, r_n) (index j K + k) with blocks C' = A' = 0,
    D'_jk = D_k + C_k M_j, B' = M_k D', b'^Y = C_k c_j + b^Y_k,
    b'^X = c_k + M_k b'^Y, SV' = SV_k + C_k Gam_j C_k^T, Delta' = M_k SV',
    SU' = Gam_k + M_k SV' M_k^T.  For a slaved AB model it has the same
    (R, Y) law and the same exact filter; the dummy r_0 is uniform."""
    K, q, s = params.K, params.q, params.s
    M, Gam, c = _slaving(params)
    _, _, C, D, _, _, SV = _blocks(params)
    bY = [params.b(k).reshape(q + s, 1)[q:] for k in range(K)]
    A2, B2, C2, D2, SU2, Dt2, SV2, b2, mu0, S0 = [], [], [], [], [], [], [], [], [], []
    P2 = np.zeros((K * K, K * K))
    pi2 = np.zeros(K * K)
    for j in range(K):
        for k in range(K):
            Dp = D[k] + C[k] @ M[j]
            SVp = SV[k] + C[k] @ Gam[j] @ C[k].T
            bYp = C[k] @ c[j] + bY[k]
            A2.append(np.zeros((q, q)))
            B2.append(M[k] @ Dp)
            C2.append(np.zeros((s, q)))
            D2.append(Dp)
            SV2.append(SVp)
            Dt2.append(M[k] @ SVp)
            SU2.append(Gam[k] + M[k] @ SVp @ M[k].T)
            b2.append(np.vstack([c[k] + M[k] @ bYp, bYp]))
            mu0.append(params.mu_z0(k).reshape(-1, 1))
            S0.append(params.Sigma_z0(k))
            pi2[j * K + k] = params.pi0[k] / K
            P2[j * K + k, k * K : (k + 1) * K] = params.P[k]
    return _gss(P2, A2, B2, C2, D2, SU2, Dt2, SV2, pi0=pi2, mu0=mu0, S0=S0, b=b2)


# ---------------------------------------------------------------------------
# S12 machinery: pair-dependent models, exact path quantities, time reversal
# ---------------------------------------------------------------------------
# A pair-dependent model is a dict with keys K, q, s, F[j][k], SW[j][k] (the
# centered transition Z_{n+1} = F[j][k] Z_n + W, W ~ N(0, SW[j][k]), when
# (r_n, r_{n+1}) = (j, k)), P, pi0, S0[k] (law N(0, S0[k]) of Z_1 | r_1 = k);
# optionally "blocks" (per-regime A..SV of a THGMSM, where F[j][k] = F_k) and
# "Pdyn"(j, k, y_n), a regime transition depending on the current observation
# (needed by the time reversal).
def exact_pair_quantities(model, ys):
    """Exact filtered and smoothed quantities of a pair-dependent model by
    path enumeration (conditional pair Kalman filter + RTS per regime path).

    Returns a dict: Ex (N, q) smoothed mean; pi (N, K) smoothed regime
    posterior; gpair (N-1, K, K) smoothed pair posterior; Exf (N, q) filtered
    mean; filt_reg, smo_reg (N, K, q) the per-regime means E[X_n | r_n, y_{1:n}]
    and E[X_n | r_n, y_{1:N}]; pair_cur, pair_next (N-1, K, K, q) the
    pair-conditional smoothed means E[X_n | r_n, r_{n+1}, y_{1:N}] and
    E[X_{n+1} | r_n, r_{n+1}, y_{1:N}].
    """
    K, q, s = model["K"], model["q"], model["s"]
    F, SW = model["F"], model["SW"]
    ys = np.asarray(ys, float).reshape(-1, s)
    N = ys.shape[0]
    H = _obs_matrix(q, s)
    recs = []
    for path in product(range(K), repeat=N):
        z, Pc, ll = _kalman_exact_y_update(
            np.zeros((q + s, 1)), model["S0"][path[0]], ys[0].reshape(s, 1), H
        )
        lw = float(np.log(model["pi0"][path[0]]) + ll)
        zf, Pf, zp, Pp, lwf = [z], [Pc], [None], [None], [lw]
        for n in range(1, N):
            j, k = path[n - 1], path[n]
            zpr = F[j][k] @ zf[-1]
            Ppr = F[j][k] @ Pf[-1] @ F[j][k].T + SW[j][k]
            z, Pc, ll = _kalman_exact_y_update(zpr, Ppr, ys[n].reshape(s, 1), H)
            ptr = model["Pdyn"](j, k, ys[n - 1]) if "Pdyn" in model else model["P"][j, k]
            lw += float(np.log(ptr) + ll)
            zf.append(z)
            Pf.append(Pc)
            zp.append(zpr)
            Pp.append(Ppr)
            lwf.append(lw)
        zs = [None] * N
        zs[N - 1] = zf[N - 1]
        for n in range(N - 2, -1, -1):
            Fjk = F[path[n]][path[n + 1]]
            G = Pf[n] @ Fjk.T @ np.linalg.inv(Pp[n + 1])
            zs[n] = zf[n] + G @ (zs[n + 1] - zp[n + 1])
        recs.append((path, lw, lwf, zf, zs))
    lws = np.array([r[1] for r in recs])
    w = np.exp(lws - lws.max())
    w /= w.sum()
    Ex = np.zeros((N, q))
    pis = np.zeros((N, K))
    gpair = np.zeros((N - 1, K, K))
    smo_num = np.zeros((N, K, q))
    cur_num = np.zeros((N - 1, K, K, q))
    nxt_num = np.zeros((N - 1, K, K, q))
    for wi, (path, _, _, _, zs) in zip(w, recs):
        for n in range(N):
            Ex[n] += wi * zs[n][:q, 0]
            pis[n, path[n]] += wi
            smo_num[n, path[n]] += wi * zs[n][:q, 0]
        for n in range(N - 1):
            gpair[n, path[n], path[n + 1]] += wi
            cur_num[n, path[n], path[n + 1]] += wi * zs[n][:q, 0]
            nxt_num[n, path[n], path[n + 1]] += wi * zs[n + 1][:q, 0]
    # filtered quantities: one weight per regime prefix r_{1:n}
    Exf = np.zeros((N, q))
    filt_num = np.zeros((N, K, q))
    filt_den = np.zeros((N, K))
    for n in range(N):
        prefixes = {}
        for path, _, lwf, zf, _ in recs:
            prefixes.setdefault(path[: n + 1], (lwf[n], zf[n][:q, 0]))
        lw_n = np.array([v[0] for v in prefixes.values()])
        wn = np.exp(lw_n - lw_n.max())
        wn /= wn.sum()
        for wi, (pre, (_, xm)) in zip(wn, prefixes.items()):
            Exf[n] += wi * xm
            filt_num[n, pre[-1]] += wi * xm
            filt_den[n, pre[-1]] += wi
    gp = np.maximum(gpair[..., None], 1e-300)
    return dict(
        Ex=Ex,
        pi=pis,
        gpair=gpair,
        Exf=Exf,
        filt_reg=filt_num / np.maximum(filt_den[:, :, None], 1e-300),
        smo_reg=smo_num / np.maximum(pis[:, :, None], 1e-300),
        pair_cur=cur_num / gp,
        pair_next=nxt_num / gp,
    )


def _stationary_pair_cov(F, SW, P, tol=1e-15, max_iter=20000):
    """Stationary regime law pi and per-regime Var[Z_n | r_n = k] of a
    pair-dependent model (fixed point of the forward moment recursion)."""
    K = P.shape[0]
    vals, vecs = np.linalg.eig(P.T)
    pi = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    pi = pi / pi.sum()
    d = F[0][0].shape[0]
    Sig = [np.eye(d) for _ in range(K)]
    for _ in range(max_iter):
        new = []
        for k in range(K):
            acc = np.zeros((d, d))
            for j in range(K):
                acc += pi[j] * P[j, k] / pi[k] * (F[j][k] @ Sig[j] @ F[j][k].T + SW[j][k])
            new.append(0.5 * (acc + acc.T))
        gap = max(np.max(np.abs(a - b)) for a, b in zip(new, Sig))
        if gap > 1e100:  # not mean-square stable: no stationary law
            return pi, [np.full((d, d), np.inf) for _ in range(K)]
        Sig = new
        if gap < tol:
            break
    return pi, Sig


def pair_thgmsm(A, B, C, D, SU, Dt, SV, P):
    """THGMSM (F[j][k] = F_k) as a pair-dependent model, started from its
    stationary regime law and the Gaussian moment-matched per-regime law."""
    K = len(A)
    q, s = A[0].shape[0], D[0].shape[0]
    F = [[np.block([[A[k], B[k]], [C[k], D[k]]]) for k in range(K)] for _ in range(K)]
    SW = [[np.block([[SU[k], Dt[k]], [Dt[k].T, SV[k]]]) for k in range(K)] for _ in range(K)]
    pi, Sig = _stationary_pair_cov(F, SW, P)
    blocks = dict(A=A, B=B, C=C, D=D, SU=SU, Dt=Dt, SV=SV)
    return dict(K=K, q=q, s=s, F=F, SW=SW, P=P, pi0=pi, S0=Sig, blocks=blocks)


def _sticky(K, p_stay=0.85):
    if K == 1:
        return np.array([[1.0]])
    P = np.full((K, K), (1.0 - p_stay) / (K - 1))
    np.fill_diagonal(P, p_stay)
    return P


def _rand_spd(rng, n, scale):
    G = rng.normal(size=(n, n)) * scale
    return G @ G.T + 0.2 * np.eye(n)


def _contracting(A, B, C, D):
    """Every per-regime F_k has spectral radius below 0.9."""
    return all(
        np.max(np.abs(np.linalg.eigvals(np.block([[a, b], [c, d]])))) < 0.9
        for a, b, c, d in zip(A, B, C, D)
    )


def random_cgomsm2(rng, K, q, s):
    """Random stable CGOMSM2: A = M C, B = M D, Delta = M SV, SU = Gam + M SV M^T."""
    while True:
        A, B, C, D, SU, Dt, SV = [], [], [], [], [], [], []
        for _ in range(K):
            SVk = _rand_spd(rng, s, 0.3)
            Mk = rng.normal(size=(q, s)) * 0.6
            Gk = _rand_spd(rng, q, 0.3)
            Ck = rng.normal(size=(s, q)) * 0.5
            Dk = rng.normal(size=(s, s)) * 0.4
            A.append(Mk @ Ck)
            B.append(Mk @ Dk)
            C.append(Ck)
            D.append(Dk)
            SU.append(Gk + Mk @ SVk @ Mk.T)
            Dt.append(Mk @ SVk)
            SV.append(SVk)
        if not _contracting(A, B, C, D):
            continue
        m = pair_thgmsm(A, B, C, D, SU, Dt, SV, _sticky(K))
        if all(np.all(np.isfinite(S)) for S in m["S0"]):
            return m


def random_cgomsm1(rng, K, q, s, memory=True):
    """Random stable CGOMSM1 (C = 0); state memory A = 0 when memory=False."""
    while True:
        A, B, C, D, SU, Dt, SV = [], [], [], [], [], [], []
        for _ in range(K):
            Wk = _rand_spd(rng, q + s, 0.4)
            A.append(rng.normal(size=(q, q)) * 0.35 if memory else np.zeros((q, q)))
            B.append(rng.normal(size=(q, s)) * 0.4)
            C.append(np.zeros((s, q)))
            D.append(rng.normal(size=(s, s)) * 0.4)
            SU.append(Wk[:q, :q])
            Dt.append(Wk[:q, q:])
            SV.append(Wk[q:, q:])
        if not _contracting(A, B, C, D):
            continue
        m = pair_thgmsm(A, B, C, D, SU, Dt, SV, _sticky(K))
        if all(np.all(np.isfinite(S)) for S in m["S0"]):
            return m


def simulate_pair(model, N, seed):
    """Observation record y_{1:N} (N, s) of a pair-dependent model."""
    rng = np.random.default_rng(seed)
    K, q, s = model["K"], model["q"], model["s"]
    r = rng.choice(K, p=model["pi0"])
    z = rng.multivariate_normal(np.zeros(q + s), model["S0"][r]).reshape(-1, 1)
    ys = [z[q:, 0].copy()]
    for _ in range(1, N):
        k = rng.choice(K, p=model["P"][r])
        w = rng.multivariate_normal(np.zeros(q + s), model["SW"][r][k]).reshape(-1, 1)
        z = model["F"][r][k] @ z + w
        r = k
        ys.append(z[q:, 0].copy())
    return np.array(ys)


def _pair_slaving(model):
    b, K = model["blocks"], model["K"]
    M = [b["Dt"][k] @ np.linalg.inv(b["SV"][k]) for k in range(K)]
    Gam = [b["SU"][k] - M[k] @ b["SV"][k] @ M[k].T for k in range(K)]
    return M, Gam


def abb15_pair_model(m2):
    """The pair-dependent CGOMSM1 of [Abb15] matched to a CGOMSM2 m2 (Remark
    rem:pairmodel of the paper): same (R,Y) law, same filter.

    Both models are started from the slaved initial law Z_1 ~ N(0, SW_{r_1});
    returns (pair model, m2 with that initial law).
    """
    b, K, q, s = m2["blocks"], m2["K"], m2["q"], m2["s"]
    M, Gam = _pair_slaving(m2)
    F = [[None] * K for _ in range(K)]
    SW = [[None] * K for _ in range(K)]
    for j in range(K):
        for k in range(K):
            Dp = b["D"][k] + b["C"][k] @ M[j]
            SVp = b["SV"][k] + b["C"][k] @ Gam[j] @ b["C"][k].T
            Dtp = M[k] @ SVp
            F[j][k] = np.block([[np.zeros((q, q)), M[k] @ Dp], [np.zeros((s, q)), Dp]])
            SW[j][k] = np.block([[Gam[k] + M[k] @ SVp @ M[k].T, Dtp], [Dtp.T, SVp]])
    S0 = [m2["SW"][k][k] for k in range(K)]
    pair = dict(K=K, q=q, s=s, F=F, SW=SW, P=m2["P"], pi0=m2["pi0"], S0=S0)
    return pair, dict(m2, S0=S0)


def time_reversal(model):
    """Moment-matched Gaussian time reversal of a pair-dependent model started
    from its stationary per-regime law S0.

    Reversed regime chain P~[k, j] = pi_j P[j, k] / pi_k; reversed transition
    Z_n = G Z_{n+1} + W~ (regression of Z_n on Z_{n+1} given (r_n, r_{n+1}) =
    (j, k)), indexed in reversed time by (k, j); reversed regime transition
    p(r_n = j | r_{n+1} = k, y_{n+1}), the form it takes when the law of X_{n+1}
    given (y_{n+1}, r_n, r_{n+1}) does not depend on r_n (CGOMSM2).  Exact at
    the first transition; for K >= 2 the per-regime law is a mixture and the
    reversal is approximate (the mixture caveat measured by S12).
    """
    K, q = model["K"], model["q"]
    pi, Sig, P = model["pi0"], model["S0"], model["P"]
    Pr = np.array([[pi[j] * P[j, k] / pi[k] for j in range(K)] for k in range(K)])
    Fr = [[None] * K for _ in range(K)]
    SWr = [[None] * K for _ in range(K)]
    SYY = [[None] * K for _ in range(K)]
    for j in range(K):
        for k in range(K):
            F = model["F"][j][k]
            S1 = F @ Sig[j] @ F.T + model["SW"][j][k]
            G = Sig[j] @ F.T @ np.linalg.inv(S1)
            R = Sig[j] - G @ S1 @ G.T
            Fr[k][j] = G
            SWr[k][j] = 0.5 * (R + R.T)
            SYY[j][k] = S1[q:, q:]

    def Pdyn(k, j, y):  # reversed step r~_m = k -> r~_{m+1} = j, y = y~_m
        y = np.asarray(y, float).reshape(-1)
        lw = np.array(
            [
                np.log(pi[i] * P[i, k])
                - 0.5 * (np.linalg.slogdet(SYY[i][k])[1] + y @ np.linalg.solve(SYY[i][k], y))
                for i in range(K)
            ]
        )
        wts = np.exp(lw - lw.max())
        return wts[j] / wts.sum()

    return dict(K=K, q=q, s=model["s"], F=Fr, SW=SWr, P=Pr, pi0=pi, S0=Sig, Pdyn=Pdyn)


def reversed_regime_state_gap(model):
    """max over (j, j', k) of the gap between the Gaussian laws of X_{n+1}
    given (y_{n+1}, r_n, r_{n+1} = k) for r_n = j and r_n = j' (regression
    coefficient and covariance) at the first transition: 0 iff the reversed
    regime transition does not involve the hidden state."""
    K, q, Sig = model["K"], model["q"], model["S0"]
    laws = {}
    for j in range(K):
        for k in range(K):
            S = model["F"][j][k] @ Sig[j] @ model["F"][j][k].T + model["SW"][j][k]
            reg = S[:q, q:] @ np.linalg.inv(S[q:, q:])
            laws[j, k] = (reg, S[:q, :q] - reg @ S[q:, :q])
    return max(
        max(
            np.max(np.abs(laws[j, k][0] - laws[i, k][0])),
            np.max(np.abs(laws[j, k][1] - laws[i, k][1])),
        )
        for k in range(K)
        for j in range(K)
        for i in range(K)
    )


def pair_ab_residual(model):
    """max over pairs of |A - M C|, |B - M D|, M = Delta SV^{-1}."""
    q, K, out = model["q"], model["K"], 0.0
    for j in range(K):
        for k in range(K):
            F, SW = model["F"][j][k], model["SW"][j][k]
            M = SW[:q, q:] @ np.linalg.inv(SW[q:, q:])
            out = max(
                out,
                np.max(np.abs(F[:q, :q] - M @ F[q:, :q])),
                np.max(np.abs(F[:q, q:] - M @ F[q:, q:])),
            )
    return out


def pair_block_max(model, which):
    """max over pairs of |A| (which="A") or |C| (which="C")."""
    q = model["q"]
    sl = {"A": (slice(0, q), slice(0, q)), "C": (slice(q, None), slice(0, q))}[which]
    K = model["K"]
    return max(np.max(np.abs(model["F"][j][k][sl])) for j in range(K) for k in range(K))


def one_step_backward_summary(m2, ys, gpair):
    """E[X_n | r_n = j, y_{n:N}] on a centered CGOMSM2 by the one-step backward
    summary (eq. (backward-summary) of the paper):
    sum_k p(r_{n+1} = k | r_n = j, y_{1:N}) [M_j y_n + W_jk (y_{n+1} - (C_k M_j + D_k) y_n)].
    Rows n = 1..N-2 (0-based) are filled: n = 0 sits on the moment-matched
    initial law, which is not slaved."""
    b, K, q, s = m2["blocks"], m2["K"], m2["q"], m2["s"]
    M, Gam = _pair_slaving(m2)
    ys = np.asarray(ys, float).reshape(-1, s)
    out = np.zeros((ys.shape[0] - 1, K, q))
    for n in range(1, ys.shape[0] - 1):
        y0, y1 = ys[n].reshape(s, 1), ys[n + 1].reshape(s, 1)
        for j in range(K):
            acc = np.zeros((q, 1))
            for k in range(K):
                S = b["C"][k] @ Gam[j] @ b["C"][k].T + b["SV"][k]
                W = Gam[j] @ b["C"][k].T @ np.linalg.inv(S)
                nu = y1 - (b["C"][k] @ M[j] + b["D"][k]) @ y0
                acc += gpair[n, j, k] * (M[j] @ y0 + W @ nu)
            out[n, j] = acc.ravel() / gpair[n, j].sum()
    return out


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------
def _med_max(vals):
    return f"median {np.median(vals):.1e}  max {np.max(vals):.1e}"


def exp_ab_gauge():
    """S1(AB), S3, S4 and the S2 discriminant, on the AB gauge (C=0.4)."""
    p = ab_model(0.4, dB=0.0)
    M, Gam, c = _slaving(p)
    g_reg, g_cgm, g_cgv, g_naive, g_gain, g_rw = [], [], [], [], [], []
    exact_by_seed = {}
    for sd in range(N_SEEDS):
        _, _, ys = _simulate(p, N_STEPS, seed=SEED0 + sd)
        Ex_s, Vx_s, gam_ex = exact_path_smoother(p, ys)
        exact_by_seed[sd] = (ys, Ex_s, Vx_s, gam_ex)
        # S1(AB): smoothed regime posterior from the (R,Y) forward-backward
        gam_h, _ = _forward_backward(_alpha1(p, ys), kernel_Qab(p, ys))
        g_reg.append(_rel(gam_h, gam_ex))
        # S3: constant-gain smoother, mean and variance
        Ex_cg, Vx_cg = cg_smoother(p, ys)
        g_cgm.append(_rel(Ex_cg, Ex_s))
        g_cgv.append(_rel(Vx_cg, Vx_s))
        # S4: naive smoothed read-out fed the EXACT smoothed regime posterior
        yv = np.asarray(ys, float).reshape(-1, p.s)
        Ex_naive = np.array(
            [
                sum(
                    gam_ex[n, k] * (M[k] @ yv[n].reshape(p.s, 1) + c[k]).ravel() for k in range(p.K)
                )
                for n in range(N_STEPS)
            ]
        )
        g_naive.append(_rel(Ex_naive, Ex_s))
        Ex_f, _, _ = exact_mixture_filter(p, ys)
        g_gain.append(_rel(Ex_f, Ex_s))
        # S2 discriminant: the {C==0} reweighting formula on the wrong branch
        Ex_rw, _ = reweight_smoother(p, ys, kernel_Qab(p, ys))
        g_rw.append(_rel(Ex_rw, Ex_s))
    print("AB gauge (C=0.4, full AB):")
    print(f"  S1  (R,Y) FB regime smoother, kernel Q : {_med_max(g_reg)}")
    print(f"  S3  constant-gain smoother, mean       : {_med_max(g_cgm)}")
    print(f"  S3  constant-gain smoother, variance   : {_med_max(g_cgv)}")
    print(
        f"  S4  naive smoothed read-out (trap)     : median"
        f" {np.median(g_naive):.2f}"
        f"  (smoothing gain scale {np.median(g_gain):.2f})"
    )
    print(
        f"  S2d reweighting formula under AB       : median"
        f" {np.median(g_rw):.2f}  (expected O(1) failure)"
    )
    return exact_by_seed


def exp_c0_gauge():
    """S1({C==0}) and S2, on the CGOMSM1 gauge with state memory."""
    p = cgo_memory_model()
    g_reg, g_m, g_v = [], [], []
    exact_by_seed = {}
    for sd in range(N_SEEDS):
        _, _, ys = _simulate(p, N_STEPS, seed=SEED0 + sd)
        Ex_s, Vx_s, gam_ex = exact_path_smoother(p, ys)
        exact_by_seed[sd] = (ys, Ex_s, Vx_s, gam_ex)
        Ker = kernel_Q0(p, ys)
        gam_h, _ = _forward_backward(_alpha1(p, ys), Ker)
        g_reg.append(_rel(gam_h, gam_ex))
        Ex_r, Vx_r = reweight_smoother(p, ys, Ker)
        g_m.append(_rel(Ex_r, Ex_s))
        g_v.append(_rel(Vx_r, Vx_s))
    print("{C==0} gauge (memory A=(0.7,0.4)):")
    print(f"  S1  (R,Y) FB regime smoother, kernel Q0: {_med_max(g_reg)}")
    print(f"  S2  reweighting smoother, mean         : {_med_max(g_m)}")
    print(f"  S2  reweighting smoother, variance     : {_med_max(g_v)}")
    return exact_by_seed


def exp_lag1(exact_ab):
    """S5: lag-1 smoother on {A==MC} \\ AB, and its reduction on AB."""
    pA = ab_model(0.4, dB=0.15)
    pab = ab_model(0.4, dB=0.0)
    g_m, g_v, g_abm, g_abv = [], [], [], []
    exact_by_seed = {}
    for sd in range(N_SEEDS):
        _, _, ys = _simulate(pA, N_STEPS, seed=SEED0 + sd)
        Ex_s, Vx_s, gam_ex = exact_path_smoother(pA, ys)
        exact_by_seed[sd] = (ys, Ex_s, Vx_s, gam_ex)
        Ex_l, Vx_l = lag1_smoother(pA, ys)
        g_m.append(_rel(Ex_l, Ex_s))
        g_v.append(_rel(Vx_l, Vx_s))
        ys2, Ex_s2, Vx_s2, _ = exact_ab[sd]
        Ex_l2, Vx_l2 = lag1_smoother(pab, ys2)
        g_abm.append(_rel(Ex_l2, Ex_s2))
        g_abv.append(_rel(Vx_l2, Vx_s2))
    print("{A==MC} \\ AB gauge (C=0.4, B=MD+0.15):")
    print(f"  S5  lag-1 smoother, mean               : {_med_max(g_m)}")
    print(f"  S5  lag-1 smoother, variance           : {_med_max(g_v)}")
    print(f"  S5  reduction on AB, mean              : {_med_max(g_abm)}")
    print(f"  S5  reduction on AB, variance          : {_med_max(g_abv)}")
    return exact_by_seed


def exp_rts(exact_ab, exact_c0, exact_A):
    """S7: pair-RTS gain degeneracy and asymmetric exactness."""
    print("S7  pair-RTS (mode-matched, GPB2 forward bank):")
    gauges = [
        ("AB gauge          ", ab_model(0.4, dB=0.0), exact_ab, "slaved"),
        ("{A==MC} \\ AB gauge", ab_model(0.4, dB=0.15), exact_A, "slaved"),
        ("{C==0} gauge      ", cgo_memory_model(), exact_c0, "c0"),
    ]
    for label, p, exact, branch in gauges:
        errs, r_gx, r_w, r_c0 = [], [], [], []
        for sd in range(N_SEEDS):
            ys, Ex_s, _, _ = exact[sd]
            Ex_r, res = pair_rts_smoother(p, ys, with_residuals=True)
            errs.append(_rel(Ex_r, Ex_s))
            r_gx.append(res["gx"])
            r_w.append(res["w"])
            r_c0.append(res["c0"])
        if branch == "slaved":
            print(f"  {label}: mean {_med_max(errs)}")
            print(
                f"      gain identity G==[0,W_jk]: |G_x|/|G_y| max"
                f" {np.max(r_gx):.1e}, |G_y-W|/|W| max {np.max(r_w):.1e}"
            )
        else:
            print(
                f"  {label}: mean median {np.median(errs):.2e}"
                f"  [{np.min(errs):.0e},{np.max(errs):.0e}]  <- BIASED"
            )
            print(
                f"      gain identity G_y==-G_x M_k: residual max"
                f" {np.max(r_c0):.1e}  (the gain is degenerate; the bias"
                " is the k-conditional backward input)"
            )


def equal_dynamics_c0_model():
    """S9 control: the CGOMSM1 gauge with EQUAL state dynamics A_1 = A_2, which
    switches off the mode-dependent backward-Jacobian defect of the
    HBH-structure two-filter while keeping (G2) (SV contrast 0.20/0.60)."""
    return _params(
        A=[0.55, 0.55],
        B=[0.10, 0.10],
        C=[0.0, 0.0],
        D=[0.50, 0.50],
        SU=[0.40, 0.35],
        Dt=[0.15, -0.20],
        SV=[0.20, 0.60],
        p_switch=0.10,
    )


def exp_hbh(exact_c0, exact_A):
    """S9: the HBH-structure two-filter smoother -- structural inapplicability
    on AB, measured gap to the exact smoother where it runs, and the
    equal-dynamics control isolating the backward-Jacobian defect."""
    print("S9  HBH-structure two-filter (backward-time IMM + fusion):")
    pab = ab_model(0.4, dB=0.0)
    dets = [float(np.linalg.det(pab.f_matrix.F(k))) for k in range(pab.K)]
    print(
        f"  AB gauge          : det F_r = {dets[0]:.2e}, {dets[1]:.2e}"
        "  -> backward-time filter inapplicable (F_r singular on AB)"
    )
    p_eq = equal_dynamics_c0_model()
    exact_eq = {}
    for sd in range(N_SEEDS):
        _, _, ys = _simulate(p_eq, N_STEPS, seed=SEED0 + sd)
        Ex_s, Vx_s, gam_ex = exact_path_smoother(p_eq, ys)
        exact_eq[sd] = (ys, Ex_s, Vx_s, gam_ex)
    for label, p, exact in (
        ("{C==0} gauge      ", cgo_memory_model(), exact_c0),
        ("{C==0}, equal A   ", p_eq, exact_eq),
        ("{A==MC} \\ AB gauge", ab_model(0.4, dB=0.15), exact_A),
    ):
        g_m, g_r = [], []
        for sd in range(N_SEEDS):
            ys, Ex_s, _, gam_ex = exact[sd]
            Ex_h, mu_h = hbh_smoother(p, ys)
            g_m.append(_rel(Ex_h, Ex_s))
            g_r.append(_rel(mu_h, gam_ex))
        print(
            f"  {label}: mean median {np.median(g_m):.2e}"
            f"  [{np.min(g_m):.0e},{np.max(g_m):.0e}]"
            f"   regime median {np.median(g_r):.2e}"
        )
    g_m = []
    for sd in range(N_SEEDS):
        ys, Ex_s, _, _ = exact_c0[sd]
        Ex_h, _ = hbh_smoother(cgo_memory_model(), ys, variant="mixed")
        g_m.append(_rel(Ex_h, Ex_s))
    print(f"  {{C==0}}, alt. n-smoother variant (Sec. V): mean median {np.median(g_m):.2e}")


def near_c0_model(eta):
    """S6b gauge: the CGOMSM1 gauge with the channel switched on -- C = eta at
    both regimes, full state memory A = (0.7, 0.4) != MC: the dual approach
    direction, toward the autonomy branch."""
    return _params(
        A=[0.7, 0.4],
        B=[0.10, 0.10],
        C=[eta, eta],
        D=[0.50, 0.50],
        SU=[0.40, 0.35],
        Dt=[0.15, -0.20],
        SV=[0.20, 0.60],
        p_switch=0.10,
    )


def exp_off_domain2(etas=(0.02, 0.08, 0.3)):
    """S6b: off-domain degradation on the dual gauge (toward autonomy):
    smoothed regime posterior, pair-RTS state mean, reweighting state mean."""
    print("S6b off-domain, dual gauge (C=eta at both regimes, memory A=(0.7,0.4)):")
    m_reg, m_rts, m_rw = [], [], []
    for eta in etas:
        p = near_c0_model(eta)
        e_reg, e_rts, e_rw = [], [], []
        for sd in range(N_SEEDS):
            _, _, ys = _simulate(p, N_STEPS, seed=SEED0 + sd)
            Ex_s, _, gam_ex = exact_path_smoother(p, ys)
            e_reg.append(_rel(gpb2_regime_smoother(p, ys), gam_ex))
            e_rts.append(_rel(pair_rts_smoother(p, ys), Ex_s))
            E_rw, _ = reweight_smoother(p, ys, kernel_Q0(p, ys))
            e_rw.append(_rel(E_rw, Ex_s))
        m_reg.append(np.median(e_reg))
        m_rts.append(np.median(e_rts))
        m_rw.append(np.median(e_rw))
        print(
            f"  eta={eta:<5} regime {m_reg[-1]:.2e}"
            f"  RTS mean {m_rts[-1]:.2e}  reweight mean {m_rw[-1]:.2e}"
        )
    x = np.log10(etas)
    sl = [np.polyfit(x, np.log10(v), 1)[0] for v in (m_reg, m_rts, m_rw)]
    print(f"  slopes: regime {sl[0]:.2f}  RTS mean {sl[1]:.2f}  reweight mean {sl[2]:.2f}")


def exp_mixed(A0s=(0.0, 0.4, 0.8), C1=0.7):
    """S10: mixed-family smoothing -- uniformity inherited, asymmetry too."""
    print(f"S10 mixed branches (r1: C=0, memory A0; r2: C1={C1}, A=MC, B!=MD):")
    for A0 in A0s:
        p = mixed_branch_model(A0, C1)
        e_lag, e_rts, e_reg, e_fm, e_fp = [], [], [], [], []
        for sd in range(N_SEEDS):
            _, _, ys = _simulate(p, N_STEPS, seed=SEED0 + sd)
            Ex_s, _, gam_ex = exact_path_smoother(p, ys)
            E_lag, _ = lag1_smoother(p, ys)
            e_lag.append(_rel(E_lag, Ex_s))
            e_rts.append(_rel(pair_rts_smoother(p, ys), Ex_s))
            e_reg.append(_rel(gpb2_regime_smoother(p, ys), gam_ex))
            Ex_f, _, pi_f = exact_mixture_filter(p, ys)
            ex_g, _, pi_g, _ = gpb2_filter(p, ys)
            e_fm.append(_rel(ex_g, Ex_f))
            e_fp.append(_rel(pi_g, pi_f))
        print(
            f"  A0={A0:<4} smoothed: lag-1 {np.median(e_lag):.1e}"
            f"  pair-RTS {np.median(e_rts):.1e}"
            f"  regime {np.median(e_reg):.1e}"
            f" | GPB2 filtered: mean {np.median(e_fm):.1e}"
            f"  post {np.median(e_fp):.1e}"
        )


def exp_matrix_cross_annihilation():
    """S11: matrix cross-annihilation gauge -- regime smoother exact, no state
    smoother of this paper exact, GPB2 filter exact (baseline)."""
    print(
        "S11 matrix case q=2 (E10 gauge): cross-annihilation C_k N_j = 0 vs"
        " rotated rows (C_k N_j != 0):"
    )
    out = {}
    for tag, ann in (("C_k N_j = 0 ", True), ("C_k N_j != 0", False)):
        p = cross_annihilation_model(annihilate=ann)
        e = {k: [] for k in ("regime", "rts", "lag1", "reweight", "filt")}
        for sd in range(N_SEEDS):
            _, _, ys = _simulate(p, N_STEPS, seed=SEED0 + sd)
            Ex_s, _, gam_ex = exact_path_smoother(p, ys)
            e["regime"].append(_rel(gpb2_regime_smoother(p, ys), gam_ex))
            e["rts"].append(_rel(pair_rts_smoother(p, ys), Ex_s))
            e["lag1"].append(_rel(lag1_smoother(p, ys)[0], Ex_s))
            e["reweight"].append(_rel(reweight_smoother(p, ys, kernel_Q0(p, ys))[0], Ex_s))
            Ex_f, _, _ = exact_mixture_filter(p, ys)
            e["filt"].append(_rel(gpb2_filter(p, ys)[0], Ex_f))
        out[ann] = {k: _med_max(v) for k, v in e.items()}
        print(
            f"  {tag}  smoothed regime {np.median(e['regime']):.1e}"
            f" | smoothed mean: pair-RTS {np.median(e['rts']):.1e}"
            f"  lag-1 {np.median(e['lag1']):.1e}"
            f"  reweighting {np.median(e['reweight']):.1e}"
            f" | GPB2 filtered mean {np.median(e['filt']):.1e}"
        )
    return out


def exp_separation(etas=(0.02, 0.08, 0.3)):
    """S8: off the union, the three formulations separate; the pairwise
    separations vanish with eta and need no ground truth (diagnostic)."""
    print("S8  off the union (C=0.4, A=MC+eta): separation of formulations")
    sep_rc, sep_rw, sep_cw = [], [], []
    err_rts, err_cg, err_rw = [], [], []
    for eta in etas:
        p = off_union_model(0.4, eta)
        e_rts, e_cg, e_rw, s_rc, s_rw_, s_cw = [], [], [], [], [], []
        for sd in range(N_SEEDS):
            _, _, ys = _simulate(p, N_STEPS, seed=SEED0 + sd)
            Ex_s, _, _ = exact_path_smoother(p, ys)
            scale = np.max(np.abs(Ex_s)) + 1e-12
            E_rts = pair_rts_smoother(p, ys)
            E_cg, _ = cg_smoother(p, ys)
            E_rw, _ = reweight_smoother(p, ys, kernel_Qab(p, ys))
            e_rts.append(_rel(E_rts, Ex_s))
            e_cg.append(_rel(E_cg, Ex_s))
            e_rw.append(_rel(E_rw, Ex_s))
            s_rc.append(float(np.max(np.abs(E_rts - E_cg)) / scale))
            s_rw_.append(float(np.max(np.abs(E_rts - E_rw)) / scale))
            s_cw.append(float(np.max(np.abs(E_cg - E_rw)) / scale))
        sep_rc.append(np.median(s_rc))
        sep_rw.append(np.median(s_rw_))
        sep_cw.append(np.median(s_cw))
        err_rts.append(np.median(e_rts))
        err_cg.append(np.median(e_cg))
        err_rw.append(np.median(e_rw))
        print(
            f"  eta={eta:<5} err: RTS {err_rts[-1]:.2e}"
            f"  one-step {err_cg[-1]:.2e}"
            f"  reweight {err_rw[-1]:.2e}"
            f" | sep: RTS/1step {sep_rc[-1]:.2e}"
            f"  RTS/rw {sep_rw[-1]:.2e}  1step/rw {sep_cw[-1]:.2e}"
        )
    x = np.log10(etas)
    sl = [np.polyfit(x, np.log10(v), 1)[0] for v in (sep_rc, sep_rw, sep_cw)]
    print(f"  separation slopes: RTS/1step {sl[0]:.2f}  RTS/rw {sl[1]:.2f}  1step/rw {sl[2]:.2f}")
    sle = [np.polyfit(x, np.log10(v), 1)[0] for v in (err_rts, err_cg, err_rw)]
    print(
        f"  error slopes (state mean): RTS {sle[0]:.2f}"
        f"  one-step {sle[1]:.2f}  reweight {sle[2]:.2f}"
    )


def exp_off_domain(etas=(0.02, 0.08, 0.3), n_seeds=20):
    """S6: first-order degradation of the collapsed smoothed regime posterior
    off the union (C=0.4, A=MC+eta), against the exact smoothed regime."""
    print("off the union (C=0.4, A=MC+eta): collapsed smoothed regime")
    meds = []
    for eta in etas:
        p = off_union_model(0.4, eta)
        errs = []
        for sd in range(n_seeds):
            _, _, ys = _simulate(p, N_STEPS, seed=SEED0 + sd)
            errs.append(_rel(gpb2_regime_smoother(p, ys), exact_regime_smoother(p, ys)))
        meds.append(np.median(errs))
        print(f"  S6  eta={eta:<5} median {meds[-1]:.2e}")
    slope = np.polyfit(np.log10(etas), np.log10(meds), 1)[0]
    print(f"  S6  fitted slope: {slope:.2f}  (first order; filtered E3 is ~3)")
    return meds, slope


S12_SHAPES = ((1, 1), (2, 1), (2, 2))
S12_N = 8  # 2^8 = 256 paths per exact evaluation
S12_SEEDS = (1, 2, 3)


def _inv_or_range(vals):
    """'<= max' when every value is at machine precision, else 'min-max'."""
    vals = np.asarray(vals)
    if vals.max() < 1e-12:
        return f"<= {vals.max():.1e}"
    return f"{vals.min():.2g}-{vals.max():.2g}"


def _perturbed(model, ys, rng, *blocks):
    """Exact quantities after standard normal additions on the observation
    blocks ys[a:b], all other observations fixed."""
    y = ys.copy()
    for a, b in blocks:
        y[a:b] += rng.normal(size=y[a:b].shape)
    return exact_pair_quantities(model, y)


def exp_time_reversal():
    """S12: the two exact smoothers as mirror filters -- dependence table,
    one-step backward summary, reversed transitions, mixture caveat, the
    [Abb15] pair-dependent model, and the K=1 memory mirror."""
    rng = np.random.default_rng(SEED0 + 12)
    N, n0 = S12_N, 4
    fams = ("C==0", "C==0, A==0", "AB")
    rows = (
        ("x_n(r_n)        vs y_{1:n-1}       ", "past"),
        ("x_n(r_n)        vs y_{n+1:N}       ", "future"),
        ("x_n(r_{n-1:n})  vs all but y_{n-1:n}", "pair_prev"),
        ("x_n(r_{n:n+1})  vs all but y_{n:n+1}", "pair_next"),
    )
    tab = {(f, r): [] for f in fams for _, r in rows}
    onestep, models = [], []
    for q, s in S12_SHAPES:
        fam_models = {
            "C==0": random_cgomsm1(rng, 2, q, s, memory=True),
            "C==0, A==0": random_cgomsm1(rng, 2, q, s, memory=False),
            "AB": random_cgomsm2(rng, 2, q, s),
        }
        models.append(fam_models)
        for f, m in fam_models.items():
            for sd in S12_SEEDS:
                ys = simulate_pair(m, N, SEED0 + sd)
                e = exact_pair_quantities(m, ys)
                e_past = _perturbed(m, ys, rng, (0, n0))
                e_fut = _perturbed(m, ys, rng, (n0 + 1, N))
                e_prev = _perturbed(m, ys, rng, (0, n0 - 1), (n0 + 1, N))
                e_next = _perturbed(m, ys, rng, (0, n0), (n0 + 2, N))
                tab[f, "past"].append(_rel(e_past["smo_reg"][n0], e["smo_reg"][n0]))
                tab[f, "future"].append(_rel(e_fut["smo_reg"][n0], e["smo_reg"][n0]))
                tab[f, "pair_prev"].append(
                    _rel(e_prev["pair_next"][n0 - 1], e["pair_next"][n0 - 1])
                )
                tab[f, "pair_next"].append(_rel(e_next["pair_cur"][n0], e["pair_cur"][n0]))
                if f == "AB":
                    bs = one_step_backward_summary(m, ys, e["gpair"])
                    onestep.append(_rel(bs[1:], e["smo_reg"][1 : N - 1]))
    print(
        f"S12 time reversal (random models, K=2, (q,s) in {list(S12_SHAPES)}, N={N},"
        f" {len(S12_SEEDS)} seeds): relative change of an exact smoothed mean"
    )
    print(f"  {'mean / perturbed block':38s}" + "".join(f"{f:>16s}" for f in fams))
    for label, r in rows:
        print(f"  {label:38s}" + "".join(f"{_inv_or_range(tab[f, r]):>16s}" for f in fams))
    print(f"  AB: per-regime smoothed mean = one-step backward summary: max {max(onestep):.1e}")

    # reversed transitions (first transition, where the Gaussian reversal is exact)
    c_rev = [pair_block_max(time_reversal(fm["AB"]), "C") for fm in models]
    a_rev = [pair_block_max(time_reversal(fm["AB"]), "A") for fm in models]
    st = {f: [reversed_regime_state_gap(fm[f]) for fm in models] for f in fams}
    ab_rev = {f: [pair_ab_residual(time_reversal(fm[f])) for fm in models] for f in fams[:2]}
    print(
        f"  reversal of AB: max|C~| {max(c_rev):.1e}  max|A~| {max(a_rev):.1e};"
        f" regime transition vs hidden state {_inv_or_range(st['AB'])}"
    )
    print(
        f"  reversal of C==0: regime transition vs hidden state {_inv_or_range(st['C==0'])}"
        f" (A==0: {_inv_or_range(st['C==0, A==0'])}); AB residual"
        f" {_inv_or_range(ab_rev['C==0'])} (A==0: {_inv_or_range(ab_rev['C==0, A==0'])})"
    )

    # mixture caveat: filter of the Gaussian reversal vs exact per-regime smoother
    for K in (1, 2):
        gaps = []
        for i, (q, s) in enumerate(S12_SHAPES):
            m = random_cgomsm2(rng, 1, q, s) if K == 1 else models[i]["AB"]
            rv = time_reversal(m)
            for sd in S12_SEEDS:
                ys = simulate_pair(m, N, SEED0 + sd)
                e, er = exact_pair_quantities(m, ys), exact_pair_quantities(rv, ys[::-1])
                gaps.append(_rel(er["filt_reg"][::-1][1:], e["smo_reg"][1:]))
        print(f"  Gaussian reversal, K={K}: reversed filter vs smoother, max {max(gaps):.1e}")

    # the [Abb15] pair-dependent CGOMSM1 matched to AB (Remark rem:pairmodel)
    g_reg, g_filt, g_naive, g_gap = [], [], [], []
    for fm in models:
        pair, m2 = abb15_pair_model(fm["AB"])
        M, _ = _pair_slaving(m2)
        gaps = []
        for sd in S12_SEEDS:
            ys = simulate_pair(m2, N, SEED0 + sd)
            e2, ep = exact_pair_quantities(m2, ys), exact_pair_quantities(pair, ys)
            naive = np.array(
                [
                    sum(e2["pi"][n, k] * (M[k] @ ys[n].reshape(-1, 1)).ravel() for k in range(2))
                    for n in range(N)
                ]
            )
            g_reg.append(_rel(ep["pi"], e2["pi"]))
            g_filt.append(_rel(ep["Exf"][1:], e2["Exf"][1:]))
            g_naive.append(_rel(ep["Ex"][1:], naive[1:]))
            gaps.append(_rel(ep["Ex"][1:-1], e2["Ex"][1:-1]))
        g_gap.append(np.median(gaps))
    print(
        f"  [Abb15] pair model vs AB: smoothed regimes {max(g_reg):.1e}, filter"
        f" {max(g_filt):.1e}; its smoother = naive read-out {max(g_naive):.1e};"
        f" gap to the AB smoother (medians) {min(g_gap):.2f}-{max(g_gap):.2f}"
    )

    # K=1 memory mirror (Remark rem:memory-mirror): stationary law only
    F = np.array([[0.5, 0.0], [0.5, 0.3]])
    for tag, S0 in (
        ("stationary S_Z1 = I     ", np.eye(2)),
        ("S_Z1 = diag(2, 0.5)    ", np.diag([2.0, 0.5])),
    ):
        m = dict(
            K=1,
            q=1,
            s=1,
            F=[[F]],
            SW=[[np.eye(2) - F @ F.T]],
            P=np.array([[1.0]]),
            pi0=np.array([1.0]),
            S0=[S0],
        )
        ys = simulate_pair(m, 10, 3)
        e = exact_pair_quantities(m, ys)
        yp, yf = ys.copy(), ys.copy()
        yp[:5] += 1.0  # past y_{1:n-1}, n = 6
        yf[7:] += 1.0  # far future y_{n+2:N}
        ep, ef = exact_pair_quantities(m, yp), exact_pair_quantities(m, yf)
        print(
            f"  K=1 memory mirror, {tag}: filter vs past {_rel(ep['Exf'][5], e['Exf'][5]):.1e},"
            f" smoother vs far future {_rel(ef['Ex'][5], e['Ex'][5]):.2g}"
        )


S13_N_K2 = 6  # the K^2 embedding has 4^N paths -- keep its records short


def _hbh_or_nan(params, ys, Ex_s, gam_ex):
    """HBH errors (state, regime); NaN where the scheme is inapplicable
    (singular pair dynamics, e.g. A = 0 or AB)."""
    try:
        Ex_h, mu_h = hbh_smoother(params, ys)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    return _rel(Ex_h, Ex_s), _rel(mu_h, gam_ex)


def exp_classical_smoothers():
    """S13: exactness domains of the classical collapsed smoothers (Kim,
    IMM-RTS, pair-RTS on both banks, HBH), on the scalar gauges and on random
    matrix models of each family, and the K^2 embedding of Section V-E of the
    exactness paper (filter transported, smoother not), and the one-step
    collapsed smoother that reaches the domain of its forward filter."""
    gauges = (
        ("AB (C=0.4)        ", lambda: ab_model(0.4, dB=0.0)),
        ("{A==MC} \\ AB      ", lambda: ab_model(0.4, dB=0.15)),
        ("{C==0}, memory    ", cgo_memory_model),
        ("{C==0}, equal A   ", equal_dynamics_c0_model),
        ("{C==0, A==0}      ", memoryless_c0_model),
        ("cross-annih. q=2  ", lambda: cross_annihilation_model(annihilate=True)),
        ("off union (eps=.08)", lambda: off_union_model(0.4, 0.08)),
        ("mixed (A0=0.4)    ", lambda: mixed_branch_model(0.4, 0.7)),
    )
    labels = [lab for lab, _, _ in CLASSICAL_SMOOTHERS] + ["HBH two-filter (S9)"]
    print("S13a classical collapsed smoothers, scalar gauges -- median state | regime error:")
    print("    " + " " * 20 + "".join(f"{lab[:26]:>28s}" for lab in labels))
    table = {}
    for name, mk in gauges:
        p = mk()
        ex = {lab: [] for lab in labels}
        rg = {lab: [] for lab in labels}
        for sd in range(N_SEEDS):
            _, _, ys = _simulate(p, N_STEPS, seed=SEED0 + sd)
            Ex_s, _, gam_ex = exact_path_smoother(p, ys)
            for lab, bank, w in CLASSICAL_SMOOTHERS:
                Ex, gam = collapsed_rts_smoother(p, ys, bank, w)
                ex[lab].append(_rel(Ex, Ex_s))
                rg[lab].append(_rel(gam, gam_ex))
            eh = _hbh_or_nan(p, ys, Ex_s, gam_ex)
            ex[labels[-1]].append(eh[0])
            rg[labels[-1]].append(eh[1])
        table[name.strip()] = {lab: (np.median(ex[lab]), np.median(rg[lab])) for lab in labels}
        cells = "".join(
            f"{'n/a (singular F)':>28s}"
            if np.isnan(ex[lab][0])
            else f"{np.median(ex[lab]):>19.1e} |{np.median(rg[lab]):>7.0e}"
            for lab in labels
        )
        print(f"    {name:20s}{cells}")

    rng = np.random.default_rng(SEED0 + 13)
    print("S13b random models (6 per row), max over models of max(state, regime) error:")
    for fam in ("C0A0", "C0mem", "AMC", "AB"):
        for K, q, s, N in ((2, 2, 1, 9), (2, 2, 2, 9), (3, 2, 1, 6)):
            err = {lab: [] for lab in labels}
            for sd in range(6):
                p = random_family_params(rng, K, q, s, fam)
                _, _, ys = _simulate(p, N, seed=SEED0 + 1300 + sd)
                Ex_s, _, gam_ex = exact_path_smoother(p, ys)
                for lab, bank, w in CLASSICAL_SMOOTHERS:
                    Ex, gam = collapsed_rts_smoother(p, ys, bank, w)
                    err[lab].append(max(_rel(Ex, Ex_s), _rel(gam, gam_ex)))
                err[labels[-1]].append(max(_hbh_or_nan(p, ys, Ex_s, gam_ex)))
            cells = "".join(
                f"{'n/a':>14s}" if np.all(np.isnan(err[lab])) else f"{np.nanmax(err[lab]):>14.1e}"
                for lab in labels
            )
            print(f"    {fam:6s} K={K} q={q} s={s}{cells}")

    print(
        f"S13c K^2 embedding of a slaved AB model (Section V-E), N={S13_N_K2},"
        " 5 records per model -- median (max):"
    )
    rows = (
        ("K^2-mode IMM filter  vs CGOMSM2 exact filter", "filt"),
        ("K^2 smoothed regimes vs CGOMSM2 smoothed regimes", "reg"),
        ("Kim, K^2 modes       vs K^2 model exact smoother", "kim2"),
        ("IMM-RTS, K^2 modes   vs K^2 model exact smoother", "imm2"),
        ("K^2 model exact smoother vs CGOMSM2 exact smoother", "gap"),
        ("IMM-RTS, K modes     vs CGOMSM2 exact smoother", "immK"),
        ("smoothing gain |filtered - smoothed| (scale)", "gain"),
    )
    out = {}
    for name, mk in (
        ("AB gauge", lambda: ab_model(0.4, dB=0.0)),
        ("random AB K=2 q=2 s=1", lambda: random_family_params(rng, 2, 2, 1, "AB")),
        ("random AB K=2 q=2 s=2", lambda: random_family_params(rng, 2, 2, 2, "AB")),
    ):
        p = slaved_init(mk())
        p2 = k2_embedding(p)
        e = {key: [] for _, key in rows}
        for sd in range(5):
            _, _, ys = _simulate(p, S13_N_K2, seed=SEED0 + 1350 + sd)
            Ex_f, _, _ = exact_mixture_filter(p, ys)
            zs2, _, al2, _ = imm_pair_bank(p2, ys)
            e["filt"].append(_rel(np.einsum("nk,nkq->nq", al2, zs2[:, :, : p.q, 0]), Ex_f))
            Ex_s, _, gam = exact_path_smoother(p, ys)
            Ex_s2, _, gam2 = exact_path_smoother(p2, ys)
            e["reg"].append(_rel(gam2.reshape(-1, p.K, p.K).sum(axis=1), gam))
            e["kim2"].append(_rel(collapsed_rts_smoother(p2, ys, "gpb2", "kim")[0], Ex_s2))
            e["imm2"].append(_rel(collapsed_rts_smoother(p2, ys, "imm", "kim")[0], Ex_s2))
            e["gap"].append(_rel(Ex_s2, Ex_s))
            e["immK"].append(_rel(collapsed_rts_smoother(p, ys, "imm", "kim")[0], Ex_s))
            e["gain"].append(_rel(Ex_f, Ex_s))
        out[name] = {key: (np.median(v), np.max(v)) for key, v in e.items()}
        print(f"  {name}:")
        for lab, key in rows:
            print(f"    {lab:52s} {np.median(e[key]):.1e} ({np.max(e[key]):.1e})")

    rng4 = np.random.default_rng(SEED0 + 1313)
    cases = (
        ("AB (C=0.4)", lambda: ab_model(0.4, dB=0.0), N_STEPS),
        ("{A==MC} \\ AB", lambda: ab_model(0.4, dB=0.15), N_STEPS),
        ("{C==0}, memory", cgo_memory_model, N_STEPS),
        ("cross-annih. q=2 (E10)", lambda: cross_annihilation_model(annihilate=True), N_STEPS),
        ("random X\\union K=2 q=3 s=1", lambda: random_cross_annihilation_params(rng4, 2, 3, 1), 8),
        ("random X\\union K=3 q=3 s=1", lambda: random_cross_annihilation_params(rng4, 3, 3, 1), 6),
        ("random X\\union K=2 q=3 s=2", lambda: random_cross_annihilation_params(rng4, 2, 3, 2), 8),
        ("random {C==0} K=2 q=2 s=2", lambda: random_family_params(rng4, 2, 2, 2, "C0mem"), 8),
        ("E10 rotated (C_k N_j != 0)", lambda: cross_annihilation_model(annihilate=False), N_STEPS),
        ("off union (eps=.08)", lambda: off_union_model(0.4, 0.08), N_STEPS),
        ("mixed (A0=0.4)", lambda: mixed_branch_model(0.4, 0.7), N_STEPS),
    )
    print(
        "S13d one-step collapsed smoother (no backward state recursion), 6 records,"
        " max of mean | variance | regime error:"
    )
    print(f"    {'':32s}{'GPB2 bank':>30s}{'IMM bank':>30s}")
    onestep = {}
    for name, mk, N in cases:
        e = {bank: ([], [], []) for bank in ("gpb2", "imm")}
        for sd in range(6):
            p = mk()
            _, _, ys = _simulate(p, N, seed=SEED0 + 2000 + sd)
            Ex_s, Vx_s, gam_ex = exact_path_smoother(p, ys)
            for bank in e:
                Ex, Vx, gam = onestep_collapsed_smoother(p, ys, bank)
                e[bank][0].append(_rel(Ex, Ex_s))
                e[bank][1].append(_rel(Vx, Vx_s))
                e[bank][2].append(_rel(gam, gam_ex))
        onestep[name] = {bank: tuple(max(v) for v in e[bank]) for bank in e}
        cells = "".join(
            f"{max(e[bank][0]):>11.1e} |{max(e[bank][1]):>7.0e} |{max(e[bank][2]):>7.0e}"
            for bank in ("gpb2", "imm")
        )
        print(f"    {name:32s}{cells}")
    return table, out, onestep


def main():
    print(f"[smoothing_exactness] N={N_STEPS}, {N_SEEDS} seeds, ground truth K^N path smoother\n")
    exact_ab = exp_ab_gauge()
    print()
    exact_c0 = exp_c0_gauge()
    print()
    exact_A = exp_lag1(exact_ab)
    print()
    exp_off_domain()
    print()
    exp_off_domain2()
    print()
    exp_rts(exact_ab, exact_c0, exact_A)
    print()
    exp_separation()
    print()
    exp_hbh(exact_c0, exact_A)
    print()
    exp_mixed()
    print()
    exp_matrix_cross_annihilation()
    print()
    exp_time_reversal()
    print()
    exp_classical_smoothers()


if __name__ == "__main__":
    main()
