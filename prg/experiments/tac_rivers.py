#!/usr/bin/env python3
"""Section V-C of the TAC paper (case ``d`` of ``tac_simulations``): river discharge, USGS NWIS.

Three river reaches of the eastern United States, 2014-2023, 3 h steps:
    X_n = log discharge at the downstream gauge (hidden state; its record is the ground truth),
    Y_n = log discharge at the upstream gauge, delayed by the travel time tau of the reach,
    u_n = (log of the summed discharge of the gauged tributaries entering the reach, delayed by a lag,
           1): a measured exogenous input (Remark 5), acting on the transition n-1 -> n,
    R_n = rising (1) / falling (0) limb of the downstream hydrograph over the last 6 h (labelled for
          training only, hidden in the filters), K = 2.
tau and the input lag are chosen on the training years only. Two splits: train 2014-2019 / test
2020-2023 and train 2018-2023 / test 2014-2017. Every configuration uses the same samples, so that the
comparisons (with / without input, between filters) are paired.

Models, fitted by maximum likelihood with the labelled regimes (two least-squares regressions per
regime and family, as in case ``c``): CGOMSM1, CGOMSM1 with |A_k| <= 0.99, CGOMSM2, THGMSM (RBPF), and
a single-regime THGMSM (Kalman filter); the static regression of X_n on (Y_n, u_n) is a reference.
Scores on the test years: RMSE of X (log units), log-likelihood of Y per sample, accuracy of the MAP
regime; 95% intervals by a paired moving-block bootstrap (blocks of 8 days).

Data: U.S. Geological Survey, National Water Information System, instantaneous discharge (parameter
00060), doi:10.5066/F7P55KJN, public domain. The hourly means used here are cached in
``data/real/rivers/<site>.csv.gz`` (UTC); without the cache they are downloaded from
waterservices.usgs.gov (about 320 MB of raw files, not kept).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import urllib.request
from pathlib import Path

import numpy as np

from prg.experiments.tac_simulations import _lognorm, _lse, _ols

CACHE = Path("data/real/rivers")
YEARS = (2014, 2023)
STEP = "3h"
RISE_H = 6  # rising limb: X_t > X_{t - 6 h}
MAX_TAU = 48  # hours
LAGS = range(0, 49, 3)  # candidate input lags (hours)
MIN_LEN = 20  # minimal length of a contiguous run of complete samples
N_PART = 500
BLOCK_DAYS = 8
N_BOOT = 2000
AMAX = 0.99

REACHES = {
    "Susquehanna": dict(
        up="01570500",  # Susquehanna River at Harrisburg, PA
        dn="01576000",  # Susquehanna River at Marietta, PA
        trib=("01573560", "01571500", "01574000", "01573710"),
    ),
    "Delaware": dict(
        up="01446500",  # Delaware River at Belvidere, NJ
        dn="01463500",  # Delaware River at Trenton, NJ
        trib=("01454700", "01457000", "01459500", "01446775"),
    ),
    "Potomac": dict(
        up="01638500",  # Potomac River at Point of Rocks, MD
        dn="01646500",  # Potomac River near Washington DC, Little Falls pump station
        trib=("01643000", "01644000", "01644280", "01645000", "01646000"),
    ),
}
SPLITS = {"A": ("2014-01-01", "2019-12-31"), "B": ("2018-01-01", "2023-12-31")}  # training years

_URL = (
    "https://waterservices.usgs.gov/nwis/iv/?format=rdb&sites={site}&parameterCd=00060"
    "&startDT={a}-01-01&endDT={b}-12-31"
)
_TZ = {"EST": 5, "EDT": 4}


# ----------------------------------------------------------------------------- data
def _parse_rdb(text):
    import pandas as pd

    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    d = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t", dtype=str).iloc[1:]
    if d.empty:
        return pd.Series(dtype=float)
    qcol = [c for c in d.columns if c.endswith("_00060")][0]
    if not d["tz_cd"].isin(_TZ).all():
        raise ValueError(f"unexpected time zones {set(d['tz_cd']) - set(_TZ)}")
    t = pd.to_datetime(d["datetime"]) + pd.to_timedelta(d["tz_cd"].map(_TZ).astype(float), "h")
    q = pd.to_numeric(d[qcol], errors="coerce")  # ice, equipment malfunction... -> NaN
    return pd.Series(q.to_numpy(), index=pd.DatetimeIndex(t.to_numpy()))


def _download_hourly(site):
    import pandas as pd

    parts = []
    for a in range(YEARS[0], YEARS[1] + 1, 2):
        url = _URL.format(site=site, a=a, b=min(a + 1, YEARS[1]))
        print(f"downloading USGS {site} {a}-{a + 1} …", flush=True)
        with urllib.request.urlopen(url, timeout=300) as r:  # noqa: S310 (fixed https URL)
            parts.append(_parse_rdb(r.read().decode()))
    s = pd.concat(parts)
    s = s[~s.index.duplicated()].sort_index()
    return s.resample("1h").mean()  # hourly means, UTC, labelled at the start of the hour


def _write_cache(h, f):
    """Gzipped CSV with mtime 0, so that identical data give byte-identical caches."""
    buf = h.to_frame("q").to_csv(index_label="time_utc", float_format="%.6g")
    with open(f, "wb") as g, gzip.GzipFile(fileobj=g, mode="wb", mtime=0) as z:
        z.write(buf.encode())


def load_hourly(site, cache=CACHE):
    """Hourly mean discharge (cubic feet per second, UTC) of a USGS gauge, from the cache or downloaded."""
    import pandas as pd

    f = Path(cache) / f"{site}.csv.gz"
    if not f.exists():
        h = _download_hourly(site)
        f.parent.mkdir(parents=True, exist_ok=True)
        _write_cache(h, f)
    d = pd.read_csv(f, index_col=0, parse_dates=True)
    return d["q"].asfreq("1h")


def data_fingerprint(cache=CACHE):
    h = hashlib.sha256()
    for f in sorted(Path(cache).glob("*.csv.gz")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- frames
def _log_pos(s):
    return np.log(s.where(s > 0))


def travel_time(x, y, train):
    """Lag (hours) maximizing corr(dX_t, dY_{t - lag}) on the training years."""
    dx, dy = x.diff(), y.diff()
    c = [dx[train].corr(dy.shift(lag)[train]) for lag in range(MAX_TAU + 1)]
    return int(np.argmax(c))


def _step(h):
    return h.resample(STEP, label="right", closed="right")


def build_frame(reach, split, cache=CACHE):
    """3 h frame of a reach with (x, y, r, trib); tau and the input lag chosen on the training years."""
    import pandas as pd

    R = REACHES[reach]
    x, y = _log_pos(load_hourly(R["dn"], cache)), _log_pos(load_hourly(R["up"], cache))
    T = pd.concat([load_hourly(s, cache) for s in R["trib"]], axis=1)
    trib = _log_pos(T.sum(axis=1, min_count=T.shape[1]))  # all tributaries present
    idx = pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[1]}-12-31 23:00", freq="1h")
    x, y, trib = x.reindex(idx), y.reindex(idx), trib.reindex(idx)
    a, b = (pd.Timestamp(v) for v in SPLITS[split])
    train_h = (idx >= a) & (idx <= b + pd.Timedelta("23h"))
    tau = travel_time(x, y, train_h)
    h = pd.DataFrame({"x": x, "y": y.shift(tau)})
    h["r"] = (h["x"] > h["x"].shift(RISE_H)).astype(float)
    h.loc[h["x"].isna() | h["x"].shift(RISE_H).isna(), "r"] = np.nan
    d = _step(h[["x", "y"]]).mean()
    d["r"] = _step(h["r"]).last()
    train = (d.index > a) & (d.index <= b + pd.Timedelta("1D"))
    # input lag: largest reduction of the MSE of the static regression X_n ~ (Y_n, u_n, 1), training years
    gains = {}
    for lag in LAGS:
        u = _step(trib.shift(lag)).mean().reindex(d.index)
        D = pd.concat([d[["x", "y"]], u.rename("u")], axis=1)[train].dropna()
        _, m1 = _ols(D["x"].to_numpy(), np.c_[D["y"], D["u"], np.ones(len(D))])
        _, m0 = _ols(D["x"].to_numpy(), np.c_[D["y"], np.ones(len(D))])
        gains[lag] = 1.0 - m1 / m0
    lag = max(gains, key=gains.get)
    d["trib"] = _step(trib.shift(lag)).mean().reindex(d.index)
    return d, train, dict(tau_h=tau, input_lag_h=lag, input_static_mse_gain=gains[lag])


def sequences(d, mask, mu, sd, use_input):
    """Standardized runs of complete samples (the same samples with and without input)."""
    sub = d[mask]
    ok = np.isfinite(sub[["x", "y", "r", "trib"]].to_numpy(float)).all(axis=1)
    idx = np.flatnonzero(ok)
    out = []
    for run in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1):
        if len(run) < MIN_LEN:
            continue
        g = sub.iloc[run]
        u = (g["trib"].to_numpy() - mu["trib"]) / sd["trib"] if use_input else np.zeros(len(g))
        out.append(
            dict(
                x=(g["x"].to_numpy() - mu["x"]) / sd["x"],
                y=(g["y"].to_numpy() - mu["y"]) / sd["y"],
                u=np.c_[u, np.ones(len(g))],
                r=g["r"].to_numpy(int),
            )
        )
    return out


# ----------------------------------------------------------------------------- models
def fit(train, K, amax=None):
    """Per-regime fits with labelled regimes (same estimators as ``tac_simulations.fit_c``).

    Without input, u_n = (0, 1): the first column is identically 0 and its coefficient is set to 0.
    With ``amax``, the CGOMSM1 is fitted under |A_k| <= amax (A_k fixed at the bound, rest refitted).
    """
    P = np.ones((K, K))  # one pseudo-count per entry
    rows = {k: [] for k in range(K)}
    first = {k: [] for k in range(K)}
    for s in train:
        x, y, u = s["x"], s["y"], s["u"]
        r = s["r"] if K > 1 else np.zeros(len(x), int)
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
        xn, yn, un, xp, yp = D[:, 0], D[:, 1], D[:, 2:-2], D[:, -2], D[:, -1]
        Z1 = np.vstack(first[k])
        base = {"S1": Z1.T @ Z1 / len(Z1)}
        des = np.c_[xn, yn, un]
        bx, _ = _ols(xp, des)
        by, _ = _ols(yp, des)
        SW = np.cov(np.c_[xp - des @ bx, yp - des @ by].T, bias=True)
        mods["thgmsm"].append(
            dict(base, a=bx[0], b=bx[1], lx=bx[2:], c=by[0], d=by[1], ly=by[2:])
            | dict(su=SW[0, 0], rho=SW[0, 1], sv=SW[1, 1])
        )
        b1, sv = _ols(yp, np.c_[yn, un])
        d, ly = b1[0], b1[1:]
        b2, se = _ols(xp, np.c_[xn, yn, yp, un])
        if amax is not None and abs(b2[0]) > amax:
            a_c = float(np.sign(b2[0]) * amax)
            rest, se = _ols(xp - a_c * xn, np.c_[yn, yp, un])
            b2 = np.r_[a_c, rest]
        g = b2[2]
        mods["cgomsm1"].append(
            dict(base, a=b2[0], b=b2[1] + g * d, lx=b2[3:] + g * ly, c=0.0, d=d, ly=ly)
            | dict(su=se + g * g * sv, rho=g * sv, sv=sv)
        )
        b1, sv = _ols(yp, des)
        c, d, ly = b1[0], b1[1], b1[2:]
        b2, gam = _ols(xp, np.c_[yp, un])
        M = b2[0]
        mods["cgomsm2"].append(
            dict(base, a=M * c, b=M * d, lx=b2[1:] + M * ly, c=c, d=d, ly=ly)
            | dict(su=gam + M * M * sv, rho=M * sv, sv=sv)
        )
    return {n: {"K": K, "P": P, "p1": p1 / p1.sum(), "reg": v} for n, v in mods.items()}


def _init(m, y1):
    S = np.array([R["S1"] for R in m["reg"]])
    ll = np.log(m["p1"]) + _lognorm(y1, 0.0, S[:, 1, 1])
    return ll, S[:, 0, 1] / S[:, 1, 1] * y1, S[:, 0, 0] - S[:, 0, 1] ** 2 / S[:, 1, 1]


def cgo_filter(m, y, u, family):
    """Exact filter of a CGOMSM1 / CGOMSM2 with input (as ``tac_simulations.cgo_filter_c``).

    Returns E[X_n | y_{1:n}], p(r_n | y_{1:n}) and the per-step log-likelihood increments.
    """
    K, N, Rg = m["K"], len(y), m["reg"]
    ll0, mu, var = _init(m, y[0])
    inc = np.empty(N)
    inc[0] = _lse(ll0)
    pi = np.exp(ll0 - inc[0])
    E, PI = np.empty(N), np.empty((N, K))
    E[0], PI[0] = pi @ mu, pi
    for n in range(1, N):
        lk, mjk = np.empty((K, K)), np.empty((K, K))
        for k, R in enumerate(Rg):
            if family == "cgomsm1":
                my = R["d"] * y[n - 1] + R["ly"] @ u[n]
                lk[:, k] = _lognorm(y[n], my, R["sv"])
                mjk[:, k] = (
                    R["a"] * mu
                    + R["b"] * y[n - 1]
                    + R["lx"] @ u[n]
                    + R["rho"] / R["sv"] * (y[n] - my)
                )
            else:
                my = R["c"] * mu + R["d"] * y[n - 1] + R["ly"] @ u[n]
                lk[:, k] = _lognorm(y[n], my, R["c"] ** 2 * var + R["sv"])
                M = R["rho"] / R["sv"]
                mjk[:, k] = M * y[n] + (R["lx"] - M * R["ly"]) @ u[n]
        L = np.log(np.maximum(pi, 1e-300))[:, None] + np.log(m["P"]) + lk
        a = L.max()
        J = np.exp(L - a)
        inc[n] = a + np.log(J.sum())
        col = J.sum(axis=0)
        mu = (J / np.maximum(col, 1e-300)[None, :] * mjk).sum(axis=0)
        pi = col / col.sum()
        if family == "cgomsm2":
            var = np.array([R["su"] - R["rho"] ** 2 / R["sv"] for R in Rg])
        E[n], PI[n] = pi @ mu, pi
    return E, PI, inc


def rbpf(m, y, u, n_part, rng):
    """RBPF (optimal proposal) for a THGMSM with input; exact Kalman filter when K = 1."""
    K, N, Rg = m["K"], len(y), m["reg"]
    ll0, m0, v0 = _init(m, y[0])
    inc = np.empty(N)
    inc[0] = _lse(ll0)
    r = rng.choice(K, size=n_part, p=np.exp(ll0 - inc[0]))
    mu, Pv = m0[r], v0[r]
    w = np.full(n_part, 1.0 / n_part)
    E, PI = np.empty(N), np.empty((N, K))
    E[0], PI[0] = w @ mu, np.bincount(r, w, K)
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
        inc[n] = _lse(logw)
        ch = np.minimum(
            (rng.random(n_part)[None, :] > np.cumsum(pj / tot, axis=0)).sum(axis=0), K - 1
        )
        mu = np.choose(ch, [c[1] for c in cand])
        Pv = np.choose(ch, [c[2] for c in cand])
        r = ch
        w = np.exp(logw - logw.max())
        w /= w.sum()
        E[n], PI[n] = w @ mu, np.bincount(r, w, K)
        if 1.0 / np.sum(w**2) < n_part / 2:  # systematic resampling
            v = (rng.random() + np.arange(n_part)) / n_part
            idx = np.minimum(np.searchsorted(np.cumsum(w), v), n_part - 1)
            mu, Pv, r = mu[idx], Pv[idx], r[idx]
            w = np.full(n_part, 1.0 / n_part)
    return E, PI, inc


# ----------------------------------------------------------------------------- experiment
FILTERS = ("cgomsm1", "cgomsm1_amax", "cgomsm2", "thgmsm_rbpf", "kalman_K1", "static")


def run_filters(train, test, sx, sy, seed=0):
    """Per-sample squared errors of X (log units), log-lik increments of Y and MAP hits, per filter."""
    rng = np.random.default_rng(seed)
    m, mc, one = fit(train, 2), fit(train, 2, AMAX), fit(train, 1)
    runs = {
        "cgomsm1": lambda y, u: cgo_filter(m["cgomsm1"], y, u, "cgomsm1"),
        "cgomsm1_amax": lambda y, u: cgo_filter(mc["cgomsm1"], y, u, "cgomsm1"),
        "cgomsm2": lambda y, u: cgo_filter(m["cgomsm2"], y, u, "cgomsm2"),
        "thgmsm_rbpf": lambda y, u: rbpf(m["thgmsm"], y, u, N_PART, rng),
        "kalman_K1": lambda y, u: rbpf(one["thgmsm"], y, u, 1, rng),
    }
    out = {}
    for name, f in runs.items():
        se, ll, hit = [], [], []
        for s in test:
            E, PI, inc = f(s["y"], s["u"])
            se.append(((E - s["x"]) * sx) ** 2)
            ll.append(inc - np.log(sy))  # log-likelihood of the unscaled log discharge
            hit.append(PI.argmax(axis=1) == s["r"])
        out[name] = dict(se=np.concatenate(se), ll=np.concatenate(ll), hit=np.concatenate(hit))
    Xtr = np.concatenate([s["x"] for s in train])
    Dtr = np.vstack([np.c_[s["y"], s["u"]] for s in train])
    b, _ = _ols(Xtr, Dtr)
    se = np.concatenate([((np.c_[s["y"], s["u"]] @ b - s["x"]) * sx) ** 2 for s in test])
    out["static"] = dict(se=se, ll=None, hit=None)
    params = {
        fam: [
            dict(
                A=float(R["a"]),
                B=float(R["b"]),
                C=float(R["c"]),
                D=float(R["d"]),
                M=float(R["rho"] / R["sv"]),
                Lbar=(np.asarray(R["lx"]) - R["rho"] / R["sv"] * np.asarray(R["ly"])).tolist(),
                Ly=np.asarray(R["ly"]).tolist(),
            )
            for R in mm["reg"]
        ]
        for fam, mm in (
            ("cgomsm1", m["cgomsm1"]),
            ("cgomsm1_amax", mc["cgomsm1"]),
            ("cgomsm2", m["cgomsm2"]),
        )
    }
    params["P"] = m["thgmsm"]["P"].tolist()
    return out, params


def _blocks(n, block, rng):
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(N_BOOT, nb))
    return (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(N_BOOT, -1)[:, :n]


def _ci(v):
    return np.percentile(v, [2.5, 97.5]).tolist()


def summarize(res, rng, block):
    """Scores and paired block-bootstrap intervals (CGOMSM2 minus each other filter, input effect)."""
    n = len(res["input"]["cgomsm2"]["se"])
    idx = _blocks(n, block, rng)
    rmse = lambda se: np.sqrt(se[idx].mean(axis=1))  # noqa: E731
    out = {}
    for key, R in res.items():
        S = {}
        for f, v in R.items():
            S[f] = dict(rmse=float(np.sqrt(v["se"].mean())))
            if v["ll"] is not None:
                S[f].update(loglik=float(v["ll"].mean()), map_acc=float(v["hit"].mean()))
        r2 = rmse(R["cgomsm2"]["se"])
        S["cgomsm2_minus"] = {
            f: dict(d_rmse=S["cgomsm2"]["rmse"] - S[f]["rmse"], ci95=_ci(r2 - rmse(R[f]["se"])))
            for f in FILTERS
            if f != "cgomsm2"
        }
        out[key] = S
    out["input_effect"] = {}
    for f in FILTERS:
        a, b = res["no_input"][f], res["input"][f]
        row = dict(
            d_rmse=float(np.sqrt(b["se"].mean()) - np.sqrt(a["se"].mean())),
            ci95=_ci(rmse(b["se"]) - rmse(a["se"])),
        )
        if a["ll"] is not None:
            row.update(
                d_loglik=float(b["ll"].mean() - a["ll"].mean()),
                ci95_loglik=_ci(b["ll"][idx].mean(axis=1) - a["ll"][idx].mean(axis=1)),
            )
        out["input_effect"][f] = row
    return out


def exp_d(out: Path, cache: Path = CACHE):
    out.mkdir(parents=True, exist_ok=True)
    res = {
        "config": dict(
            years=YEARS,
            step=STEP,
            rise_h=RISE_H,
            n_part=N_PART,
            block_days=BLOCK_DAYS,
            n_boot=N_BOOT,
            amax=AMAX,
            splits=SPLITS,
            reaches=REACHES,
        ),
        "runs": [],
    }
    steps_per_day = 24 // int(STEP.rstrip("h"))
    trace = None
    for reach in REACHES:
        for split in SPLITS:
            d, trm, info = build_frame(reach, split, cache)
            cols = ["x", "y", "trib"]
            mu, sd = d.loc[trm, cols].mean(), d.loc[trm, cols].std()
            raw, par = {}, {}
            for key, use in (("no_input", False), ("input", True)):
                train = sequences(d, trm, mu, sd, use)
                test = sequences(d, ~trm, mu, sd, use)
                raw[key], par[key] = run_filters(train, test, sd["x"], sd["y"])
            hits = np.concatenate([s["r"] for s in test])
            row = dict(
                reach=reach,
                split=split,
                **info,
                n_train=int(sum(len(s["x"]) for s in train)),
                n_test=int(len(hits)),
                majority=float(max(hits.mean(), 1 - hits.mean())),
                scales=dict(x=float(sd["x"]), y=float(sd["y"]), trib=float(sd["trib"])),
                corr_train=d.loc[trm, cols].corr().round(4).to_dict(),
                params=par,  # standardized variables (see scales)
                **summarize(raw, np.random.default_rng(1), BLOCK_DAYS * steps_per_day),
            )
            res["runs"].append(row)
            if reach == "Susquehanna" and split == "A":  # figure: largest test flood
                trace = _trace(d, trm, mu, sd)
            print(_line(row), flush=True)
    res["data_sha256"] = data_fingerprint(cache)
    res["trace"] = trace
    (out / "exp_d.json").write_text(json.dumps(res, indent=1, default=float))
    _table_d(res, out / "table_d.tex")
    _plot_d(res, out / "fig_d_rivers.pdf")
    return res


def _trace(d, trm, mu, sd):
    """Exact CGOMSM1 / CGOMSM2 estimates (with input) around the largest flood of the test years."""
    train = sequences(d, trm, mu, sd, True)
    test = sequences(d, ~trm, mu, sd, True)
    m = fit(train, 2)
    s = max(test, key=lambda s: s["x"].max())
    i = int(np.argmax(s["x"]))
    a, b = max(0, i - 80), min(len(s["x"]), i + 160)
    tr = {"t_h": (np.arange(b - a) * 3).tolist(), "x": (s["x"][a:b] * sd["x"] + mu["x"]).tolist()}
    for fam in ("cgomsm1", "cgomsm2"):
        E, PI, _ = cgo_filter(m[fam], s["y"], s["u"], fam)
        tr[fam] = (E[a:b] * sd["x"] + mu["x"]).tolist()
        tr[fam + "_p_rising"] = PI[a:b, 1].tolist()
    tr["r"] = s["r"][a:b].tolist()
    return tr


def _line(row):
    S, T = row["no_input"], row["input"]
    return (
        f"{row['reach']:11s} {row['split']} tau={row['tau_h']}h lag={row['input_lag_h']}h"
        f" | RMSE no u / u: "
        + "  ".join(f"{f} {S[f]['rmse']:.4f}/{T[f]['rmse']:.4f}" for f in FILTERS)
        + f" | MAP C2 {T['cgomsm2']['map_acc']:.3f} (maj {row['majority']:.3f})"
    )


def _num(v, fmt=".3f"):
    s = format(v, fmt)
    return f"${s.replace('-', '{-}')}$" if s.startswith("-") else f"${s}$"


def _table_d(res, path):
    """Table: RMSE of the log discharge (test years, split A) with and without the input, per reach."""
    names = {
        "cgomsm1": "CGOMSM1",
        "cgomsm1_amax": r"CGOMSM1, $|A_{k}| \leq 0.99$",
        "cgomsm2": "CGOMSM2",
        "thgmsm_rbpf": "THGMSM (RBPF)",
        "kalman_K1": "Kalman ($K = 1$)",
        "static": "static regression",
    }
    runs = [r for r in res["runs"] if r["split"] == "A"]
    lines = [
        r"\begin{tabular}{@{}l" + "cc" * len(runs) + r"@{}}",
        r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{r['reach']}}}" for r in runs) + r" \\",
        " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(runs))),
        " & " + " & ".join(r"no $u$ & $u$" for _ in runs) + r" \\",
        r"\midrule",
    ]
    for f in FILTERS:
        vals = []
        for r in runs:
            vals += [_num(r["no_input"][f]["rmse"]), _num(r["input"][f]["rmse"])]
        lines.append(f"{names[f]} & " + " & ".join(vals) + r" \\")
    lines.append(r"\midrule")
    for lab, key in (
        ("log-lik.\\ of $Y$, CGOMSM1", "cgomsm1"),
        ("log-lik.\\ of $Y$, CGOMSM2", "cgomsm2"),
    ):
        vals = []
        for r in runs:
            vals += [
                _num(r["no_input"][key]["loglik"], ".2f"),
                _num(r["input"][key]["loglik"], ".2f"),
            ]
        lines.append(f"{lab} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def _plot_d(res, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
        }
    )
    tr = res["trace"]
    t = np.asarray(tr["t_h"]) / 24.0
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    ax.plot(t, tr["x"], "k-", lw=1.2, label="measured")
    ax.plot(t, tr["cgomsm1"], "--", color="C0", lw=1.0, label="CGOMSM1 filter")
    ax.plot(t, tr["cgomsm2"], "-.", color="C3", lw=1.0, label="CGOMSM2 filter")
    ax.set_xlabel("time (days)")
    ax.set_ylabel("log discharge at Marietta")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)


if __name__ == "__main__":
    exp_d(Path("results/tac_simulations"))
