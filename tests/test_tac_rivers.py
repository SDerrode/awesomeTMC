#!/usr/bin/env python3
"""
tests/test_tac_rivers.py
========================
Checks of ``prg/experiments/tac_rivers.py`` (Section V-C of the IEEE TAC paper, river discharge):

* the exact CGOMSM1 / CGOMSM2 filters with input against brute-force enumeration of the regime
  paths (means, log-likelihood increments) and against ``tac_simulations.cgo_filter_c``;
* the RBPF with one regime against the Kalman filter;
* the fits (family constraints, |A_k| <= amax for the constrained CGOMSM1);
* the parsing of the USGS RDB format (time zones) and the frame built from the cached data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prg.experiments import tac_rivers as R
from prg.experiments import tac_simulations as T
from tests.test_tac_simulations import _brute_force, _random_c_model, _simulate_c


@pytest.mark.parametrize("family", ["cgomsm1", "cgomsm2"])
@pytest.mark.parametrize("K", [2, 3])
def test_cgo_filter_matches_enumeration_and_case_c(family, K):
    rng = np.random.default_rng(20 * K + (family == "cgomsm2"))
    m = _random_c_model(rng, K, family)
    y, u = _simulate_c(m, 6, rng)
    E, PI, inc = R.cgo_filter(m, y, u, family)
    Eb, _, LLb = _brute_force(m["reg"], m["P"], m["p1"], y, u)
    np.testing.assert_allclose(E, Eb, atol=1e-11)
    assert inc.sum() == pytest.approx(LLb, abs=1e-10)
    np.testing.assert_allclose(PI.sum(axis=1), 1.0, atol=1e-12)
    Ec, LLc = T.cgo_filter_c(m, y, u, family)
    np.testing.assert_allclose(E, Ec, atol=1e-13)
    assert inc.sum() == pytest.approx(LLc, abs=1e-11)


def test_rbpf_with_one_regime_is_the_kalman_filter():
    rng = np.random.default_rng(3)
    m = _random_c_model(rng, 1, "thgmsm")
    y, u = _simulate_c(m, 30, rng)
    E, PI, inc = R.rbpf(m, y, u, 1, np.random.default_rng(0))
    Eb, _, LLb = _brute_force(m["reg"], m["P"], m["p1"], y, u)
    np.testing.assert_allclose(E, Eb, atol=1e-11)
    assert inc.sum() == pytest.approx(LLb, abs=1e-9)
    np.testing.assert_allclose(PI, 1.0)


def _labelled_sequences(rng, n_seq=4, N=400):
    """Sequences of a random 2-regime THGMSM with input, in the format of ``tac_rivers.sequences``."""
    m = _random_c_model(rng, 2, "thgmsm")
    out = []
    for _ in range(n_seq):
        u = np.c_[rng.normal(size=N), np.ones(N)]
        r = np.empty(N, int)
        z = np.empty((N, 2))
        r[0] = rng.choice(2, p=m["p1"])
        z[0] = rng.multivariate_normal(np.zeros(2), m["reg"][r[0]]["S1"])
        for n in range(1, N):
            r[n] = rng.choice(2, p=m["P"][r[n - 1]])
            Q = m["reg"][r[n]]
            w = rng.multivariate_normal(np.zeros(2), [[Q["su"], Q["rho"]], [Q["rho"], Q["sv"]]])
            z[n] = [
                Q["a"] * z[n - 1, 0] + Q["b"] * z[n - 1, 1] + Q["lx"] @ u[n] + w[0],
                Q["c"] * z[n - 1, 0] + Q["d"] * z[n - 1, 1] + Q["ly"] @ u[n] + w[1],
            ]
        out.append(dict(x=z[:, 0], y=z[:, 1], u=u, r=r))
    return out


def test_fits_satisfy_the_family_constraints():
    seqs = _labelled_sequences(np.random.default_rng(5))
    m = R.fit(seqs, 2)
    for Q in m["cgomsm1"]["reg"]:
        assert Q["c"] == 0.0
    for Q in m["cgomsm2"]["reg"]:
        M = Q["rho"] / Q["sv"]
        assert Q["a"] == pytest.approx(M * Q["c"]) and Q["b"] == pytest.approx(M * Q["d"])
    for fam in ("cgomsm1", "cgomsm2"):  # same checks as case c
        T._check_family(m[fam], fam)
    mc = R.fit(seqs, 2, amax=0.1)
    assert all(abs(Q["a"]) <= 0.1 + 1e-12 for Q in mc["cgomsm1"]["reg"])
    np.testing.assert_allclose(m["thgmsm"]["P"], mc["thgmsm"]["P"])


def test_parse_rdb_converts_to_utc():
    text = "\n".join(
        [
            "# comment",
            "agency_cd\tsite_no\tdatetime\ttz_cd\t1_00060\t1_00060_cd",
            "5s\t15s\t20d\t6s\t14n\t10s",
            "USGS\t0\t2020-01-01 00:00\tEST\t10\tA",
            "USGS\t0\t2020-07-01 00:00\tEDT\t20\tA",
            "USGS\t0\t2020-07-01 00:15\tEDT\tIce\tA",
        ]
    )
    s = R._parse_rdb(text)
    assert list(s.index.strftime("%Y-%m-%d %H:%M")) == [
        "2020-01-01 05:00",
        "2020-07-01 04:00",
        "2020-07-01 04:15",
    ]
    assert s.iloc[0] == 10 and s.iloc[1] == 20 and np.isnan(s.iloc[2])


@pytest.mark.skipif(not (R.CACHE / "01576000.csv.gz").exists(), reason="no cached USGS data")
def test_frame_of_the_cached_data():
    d, train, info = R.build_frame("Susquehanna", "A", Path(R.CACHE))
    assert info["tau_h"] == 11 and info["input_lag_h"] == 3
    assert train.sum() > 15000 and (~train).sum() > 11000
    assert set(np.unique(d["r"].dropna())) == {0.0, 1.0}
