#!/usr/bin/env python3
"""
awesometmc/experiments/fuzzy_cgofmsm_paper.py
=============================================
Reproduction of the simulation study (Section VI) of

    Z. Bouyahia, S. Derrode, W. Pieczynski, "Filtering in Gaussian linear systems
    with fuzzy switches", IEEE Trans. Fuzzy Systems 28(8), 2020 -- [BDP20],

with the reference implementation of ``fuzzy_cgofmsm.py``.

Two experiments
---------------
figure5   MSE of the restored jumps and states against the number F of fuzzy
          steps, FMC1 law (alpha, eta, delta) = (0.07, 0.16, 0.05), 50 runs of
          N = 300. Reference values are read off Fig. 5 of the paper.
table1    MSE against the proportion of hard regimes, FMC1 with F = 4 and FMC2
          with F = 10, 20 runs of N = 1000. The paper gives the proportions but
          not the parameters behind them; the families used here are a choice
          (see ``fmc1_family`` and ``fmc2_family``), so only the column of FMC1 at
          67 % -- the parameters of Fig. 5 -- is a like-for-like comparison.

Each experiment runs on two Gaussian models: the matrices as printed in the
paper (``a3 != 0`` at two corners and inside the square), and the same matrices
with the CGOMSM condition ``a3 = 0`` enforced after interpolation.

``--api public`` runs the same study through the library interface
``awesometmc.fuzzy`` (discretisation into a ``PairGSSParams``, ``GSSFilter``).

CLI:  python -m awesometmc.experiments.fuzzy_cgofmsm_paper [--runs-scale 1.0] [--api public] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import warnings
from types import SimpleNamespace

import numpy as np

from awesometmc.experiments.fuzzy_cgofmsm import (
    FMC1,
    FMC2,
    FuzzyGaussianModel,
    FuzzyMarkovLaw,
    discretise,
    harden,
    mpm_jumps,
    paper_gaussian_model,
    simulate,
)
from awesometmc.experiments.fuzzy_cgofmsm import fuzzy_filter as _reference_filter
from awesometmc.experiments.fuzzy_cgofmsm import known_jumps_filter as _reference_known_jumps
from awesometmc.fuzzy import FuzzyCGOMSM, FuzzyDiscretisation

__all__ = ["PAPER_FIG5", "PAPER_TABLE1", "fmc1_family", "fmc2_family", "mse_study", "main"]

# Fig. 5 of [BDP20], read off the published figure (two significant digits at best).
PAPER_FIG5 = {
    "F": (0, 1, 2, 3, 5, 7, 9),
    "mse_r_fuzzy": (0.077, 0.052, 0.043, 0.042, 0.041, 0.040, 0.040),
    "mse_x_fuzzy": (0.485, 0.448, 0.428, 0.427, 0.427, 0.427, 0.427),
    "mse_r_hard": 0.077,
    "mse_x_hard": 0.495,
    "mse_x_kj": 0.35,
}

# Table I of [BDP20]: (percentage of hard samples, jumps and states MSE of the fuzzy
# filter, jumps and states MSE of the hard filter).
PAPER_TABLE1 = {
    "FMC1": [
        (48, 0.051, 0.475, 0.126, 0.599),
        (58, 0.046, 0.446, 0.094, 0.535),
        (67, 0.039, 0.431, 0.074, 0.500),
        (75, 0.034, 0.415, 0.057, 0.468),
        (93, 0.017, 0.368, 0.020, 0.386),
    ],  # fmt: skip
    "FMC2": [
        (48, 0.030, 0.442, 0.070, 0.538),
        (62, 0.026, 0.442, 0.040, 0.479),
        (75, 0.021, 0.420, 0.033, 0.457),
        (87, 0.014, 0.399, 0.021, 0.422),
        (100, 0.012, 0.37, 0.011, 0.374),
    ],  # fmt: skip
}


def fmc1_family(p_hard: float) -> FMC1:
    """FMC1 law with a given proportion of hard regimes.

    The fuzzy part ``(eta, delta)`` of the Fig. 5 law is scaled, and the hard mass
    it frees is shared between ``alpha`` and ``beta`` in the proportion of Fig. 5.
    ``fmc1_family(2 / 3)`` is the Fig. 5 law itself.
    """
    base = FMC1(0.07, 0.16, 0.05)
    t = (1.0 - p_hard) / base.p_fuzzy
    eta, delta = t * base.eta, t * base.delta
    corner = (1.0 - 2.5 * (delta + eta) + (delta - eta) / 6.0) / 2.0  # alpha + beta
    return FMC1(corner * base.alpha / (base.alpha + base.beta), eta, delta)


def fmc2_family(p_hard: float) -> FMC2:
    """FMC2 law with a given proportion of hard regimes: band of half-width 0.2 as in
    Fig. 1(b), ``gamma`` set by the proportion, ``alpha / (alpha + beta)`` as in Fig. 1(b)."""
    base = FMC2(0.15, 0.60, 0.20)
    d = base.delta
    gamma = (1.0 - p_hard) / (d * (4.0 - d))
    corner = (1.0 - gamma * d * (6.0 - d)) / 2.0
    return FMC2(corner * base.alpha / (base.alpha + base.beta), gamma, d)


class _PublicModel:
    """A discretised model of ``awesometmc.fuzzy`` behind the interface ``mse_study`` uses."""

    def __init__(self, disc: FuzzyDiscretisation) -> None:
        self.disc, self.nodes = disc, disc.nodes

    def run(self, y):
        filt = self.disc.filter()
        out = [filt.step(yn) for yn in np.asarray(y, float).ravel()]
        return SimpleNamespace(
            E_x=np.array([o.E_x.item() for o in out]), post=np.array([o.pi for o in out])
        )


def mse_study(
    law: FuzzyMarkovLaw,
    gauss: FuzzyGaussianModel,
    F_list,
    N: int,
    n_runs: int,
    seed: int = 0,
    likelihood: str = "paper",
    api: str = "reference",
) -> list[dict]:
    """Mean squared errors of the fuzzy filter (one row per F), the hard filter and
    the known-jumps filter, averaged over ``n_runs`` samples of length ``N``.

    api
        ``"reference"``: the scalar implementation of ``fuzzy_cgofmsm.py``.
        ``"public"``: the library interface, ``awesometmc.fuzzy.FuzzyCGOMSM`` discretised
        and filtered by ``GSSFilter`` (``likelihood`` is then ignored: the mode is
        ``"cgo"`` when ``gauss.enforce_cgo``, ``"gpb2"`` otherwise). The samples are the
        same in both cases, so the two columns of numbers are directly comparable.
    """
    if api not in ("reference", "public"):
        raise ValueError("api must be 'reference' or 'public'")
    rng = np.random.default_rng(seed)
    if api == "public":
        with warnings.catch_warnings():
            # the matrices printed in the paper violate the CGO identity at two corners;
            # interpolation="cgo" corrects them, which is what enforce_cgo does
            warnings.simplefilter("ignore", RuntimeWarning)
            model = FuzzyCGOMSM(
                law,
                means=(gauss.M0, gauss.M1),
                gammas=gauss.Gamma,
                interpolation="cgo" if gauss.enforce_cgo else "bilinear",
            )
        models = {F: _PublicModel(model.discretise(F)) for F in F_list}
        hard = _PublicModel(model.harden())

        def fuzzy_filter(m, y, likelihood=None):  # noqa: F811 -- same call, other engine
            return m.run(y)

        def known_jumps_filter(_gauss, r, y):  # noqa: F811
            e, v = model.known_jumps_filter(r, y)
            return e[:, 0], v[:, 0, 0]
    else:
        fuzzy_filter, known_jumps_filter = _reference_filter, _reference_known_jumps
        models = {F: discretise(law, gauss, F) for F in F_list}
        hard = harden(law, gauss)
    acc = {F: [] for F in F_list}
    base = []
    for _ in range(n_runs):
        r, x, y = simulate(law, gauss, N, rng)
        res_h = fuzzy_filter(hard, y, likelihood=likelihood)
        e_kj, _ = known_jumps_filter(gauss, r, y)
        base.append(
            (
                np.mean((res_h.E_x - x) ** 2),
                np.mean((mpm_jumps(res_h.post, hard.nodes) - r) ** 2),
                np.mean((e_kj - x) ** 2),
                np.mean((r == 0.0) | (r == 1.0)),
            )
        )
        for F in F_list:
            res = fuzzy_filter(models[F], y, likelihood=likelihood)
            acc[F].append(
                (
                    np.mean((res.E_x - x) ** 2),
                    np.mean((mpm_jumps(res.post, models[F].nodes) - r) ** 2),
                )
            )
    b = np.mean(base, axis=0)
    rows = []
    for F in F_list:
        a = np.array(acc[F])
        rows.append(
            dict(
                F=F,
                mse_x_fuzzy=a[:, 0].mean(),
                se_x_fuzzy=a[:, 0].std(ddof=1) / np.sqrt(n_runs),
                mse_r_fuzzy=a[:, 1].mean(),
                mse_x_hard=b[0],
                mse_r_hard=b[1],
                mse_x_kj=b[2],
                p_hard_theory=law.p_hard,
                p_hard_sample=b[3],
            )
        )
    return rows


def _variants():
    return {"printed": paper_gaussian_model(False), "cgo": paper_gaussian_model(True)}


def run_figure5(n_runs: int = 50, seed: int = 0, api: str = "reference") -> list[dict]:
    law = FMC1(0.07, 0.16, 0.05)
    out = []
    for name, gauss in _variants().items():
        for row in mse_study(law, gauss, PAPER_FIG5["F"], N=300, n_runs=n_runs, seed=seed, api=api):
            out.append(dict(experiment="figure5", gaussian=name, law="FMC1", **row))
    return out


def run_table1(n_runs: int = 20, seed: int = 0, api: str = "reference") -> list[dict]:
    out = []
    for law_name, family, F in (("FMC1", fmc1_family, 4), ("FMC2", fmc2_family, 10)):
        for pct, *_ in PAPER_TABLE1[law_name]:
            law = family(pct / 100.0)
            for name, gauss in _variants().items():
                (row,) = mse_study(law, gauss, [F], N=1000, n_runs=n_runs, seed=seed, api=api)
                out.append(dict(experiment="table1", gaussian=name, law=law_name, pct=pct, **row))
    return out


def _print_figure5(rows) -> None:
    ref = PAPER_FIG5
    print("\nFigure 5 -- FMC1 (0.07, 0.16, 0.05), N = 300")
    print("  F | paper: jumps states | printed: jumps states | a3 = 0: jumps states")
    for i, F in enumerate(ref["F"]):
        cells = []
        for name in ("printed", "cgo"):
            (r,) = [x for x in rows if x["gaussian"] == name and x["F"] == F]
            cells.append(f"{r['mse_r_fuzzy']:.3f}  {r['mse_x_fuzzy']:.3f}")
        print(f"  {F} | {ref['mse_r_fuzzy'][i]:.3f}  {ref['mse_x_fuzzy'][i]:.3f}        | {cells[0]}          | {cells[1]}")  # fmt: skip
    for name in ("printed", "cgo"):
        r = next(x for x in rows if x["gaussian"] == name)
        print(
            f"  {name:8s}: hard filter jumps {r['mse_r_hard']:.3f} (paper {ref['mse_r_hard']}), "
            f"states {r['mse_x_hard']:.3f} (paper {ref['mse_x_hard']}); "
            f"known jumps {r['mse_x_kj']:.3f} (paper {ref['mse_x_kj']}); "
            f"hard share {r['p_hard_sample']:.3f}"
        )


def _print_table1(rows) -> None:
    for law_name, ref in PAPER_TABLE1.items():
        print(f"\nTable I -- {law_name}: jumps / states MSE, fuzzy filter then hard filter")
        print("  % hard | paper                      | printed                    | a3 = 0")
        for pct, rf, xf, rh, xh in ref:
            cells = []
            for name in ("printed", "cgo"):
                (r,) = [
                    x
                    for x in rows
                    if x["law"] == law_name and x["pct"] == pct and x["gaussian"] == name
                ]
                cells.append(
                    f"{r['mse_r_fuzzy']:.3f} {r['mse_x_fuzzy']:.3f} / {r['mse_r_hard']:.3f} {r['mse_x_hard']:.3f}"
                )
            print(
                f"  {pct:5d}  | {rf:.3f} {xf:.3f} / {rh:.3f} {xh:.3f}  | {cells[0]}  | {cells[1]}"
            )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Reproduce Section VI of Bouyahia et al. (2020).")
    ap.add_argument("--runs-scale", type=float, default=1.0, help="multiply the numbers of runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--api",
        choices=("reference", "public"),
        default="reference",
        help="scalar reference implementation, or the library interface awesometmc.fuzzy",
    )
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/fuzzy_cgofmsm"))
    args = ap.parse_args(argv)
    fig5 = run_figure5(max(2, round(50 * args.runs_scale)), args.seed, args.api)
    _print_figure5(fig5)
    tab1 = run_table1(max(2, round(20 * args.runs_scale)), args.seed, args.api)
    _print_table1(tab1)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("figure5", fig5), ("table1", tab1)):
        with open(args.out / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"\nCSV written to {args.out}/")


if __name__ == "__main__":
    main()
