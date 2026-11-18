"""
awesometmc/fuzzy
================
Gaussian linear systems with **fuzzy** switches: the regime takes its values in
``[0, 1]``, with positive probability on the two "hard" regimes 0 and 1 and a
density in between (Bouyahia, Derrode, Pieczynski, IEEE TFS 2020).

laws    ``FuzzyMarkovLaw``, ``FMC1``, ``FMC2``, ``FuzzyGrid``, ``mpm_jumps``
model   ``FuzzyCGOMSM`` (specification, simulation, discretisation) and
        ``FuzzyDiscretisation`` (the model on a grid, as a ``PairGSSParams``)
estimation  ``fit_supervised``, ``fit_law``, ``fit_gaussian``: a model from a sample
        whose regimes are known ([BDP20], Section VII)

A fuzzy model is filtered through its discretisation, by ``GSSFilter``:

    from awesometmc.fuzzy import FMC1, FuzzyCGOMSM

    model = FuzzyCGOMSM(FMC1(0.07, 0.16, 0.05), means=(M0, M1), gammas=Gamma)
    r, x, y = model.simulate(N=300, seed=0)
    disc = model.discretise(F=5)
    filt = disc.filter()
    estimates = [filt.step(yn).E_x for yn in y]

and smoothed the same way, all the observations at once:

    E_x, Var_x, gamma, log_lik = disc.smoother(y)
    r_hat = disc.mpm(gamma)

A model is estimated from a sample whose regimes are known by

    from awesometmc.fuzzy import fit_supervised

    fit = fit_supervised(r, x, y, law="fmc1")
    disc = fit.model.discretise(F=5)
"""

from awesometmc.fuzzy.estimation import (
    FuzzyFit,
    GaussianFit,
    fit_gaussian,
    fit_law,
    fit_supervised,
)
from awesometmc.fuzzy.laws import FMC1, FMC2, FuzzyGrid, FuzzyMarkovLaw, mpm_jumps
from awesometmc.fuzzy.model import INTERPOLATIONS, FuzzyCGOMSM, FuzzyDiscretisation

__all__ = [
    "FMC1",
    "FMC2",
    "INTERPOLATIONS",
    "FuzzyCGOMSM",
    "FuzzyDiscretisation",
    "FuzzyFit",
    "FuzzyGrid",
    "FuzzyMarkovLaw",
    "GaussianFit",
    "fit_gaussian",
    "fit_law",
    "fit_supervised",
    "mpm_jumps",
]
