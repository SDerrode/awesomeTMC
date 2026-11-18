# Usage notebooks

Worked examples for the package's fast exact jump-filter (CGOMSM2-KF), and for the
fuzzy switching models of `awesometmc.fuzzy`. Parameter learning by EM is not
covered here.

| Notebook | Contents |
|---|---|
| [`01_filtering_quickstart.ipynb`](01_filtering_quickstart.ipynb) | Build a model, simulate, run the filter (`GSSFilter.step`), read off `E[X|y]`, `Var[X|y]`, `p(r|y)`; verify exactness against the brute-force filter. |
| [`02_filters_comparison.ipynb`](02_filters_comparison.ipynb) | Exactness across M1/M2/M3, the value of modelling the switching (vs a pairwise Kalman and an oracle), multivariate tracking, and CGOMSM2-KF vs the general IMM. |
| [`03_fuzzy_switches.ipynb`](03_fuzzy_switches.ipynb) | Regimes in `[0, 1]` (`awesometmc.fuzzy`): specify and simulate a model, filter and smooth it through its discretisation, compare with the hardened two-regime model, estimate a model from labelled regimes (`fit_supervised`). |

## Running

From the repo root, with the project installed (`pip install -e .`):

```bash
jupyter lab notebooks/        # or: jupyter notebook
```

Use the kernel of the project's environment (the one where `import awesometmc` works).
The notebooks import the public filtering API (`GSSFilter`, `GSSSimulator`) plus
the paper's example models and reference filters in `awesometmc.experiments`.
