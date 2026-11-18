# API Overview

Module map and the role of each component.

## Top-level layout

```
awesometmc/
├── classes/        # Data containers (parameters, simulator, F matrix)
├── filter/         # Optimal filter (ngh_kf, gpb2 and cgo modes)
├── learning/       # Supervised + semi-supervised estimators
├── models/         # Built-in BaseGSSModel subclasses
├── experiments/    # Paper-reproduction pipelines (§6, §7)
├── gui/            # PyQt6 interface (optional)
├── utils/          # Errors, matrix checks, AB constraint closed form
├── simulate.py     # CLI: simulator
└── filter/main.py  # CLI: filter
```

## Core classes

### `GSSParams` (`awesometmc/classes/GSSParams.py`)

Aggregates all GSS model parameters and validates them. Construct from
a `BaseGSSModel`:

```python
params = GSSParams.from_model(MyModel())
print(params.K, params.q, params.s)
print(params.stationary_distribution())   # π_∞ from P
```

Mutable in tests, immutable in normal use. Build instances with
`from_model`; it exposes `K`, `q`, `s`, the `f_matrix`/`noise_cov`
blocks and `stationary_distribution()`.

### `PairGSSParams` (`awesometmc/classes/PairGSSParams.py`)

Parameters indexed by the regime **pair** `(r_n, r_{n+1})` instead of the arrival
regime: `F(j, k)`, `Sigma_W(j, k)`, `b(j, k)` and `G(j, k)` describe the transition
from regime `j` to regime `k`. This is the setting of the pair-dependent CGOMSM of
Abbassi et al. (2015), and of a fuzzy switching model discretised on a grid.

```python
from awesometmc.classes import PairGSSParams

pair = params.as_pair()                    # broadcast an arrival-indexed GSSParams
pair = PairGSSParams(K=2, q=1, s=1, P=P, F=F, Sigma_W=SW, pi0=None,
                     mu_z0_list=mu0, Sigma_z0_list=S0)   # F, SW of shape (K, K, 2, 2)

sim  = GSSSimulator(pair, N=1000, seed=42)
filt = GSSFilter(pair)                     # mode "cgo" when pair.is_cgo(), else "gpb2"
```

Accepted by `GSSSimulator` and by `GSSFilter` in modes `"gpb2"` and `"cgo"`. Parameters
broadcast with `as_pair()` reproduce the arrival-indexed trajectory and filter
output bit for bit. `to_arrival()` converts back when nothing depends on the
departure regime; `augmented()` returns the same model as a `GSSParams` on the
`K²` regimes `(r_{n-1}, r_n)`, for the tools that take arrival-indexed parameters.

### `GSSSimulator` (`awesometmc/classes/GSSSimulator.py`)

Iterator: yields `(n, r, x, y)` tuples one step at a time.

```python
sim = GSSSimulator(params, seed=42)
for n, r, x, y in sim:
    ...                                    # n is 1-based, length unbounded
sim_df = sim.to_dataframe(N=1000)         # convenience: collect as DataFrame
```

### `FMatrix`, `NoiseCovariance` (`awesometmc/classes/`)

Block representations of \(F(k)\) and \(\Sigma_W(k)\) with cached
spectral radius, conditioning, etc.

## Filtering

### `GSSFilter` (`awesometmc/filter/gss_filter.py`)

```python
from awesometmc.filter import GSSFilter

filt = GSSFilter(params, mode="ngh_kf")  # or "gpb2", "cgo"
res  = filt.step(y)                         # one observation
print(res.E_x, res.Var_x, res.pi, res.log_lik, res.innovation)
```

| `mode` | Cost per step | Assumption |
|---|---|---|
| `ngh_kf` | constant Kalman gain (precomputed) | requires AB |
| `gpb2` | full IMM mixing | works for any GSS |
| `cgo` | vectorised over the K² pairs, no Kalman update | requires C = 0 |

`ngh_kf` and `gpb2` use the exact pair-conditional predictive covariance, so on
AB-constrained params they produce numerically identical output (checked
in `tests/test_filter_modes.py`); `gpb2` is the fallback for
unconstrained models.

`cgo` is the exact filter of the models whose observation does not read the
state (C = 0, for arrival-indexed or pair-indexed parameters). On these models
it returns the same estimates as `gpb2` — which is exact there too — 4 to 100
times faster for K = 2 to 12, because every gain is computed once at
construction. It is the default mode of a `PairGSSParams` with `is_cgo()`, and
warns when C is not zero.

Note that `ngh_kf` issues a `RuntimeWarning` if the params do not
satisfy AB — you can suppress this with `warnings.catch_warnings()`
when you knowingly want to feed unconstrained params (as in the §7
fairness comparison).

## Learning

### `fit_supervised(rs, xs, ys, K, q, s, constraint=None, …)`

Per-regime weighted OLS in closed form. `constraint='ab'` applies a
post-hoc AB projection. Returns a parameter dict
ready for `GSSParams`.

### `fit_semi_supervised(xs, ys, K, n_inits=10, max_iter=100, constraint=None, constraint_each_iter=False, …)`

Baum-Welch EM with k-means initialisation and multi-start. Returns
`(params_dict, info_dict)` where `info_dict` contains
`best_log_lik`, `all_log_liks`, `log_lik_history`, `best_n_iter`,
`best_converged`.

Set `constraint_each_iter=True` to obtain the **GEM** variant of
the paper (projection at every M-step, log-lik no longer monotone but
constraint holds throughout).

## Utilities

### `apply_AB_constraint(params)`

Recompute \(A(k)\) and \(B(k)\) for every regime from
\((C, D, \Delta, \Sigma_V)\) via the closed-form AB constraint
\(A = \Delta\,\Sigma_V^{-1}\,C\), \(B = \Delta\,\Sigma_V^{-1}\,D\).

### `compute_AB(C, D, Delta, Sv)`

Same, for a single regime. Returns the tuple `(A, B)`.

### `compute_ab_block_residuals(A, B, C, D, Delta, Sv, *, relative=False)`

Block residuals of the AB constraint for one regime,
\(R_A = A - M C\) and \(R_B = B - M D\) with \(M = \Delta \Sigma_V^{-1}\)
(shapes `(q, q)` and `(q, s)`). Both vanish \(\iff\) the regime satisfies
AB. With `relative=True` each is divided by the scale of the two terms of
its difference, \(\max(\|A\|_F, \|M C\|_F)\), resp.
\(\max(\|B\|_F, \|M D\|_F)\).

### `ab_residual_max(params, *, relative=True)`

The AB test. Returns `(max_resid, (k, block))`: the largest (relative) block
residual over all regimes \(k\) and both blocks `"A"`, `"B"`.
`max_resid == 0` \(\iff\) the model is AB-constrained. This is the test
used by `validate_ngh_msm`, `exactness_domains` and `GSSFilter` (the
`ngh_kf` warning).

### `compute_pu_residual(A, B, C, D, Su, Delta, Sv)`

Same-regime \((k, k)\) regression residual of property (PU) — the
conditional independence of \(X_{n+1}\) and \(Y_n\) given
\((R_n, R_{n+1}, Y_{n+1})\) — for the joint law of \((X_n, Y_n)\) whose
covariance given \(r_n = k\) is \(\Sigma_W(k)\) (the stationary law of an AB
model): \(\Delta^T A^T + \Sigma_V B^T - P M^{-1}(Q A^T + R B^T + \Delta^T)\),
a `(s, q)` array. **Necessary but not sufficient** for AB: it ignores the
cross pairs, and it is a condition on one particular law.

### `compute_pu_pair_residual(A_k, B_k, C_k, D_k, Delta_k, Sv_k, Su_j, Delta_j, Sv_j)`

Pairwise (PU) residual \(\beta_1(j, k)\) (shape `(q, s)`) — the loading on
\(Y_n\) of the regression of \(X_{n+1}\) on \((Y_n, Y_{n+1})\) given
\(r_n = j,\ r_{n+1} = k\), for the same particular law.

### `pu_residual_max(params, *, relative=True)`

Returns `(max_resid, (j, k))`: the largest pairwise (PU) residual over all
\(K^2\) ordered pairs (normalised by the regression scale when
`relative=True`). It is 0 on every AB model, but also on models that are not
AB (e.g. the JMLS `awesometmc.experiments.cns_exactness.jmls_cross_annihilation_model`,
on which the constant-gain filter is biased): a diagnostic, not the AB test.

### Error hierarchy

```
GSSError
├── GSSConfigError        # bad parameters at construction time
├── GSSStateError         # invalid runtime state (e.g. NaN posterior)
└── GSSConstraintError    # AB not satisfied / cannot be enforced
```

## Reproducibility helpers

The `awesometmc/experiments/` package is structured so each script can be
invoked as a module *or* called from Python:

```python
from awesometmc.experiments.run_real_data import (
    load_enso, run_e1, run_e2, run_e3,
)

data = load_enso(Path("data/real/enso_sst.csv"))
e2   = run_e2(data, K=3)
print([s for s in e2["scores"]])
```

This makes it trivial to re-run an experiment with different seeds or
hyper-parameters from a notebook.

## See also

- [Tutorial](Tutorial) — end-to-end example
- [Paper-Reproduce](Paper-Reproduce) — paper experiments
