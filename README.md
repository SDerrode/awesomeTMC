# awesomeTMC — Triplet Markov Chains for Fast Optimal Filtering in Gaussian Markov Switching Systems

Simulation and **fast exact jump-filtering** algorithms for the paper:

> *On a New Family of Models for Exact Fast Filtering in Gaussian Markov Switching Systems*  
> Stéphane Derrode, Clément Fernandes, Ugo Hubert-Almuzara, Frédéric Lehmann & Wojciech Pieczynski (preprint, 2026)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23081190.svg)](https://doi.org/10.5281/zenodo.23081190)

The paper introduces an exact, **linear-time** optimal filter for Gaussian
switching state-space models satisfying a closed-form structural constraint
(the **AB constraint**; these models are called CGOMSM2 in the paper, and the
models with C = 0 are called CGOMSM1). The command that regenerates the tables and
figures of its Section V from this code is in
[Reproducing Section V of the paper](#reproducing-section-v-of-the-paper) below.

This README documents the **jump-filtering** pipeline: define a model →
simulate → filter exactly.

---

## Scientific context

The project targets **Gaussian Switching Systems (GSS)**, a class of hidden Markov models where the state $(X_n, Y_n)$ evolves according to a regime $R_n \in \{0,\ldots,K-1\}$:

$$Z_{n+1} = F(R_{n+1})\,Z_n + W_{n+1}, \qquad Z_n = \begin{bmatrix}X_n \\ Y_n\end{bmatrix}$$

with $F(k) = \begin{bmatrix}A_k & B_k \\ C_k & D_k\end{bmatrix}$ and a regime-dependent Gaussian noise $W_{n+1}\mid R_{n+1}=k \sim \mathcal{N}(0,\Sigma_W(k))$ whose covariance splits into blocks aligned with $(X_n, Y_n)$:

$$\Sigma_W(k) = \begin{bmatrix}\Sigma_U(k) & \Delta(k) \\ \Delta(k)^{\top} & \Sigma_V(k)\end{bmatrix},$$

so that $\Sigma_U(k)$ is the state-noise covariance, $\Sigma_V(k)$ the observation-noise covariance, and $\Delta(k)$ the cross-covariance between the state and observation noises.

The central objective is to compute $\mathbb{E}[X_n \mid Y_{1:n}]$ efficiently (fast optimal filter).

The AB structural assumption translates into an algebraic constraint
between the seven block matrices $(A, B, C, D, \Sigma_U, \Sigma_V, \Delta)$
of each regime. Since **v0.13.0** the constraint is enforced by the
closed-form **AB constraint**:

$$
A(k) = \Delta(k)\,\Sigma_V(k)^{-1}\,C(k),
\qquad
B(k) = \Delta(k)\,\Sigma_V(k)^{-1}\,D(k).
$$

This is the unique parametrisation of $(A, B)$ compatible with AB
*uniformly* in the joint covariance $\Sigma(r_1)$, making the $K^2$
regime-pair equations of AB trivially satisfied. The free blocks per
regime become $(C, D, \Sigma_U, \Sigma_V, \Delta)$.

```python
from prg.utils.ab_constraint import compute_AB, apply_AB_constraint

# Single regime: closed-form helper
A, B = compute_AB(C, D, Delta, Sigma_V)

# All regimes at once: project an existing GSSParams onto the AB manifold
constrained_params = apply_AB_constraint(params)
```

In the GUI, a single "AB constraint on (A(k), B(k))" checkbox per
regime locks both blocks simultaneously. See the
[CHANGELOG](CHANGELOG.md) for details and the v0.11 → v0.13 migration
table.

**Why this matters for filtering.** The AB constraint is exactly the
closed-form condition under which the observation marginal $(R_n, Y_n)$ is
itself a homogeneous Markov chain. This *marginal Markovianity* is what
makes an **exact, linear-time** optimal filter possible: $p(R_n \mid
Y_{1:n})$ is propagated by a forward recursion and $\mathbb{E}[X_n \mid
Y_{1:n}]$ then follows in closed form, with no Riccati covariance
propagation to carry. The condition $C(k) \neq 0$ characterises this new
family, distinguishing it from the classical conditionally-Gaussian
switching models ($C = 0$); it is checked at model-build time.

---

## Installation

Python **3.14** is required (declared in `pyproject.toml`).  
The package is **not on PyPI** — install it from a clone of the repository. A project virtual environment is recommended (the system-wide Homebrew NumPy does not load correctly on this platform).

```bash
# Clone the repository
git clone https://github.com/SDerrode/awesomeTMC.git
cd awesomeTMC

# Create and activate the venv (must point to Python 3.14)
python3.14 -m venv .venv
source .venv/bin/activate

# Install the package (editable) + dev dependencies
pip install -e ".[dev]"

# Optional: GUI dependencies (PyQt6 + matplotlib)
pip install -e ".[gui]"
```

> The `.venv/` directory is excluded from version control (see `.gitignore`).

### Recommended: use the `Makefile`

A `Makefile` wraps the canonical workflow and **always invokes the venv
interpreter explicitly** (`.venv/bin/python`), which avoids the silent
"wrong interpreter" failures described below:

```bash
make venv      # create .venv with python3.14
make install   # editable install + dev deps
make test      # run the test suite
make check     # lint + typecheck + tests (CI parity)
make help      # list all targets
```

Override the bootstrap interpreter if needed: `make venv PYTHON_SYS=/path/to/python3.14`.

### Troubleshooting: `Package 'awesometmc' requires a different Python`

If `pip install` fails with

```
ERROR: Package 'awesometmc' requires a different Python: 3.11.15 not in '>=3.14'
```

it means `pip` is bound to a Python interpreter older than 3.14 (typical
on macOS where Homebrew installs `pip` for `python@3.11`). Check with:

```bash
pip --version
# pip 26.x from /opt/homebrew/lib/python3.11/site-packages/pip (python 3.11)  ← wrong
```

Fix it with **one** of the following:

```bash
# (a) Activate the project venv first — cleanest for development
source .venv/bin/activate && pip install -e ".[dev]"

# (b) Use the venv's pip explicitly, no activation needed
./.venv/bin/pip install -e ".[dev]"

# (c) Bypass the global pip shim and use python 3.14 directly
python3.14 -m pip install -e ".[dev]"

# (d) Use the version-suffixed pip
pip3.14 install -e ".[dev]"
```

After running any of these, `pip --version` from within the venv should report `(python 3.14)`.

---

## Running the simulator

```bash
# Basic run: 1 000 steps, reproducible seed
python -m prg.simulate --model model_gss_K2_q1_s1 -N 1000 --seed 42

# Dry run (no CSV written), full debug output
python -m prg.simulate --model model_gss_K2_q1_s1 -N 500 --no-save --log-level DEBUG -v 2

# Enforce AB constraint: A, B from (C, D, Δ, Σ_V) before simulating
python -m prg.simulate --model model_gss_K2_q1_s1 -N 1000 --seed 42 --constraint

# Custom output directory
python -m prg.simulate --model model_gss_K2_q1_s1 -N 2000 --seed 0 --output my_run.csv
```

Output CSV columns: `n, r, x_0, …, x_{q-1}, y_0, …, y_{s-1}`  
Log files are written automatically to `logs/`.

### CLI options

| Option | Default | Description |
|---|---|---|
| `--model` | — | Model name in `prg/models/` (required) |
| `-N` | — | Number of time steps (required) |
| `--seed` | `None` | Random seed (omit for non-deterministic) |
| `--output` | auto | Output CSV filename |
| `--log-level` | from `config.toml` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |
| `--no-save` | `False` | Skip CSV writing (dry run) |
| `--constraint` | `False` | Apply AB constraint: A=Δ Σ_V⁻¹ C, B=Δ Σ_V⁻¹ D before run |
| `-v` / `--verbose` | `1` | Console verbosity: 0=silent, 1=normal, 2=debug |

---

## Running the filter

The filter computes $\mathbb{E}[X_n \mid Y_{1:n}]$, $\mathbb{E}[X_nX_n^T \mid Y_{1:n}]$, and $p(R_n \mid Y_{1:n})$ recursively, in time linear in $n$.

```bash
# Simulate 1 000 steps and filter in one command
python -m prg.filter.main --model model_gss_K2_q1_s1 -N 1000 --seed 42

# Filter from an existing simulation CSV
python -m prg.filter.main --model model_gss_K2_q1_s1 \
    --csv data/simulated/simulated_model_gss_K2_q1_s1_N1000_seed42.csv

# Enforce AB constraint before filtering
python -m prg.filter.main --model model_gss_K2_q1_s1 -N 1000 --seed 42 --constraint

# Dry run (no CSV written)
python -m prg.filter.main --model model_gss_K2_q1_s1 -N 500 --no-save -v 2
```

Output CSV columns: `n, E_x_0, …, E_x_{q-1}, V_x_0, …, V_x_{q-1}, p_r_0, …, p_r_{K-1}, sq_err`

| Column | Description |
|---|---|
| `E_x_i` | $\mathbb{E}[X_{n,i} \mid Y_{1:n}]$ — filtered mean |
| `V_x_i` | $\mathrm{Var}(X_{n,i} \mid Y_{1:n})$ — posterior variance (diagonal) |
| `p_r_k` | $p(R_n = k \mid Y_{1:n})$ — regime probability |
| `sq_err` | $\|X_n - \mathbb{E}[X_n \mid Y_{1:n}]\|^2$ — squared error (requires true X) |

### CLI options

| Option | Default | Description |
|---|---|---|
| `--model` | — | Model name in `prg/models/` (required) |
| `--csv` | `None` | Path to an existing simulation CSV |
| `-N` | — | Steps to simulate (required without `--csv`) |
| `--seed` | `None` | Random seed for simulation |
| `--output` | auto | Output CSV filename for filter results |
| `--log-level` | from `config.toml` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |
| `--no-save` | `False` | Skip all CSV writing (dry run) |
| `--constraint` | `False` | Apply AB constraint: A=Δ Σ_V⁻¹ C, B=Δ Σ_V⁻¹ D before run |
| `-v` / `--verbose` | `1` | Console verbosity: 0=silent, 1=normal, 2=debug |

### Python API

```python
from prg.classes.GSSParams import GSSParams
from prg.models.model_gss_K2_q1_s1 import ModelGssK2Q1S1
from prg.filter import GSSFilter

params = GSSParams.from_model(ModelGssK2Q1S1())
filt   = GSSFilter(params)

# Step by step
for y in observations:          # y shape (s,) or (s, 1)
    res = filt.step(y)
    print(res.E_x)              # E[X_n | y_{1:n}]  shape (q, 1)
    print(res.Var_x)            # Var[X_n | y_{1:n}] shape (q, q)
    print(res.pi)               # p(r_n | y_{1:n})   shape (K,)

# Or in one call
sim_path, df = filt.run(N=1000, seed=42, output_dir="data/simulated")
```

> **Worked examples:** runnable filtering notebooks are in
> [`notebooks/`](notebooks/) — a quickstart (`01_filtering_quickstart.ipynb`) and
> a filter comparison (`02_filters_comparison.ipynb`).

### AB constraint API

```python
from prg.utils.ab_constraint import (
    apply_AB_constraint, compute_AB, compute_ab_residual,
)

# Apply A = Δ Σ_V⁻¹ C, B = Δ Σ_V⁻¹ D to all regimes at once
constrained_params = apply_AB_constraint(params)

# Or compute (A, B) for a single regime from (C, D, Δ, Σ_V)
A, B = compute_AB(C, D, Delta, SV)

# Verify any (A, B, C, D, Σ_U, Δ, Σ_V) satisfies AB:
F = compute_ab_residual(A, B, C, D, SU, Delta, SV)  # ‖F‖_F = 0 ⇔ AB holds
```

---

## Graphical interface

The GUI requires `pip install -e ".[gui]"` (PyQt6 + matplotlib).

```bash
# Standalone launch: choose K, q, s interactively
python -m prg.gui.main -K 2 -q 1 -s 1

# Pre-fill from an existing model file
python -m prg.gui.main --model model_gss_K2_q1_s1
```

**Left panel — Parameters**

| Widget | Description |
|---|---|
| Preset selector | Load any model from `prg/models/` in one click; triggers a full window restart when K, q, or s differ |
| Parameter tabs | One tab per Markov state; editable F(k), Σ_W(k), μ_{z0}(k), b(k) with block colour coding (A=blue, B=green, C=yellow, D=pink) |
| AB constraint checkboxes | Four per tab (A / B / Σ_U / Δ=0); auto-compute B(k) or enforce Δ=0 in real-time; affected cells become read-only |
| Stability badges | Spectral radii ρ(F), ρ(A), ρ(D) shown live below the F(k) table |
| Randomize button 🎲 | Fills F(k) and Σ_W(k) with random stable parameters |
| Transition matrix P | Editable (K×K) row-stochastic matrix; stationary distribution π* shown live |
| N / Seed | Number of steps (default 1 000); optional integer seed |
| Monte Carlo checkbox | Enables M-trajectory mode; M spinbox (default 50) |
| Auto-filter checkbox | Runs Filter automatically after each single Simulate |

**Right panel — Plots** (`2 + q + 2s` subplots, shared x-axis)

| Subplot | Content |
|---|---|
| R_n | Regime sequence (step plot) |
| π_n(k) | Filtered regime posteriors — one line per state, populated after Filter |
| X^i (×q) | Hidden component(s); filter overlay (dashed mean ± 2σ band) added after Filter |
| Y^i (×s) | Observed component(s) |
| ν^i (×s) | Filter innovations, populated after Filter |

**Simulation modes**

| Mode | How |
|---|---|
| Single trajectory | Simulate → optional Filter (or tick Auto-filter) |
| Monte Carlo | Tick Monte Carlo, set M, Simulate — shows mean ± 2σ + median over M runs |
| Load CSV | File → Load CSV — display external data, then run Filter with current params |

**Filter quality frame** (shown after Filter)

| Metric | Description |
|---|---|
| log L | Total log-likelihood and per-step mean |
| MSE / RMSE | Against ground-truth X (only when X is available) |
| Ljung-Box | Whiteness test per innovation component (green/amber/red badge) |
| Skew · Kurt | Skewness and excess kurtosis badges; note that GSS innovations are theoretically a mixture of Gaussians, so non-zero kurtosis is expected |

**Param drift indicator** — if you edit any parameter after Simulate and before Filter, the Filter button shows `⚠ Filter` with a tooltip; the Filter will use the parameters captured at Simulate, not the current widget values. Re-run Simulate to apply the new parameters.

**Keyboard shortcuts**

| Shortcut | Action |
|---|---|
| Ctrl+R | Simulate |
| Ctrl+F | Filter |
| Ctrl+Shift+R | Reset |
| Ctrl+S | Save CSV |
| Ctrl+O | Load CSV |
| Ctrl+E | Export model code (.py) |
| Ctrl+Shift+E | Export plots (PNG/PDF/SVG) |
| Ctrl+I | Innovation histograms |
| Ctrl+Shift+X | MC X distributions |

Window geometry, splitter position, M, seed, and auto-filter state are restored across sessions.

---

## Adding a new model

Create a file `prg/models/model_<name>.py` that subclasses `BaseGSSModel`:

```python
import numpy as np
from prg.models.base_gss_model import BaseGSSModel

class ModelMyGSS(BaseGSSModel):
    K, q, s = 3, 2, 1

    P = np.array(...)          # (K, K) row-stochastic

    A_list = [...]             # list of K arrays, each (q, q)
    B_list = [...]             # list of K arrays, each (q, s)
    C_list = [...]             # list of K arrays, each (s, q)
    D_list = [...]             # list of K arrays, each (s, s)

    Sigma_U_list = [...]       # list of K arrays, each (q, q) — SPD
    Delta_list   = [...]       # list of K arrays, each (q, s)
    Sigma_V_list = [...]       # list of K arrays, each (s, s) — SPD

    pi0 = None                 # None → stationary distribution of P
    mu_z0_list    = [...]      # list of K arrays, each (q+s, 1)
    Sigma_z0_list = [...]      # list of K arrays, each (q+s, q+s) — SPD

    def get_params(self) -> dict:
        return {k: getattr(self, k) for k in (
            "K", "q", "s", "P",
            "A_list", "B_list", "C_list", "D_list",
            "Sigma_U_list", "Delta_list", "Sigma_V_list",
            "pi0", "mu_z0_list", "Sigma_z0_list",
        )}
```

Then run with `--model model_my_gss`.

---

## Reproducing Section V of the paper

All the tables and figures of Section V (Experiments) of *"On a New Family of Models
for Exact Fast Filtering in Gaussian Markov Switching Systems"* are regenerated by
[`prg/experiments/tac_simulations.py`](prg/experiments/tac_simulations.py), one case
per subsection:

```bash
# from the repo root, inside the project venv (pip install -e ".[paper]")
python -m prg.experiments.tac_simulations a          # Section V-A  (~10 s)
python -m prg.experiments.tac_simulations b          # Section V-B  (~15 min)
python -m prg.experiments.tac_simulations d          # Section V-C  (~2 min, cached USGS data in data/real/rivers)
python -m prg.experiments.tac_simulations c          # vehicle data of an earlier version (downloads the data once)
python -m prg.experiments.tac_simulations all        # a, b and c
python -m prg.experiments.tac_simulations timing     # computing times per time step
python -m prg.experiments.tac_simulations replot     # tables and figures from the stored results
python -m prg.experiments.tac_simulations transient  # only the transient study of case b
```

The outputs are written to `results/tac_simulations/` (`--out` to change it):

| Case | Paper | Outputs | Content |
|---|---|---|---|
| `a` | §V-A, Table I and the λ figure | `table_a.tex`, `fig_a_lambda.pdf`, `exp_a.json` | Error increase at n = 2 when a THGMSM is replaced by a CGOMSM1 or a CGOMSM2, regime known: closed forms, checked by Monte Carlo |
| `b` | §V-B, Table II and the path figure | `table_b.tex`, `fig_b_homotopy.pdf`, `exp_b.json`, `exp_b_seq.npz` | Path of THGMSMs from a CGOMSM1 to a CGOMSM2: exact filters of both families against the exact filter by enumeration (N = 10) and an RBPF (N = 200); Monte Carlo half-widths, bootstrap of the crossing point, RBPF accuracy check, initial transient at the two ends of the path |
| `d` | §V-C, Table III and the flood figure | `table_d.tex`, `fig_d_rivers.pdf`, `exp_d.json` | Real river-discharge data (USGS, three reaches, 2014–2023) with the gauged tributary inflow as a measured input: THGMSM, CGOMSM1 (also with \|A\| ≤ 0.99) and CGOMSM2 fitted per rising/falling regime, filters run on the test years, static regression as a reference, paired block-bootstrap intervals, regime accuracy ([`tac_rivers.py`](prg/experiments/tac_rivers.py)) |
| `c` | earlier version of §V-C | `table_c.tex`, `fig_c_vehicle.pdf`, `exp_c.json` | Real driving data with a known steering input (replaced by case `d` in the current version of the paper) |
| `timing` | §V | `timing.json` | Time per step of the exact CGOMSM1/CGOMSM2 filters, of the RBPF and of the enumeration |

**Notes**

- **Self-contained filters.** The script contains its own numpy implementations of
  the filters used in Section V (scalar case): the exact filters of the CGOMSM1 and of
  the CGOMSM2 for an arbitrary initial law, the exact filter by enumeration of the
  regime paths, and the RBPF, with their versions with a known input.
  `tests/test_tac_simulations.py` checks them against brute-force enumeration (to
  about 1e-12). They do not use `GSSFilter`, whose `ngh_kf` mode starts from the
  Gaussian law with the stationary moments, whatever the initial law of the model.
- **Determinism.** All the random numbers are seeded; the seeds are recorded in
  `exp_b.json` (`cfg`). The PDF figures carry no creation date.
- **River data.** Case `d` uses the instantaneous discharge of 19 USGS gauges (U.S. Geological
  Survey, National Water Information System, doi:10.5066/F7P55KJN, public domain), averaged
  hourly (UTC) and cached in `data/real/rivers/` (6.6 MB, kept in the repository because the
  provisional USGS records may be revised); without the cache, they are downloaded again from
  waterservices.usgs.gov. `exp_d.json` records a SHA-256 fingerprint of the cache.
- **Vehicle data.** Case `c` uses the aggressive-driver recordings of the open dataset
  Figshare `10.6084/m9.figshare.28078274.v1` (CC BY 4.0; 135 recordings at 100 Hz, of
  which the 134 longer than 5 s are used). The ~20 MB archive is downloaded once,
  checked against its published MD5, and cached under `data/real/vehicle/` (`--data`
  to change it); `exp_c.json` records a SHA-256 fingerprint of the files.

## Figures of an earlier draft

An earlier draft of the paper, entitled *"Exact Constant-Gain Filtering in Gaussian
Markov Switching Systems"*, had other experiments. Its figures are regenerated by
[`prg/experiments/make_paper_figures.py`](prg/experiments/make_paper_figures.py):

```bash
# from the repo root, inside the project venv
python -m prg.experiments.make_paper_figures
```

It writes the four figures of that draft into `docs/NGH-MSM_V2/figures/` (the
directory is created if needed):

| Figure file | Draft | Experiment |
|---|---|---|
| `e2_speed.pdf` | Fig. 3 (§VI-A) | **Linear-time cost** — wall time vs. $N$, and the intractable $K^N$ enumeration |
| `e8_c_influence.pdf` | Fig. 4 top (§VI-B) | **Coupling sweep** — regime accuracy and state RMSE as $C$ grows |
| `e9_c_mismatch.pdf` | Fig. 4 bottom (§VI-B) | **Mismatch** — filtering $C \neq 0$ data with the correct vs. the old $C = 0$ filter |
| `vehicle_consigne.pdf` | Fig. 5 (§VI-C) | **Real driving data** — the exogenous-input (consigne) read-out for the yaw rate |

The model parameters of the three synthetic experiments are listed in Table I of
that draft (and in full in its supplementary material).

**Notes**

- **Determinism.** The synthetic experiments are fully seeded, so the figures and
  the reported RMSE / timing numbers are bit-for-bit reproducible.
- **Real data.** The vehicle figure downloads the ~20 MB open dataset once
  (Figshare `10.6084/m9.figshare.28078274.v1`, CC BY 4.0; aggressive-driver
  recordings on a test track, 100 Hz; the Mendeley record `10.17632/x7n6jnjh36.3`
  only holds a placeholder) and caches it under `data/real/vehicle/`. The read-out
  gains $M_r, N_r$ are fitted by ordinary least squares, per speed / steering regime, on
  an 80-file training split; the 30 held-out files give the reported RMSEs
  (details in the supplementary material).
- **Cost.** The whole suite runs in ~10 minutes single-threaded; no GPU needed.

The broader, exploratory harness `prg/experiments/study.py` (run as
`python -m prg.experiments.study <outdir>`) still generates the full candidate set
of development experiments; `make_paper_figures.py` selects and assembles exactly
the four that appear in that draft.

---

## Testing

```bash
source .venv/bin/activate
pytest                        # full suite, ~45 s
mypy                          # strict typing on prg/utils/ab_constraint.py
```

The suite covers matrix diagnostics, parameter validation, the
iterator-based simulator, reproducibility, CSV output, statistical sanity,
exact-filter correctness, and the AB constraint computation.

---

## Configuration

Edit `config.toml` to change default paths and log level:

```toml
[general]
log_level = "INFO"    # DEBUG | INFO | WARNING | ERROR

[paths]
data_simulated  = "data/simulated"
logs            = "logs"
```

---

## Citing

If you use this code or the AB framework in your work, please cite the paper and
the software:

```bibtex
@misc{derrode_newfamily_2026,
  author       = {Derrode, St{\'e}phane and Fernandes, Cl{\'e}ment and Hubert-Almuzara, Ugo and Lehmann, Fr{\'e}d{\'e}ric and Pieczynski, Wojciech},
  title        = {{On a New Family of Models for Exact Fast Filtering in Gaussian Markov Switching Systems}},
  year         = {2026},
  note         = {Preprint},
}

@software{derrode_awesomeTMC_2026,
  author       = {Derrode, St{\'e}phane},
  title        = {{awesomeTMC}: Triplet {Markov} Chains for Fast Optimal Filtering in {Gaussian} {Markov} Switching Systems},
  year         = {2026},
  doi          = {10.5281/zenodo.23081190},
  url          = {https://github.com/SDerrode/awesomeTMC},
}
```

The concept DOI above always resolves to the latest version; each release also has
its own DOI on Zenodo.

A `CITATION.cff` file is also provided at the repo root.

---

## Authors

- **Stéphane Derrode** — [stephane.derrode@ec-lyon.fr](mailto:stephane.derrode@ec-lyon.fr) ([École Centrale de Lyon](https://www.ec-lyon.fr/) — LIRIS, CNRS UMR 5205)
- **Clément Fernandes** — [Télécom SudParis](https://www.telecom-sudparis.eu/) (SAMOVAR, Institut Polytechnique de Paris)
- **Ugo Hubert-Almuzara** — Safran Electronics & Defense
- **Frédéric Lehmann** — [Télécom SudParis](https://www.telecom-sudparis.eu/) (SAMOVAR)
- **Wojciech Pieczynski** — [wojciech.pieczynski@telecom-sudparis.eu](mailto:wojciech.pieczynski@telecom-sudparis.eu) ([Télécom SudParis](https://www.telecom-sudparis.eu/) — SAMOVAR)

## License

GNU Affero General Public License v3.0 — see `LICENSE`.
