# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`rabbit` is a Python package for performing complex profile binned maximum likelihood fits using differential programming (TensorFlow 2 + TensorFlow Probability), interfaced with SciPy minimizers.

## Environment Setup

```bash
source setup.sh   # sets RABBIT_BASE, PYTHONPATH, PATH — required before running any code
```

The virtual environment lives in `env/`. Activate with `source env/bin/activate`.

## Common Commands

**Create test input tensor:**
```bash
python tests/make_tensor.py -o /tmp/
```

**Run a fit:**
```bash
rabbit_fit.py test_tensor.hdf5 -o results/ -t 0 --doImpacts --globalImpacts --saveHists --computeHistErrors
```

**Diagnostics:**
```bash
debug_inputdata.py test_tensor.hdf5
rabbit_plot_inputdata.py test_tensor.hdf5 -o results/
rabbit_print_pulls_and_constraints.py results/fitresults.hdf5
rabbit_print_impacts.py results/fitresults.hdf5 -s
```

## Linting

The CI and pre-commit hooks enforce `isort`, `black` (line length 88), `flake8`, `autoflake`, and `pylint`. Activate pre-commit hooks once after cloning:

```bash
git config --local include.path ../.gitconfig
```

Run linters manually:
```bash
isort . --profile black --line-length 88
black .
flake8 . --max-line-length 88 --select=F401,F402,...
```

The pre-commit hook runs `autoflake`, `isort`, and `black` automatically on staged files, then aborts if files were modified (requiring re-staging).

## Architecture

The workflow has three stages: **write input tensor → run fit → post-process results**.

### 1. Input: `rabbit/tensorwriter.py`
`TensorWriter` constructs the input HDF5 tensor. Users call:
- `add_channel(name, axes, masked=False)` — define fit regions
- `add_process(name, channel, histogram, signal=True/False)` — add templates
- `add_systematic(name, channel, process, up_hist, down_hist)` — add variations
- `write(filename)` — serialize to HDF5

Supports dense and sparse tensor representations, symmetric/asymmetric systematics, and log-normal or normal systematic types.

### 2. Fit: `rabbit/fitter.py`
`Fitter` takes a `FitInputData` object (loaded from HDF5 by `rabbit/inputdata.py`), a `ParamModel`, and options. It builds a differentiable negative log-likelihood using TensorFlow and minimizes it via SciPy. Results are written through a `Workspace`.

### 3. Output: `rabbit/workspace.py`
`Workspace` collects post-fit histograms, covariance matrices, impacts, and likelihood scans into an HDF5 output file using `hist.Hist` objects (via the `wums` library).

### Param Models: `rabbit/param_models/param_model.py`
Base class `ParamModel` defines `compute(param)` which returns a `[1, nproc]` tensor scaling process yields. Each model declares:
- `nparams`: total parameters (POIs + model nuisances)
- `npoi`: true parameters of interest (first `npoi` entries; reported as POIs in outputs)
- `npou`: model nuisance parameters (`nparams - npoi`; reported as nuisances in outputs)

Built-in models:
- `Mu` (default): one POI per signal process
- `Ones`: no parameters (all yields fixed to MC)
- `Mixture`: mixing POIs between pairs of processes
- `ABCD`: data-driven background estimation using four regions; `npoi=0`, `npou=3*n_bins`. CLI: `--paramModel ABCD <process> <ch_A> [ax:val ...] <ch_B> [ax:val ...] <ch_C> [ax:val ...] <ch_D> [ax:val ...]` where `ax:val` pairs optionally select a single bin along a named axis (e.g. `iso:0`). Regions A, B, C have free parameters; D is predicted as `a*c/b` times an MC correction factor.
- `SmoothABCD`: like ABCD but one axis is parameterised with an exponential Chebyshev polynomial `val(x)=exp(p_0·T_0(x̃)+p_1·T_1(x̃)+...)` (x̃ ∈ [-1, 1] via the axis edges) instead of per-bin free parameters. CLI: `--paramModel SmoothABCD <axis> [order:N] <process> <ch_A> ... <ch_D>`. Default order=1 (log-linear). Reduces parameters from `3·n_bins` to `3·n_outer·(order+1)`.
- `ExtendedABCD`: 6-region ABCD using two sideband bins in the x direction (Ax, Bx further from signal, A/B in the middle). Fake rate is log-linearly extrapolated: `D = C·Ax·B² / (Bx·A²)`. `npoi=0`, `npou=5·n_bins`. CLI: `--paramModel ExtendedABCD <process> <ch_Ax> [ax:val ...] <ch_Bx> [ax:val ...] <ch_A> [ax:val ...] <ch_B> [ax:val ...] <ch_C> [ax:val ...] <ch_D> [ax:val ...]`.
- `SmoothExtendedABCD`: like `ExtendedABCD` but all five free-parameter regions (A, B, C, Ax, Bx) are parameterised with an exponential Chebyshev polynomial along one smoothing axis (same basis as `SmoothABCD`). `npoi=0`, `npou=5·n_outer·(order+1)`. CLI: `--paramModel SmoothExtendedABCD <axis> [order:N] <process> <ch_Ax> [ax:val ...] <ch_Bx> [ax:val ...] <ch_A> [ax:val ...] <ch_B> [ax:val ...] <ch_C> [ax:val ...] <ch_D> [ax:val ...]`.

Custom models are loaded by providing a dotted Python path (e.g. `--paramModel mymod.MyModel`); the module must be on `$PYTHONPATH` with an `__init__.py`.

### Mappings: `rabbit/mappings/`
Base class `Mapping` in `mapping.py` defines `compute_flat(params, observables)`, which is a differentiable transformation of the flat bin vector. The framework propagates uncertainties through it via automatic differentiation (`tf.GradientTape`). Built-in mappings (`Select`, `Project`, `Normalize`, `Ratio`, `Normratio`) live in `project.py` and `ratio.py`. Custom mappings follow the same pattern as param models.

### Bin scripts: `bin/`
Entry points registered in `pyproject.toml`. The main one is `rabbit_fit.py`; others are diagnostic/plotting scripts. All use `rabbit/parsing.py` for shared CLI arguments.

## Custom Extensions

Custom mappings and param models must:
1. Subclass `Mapping` or `ParamModel` respectively
2. Implement `compute_flat` (mapping) or `compute` (param model) as TF-differentiable functions
3. Be importable from `$PYTHONPATH` (directory needs `__init__.py`)
4. Be referenced with dotted module path: `-m my_package.MyMapping` or `--paramModel my_package.MyModel`

See `tests/my_custom_mapping.py` and `tests/my_custom_model.py` for examples.
