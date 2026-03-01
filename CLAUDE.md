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
`Fitter` takes a `FitInputData` object (loaded from HDF5 by `rabbit/inputdata.py`), a `POIModel`, and options. It builds a differentiable negative log-likelihood using TensorFlow and minimizes it via SciPy. Results are written through a `Workspace`.

### 3. Output: `rabbit/workspace.py`
`Workspace` collects post-fit histograms, covariance matrices, impacts, and likelihood scans into an HDF5 output file using `hist.Hist` objects (via the `wums` library).

### POI Models: `rabbit/poi_models/poi_model.py`
Base class `POIModel` defines `compute(poi)` which returns a `[1, nproc]` tensor scaling signal processes. Built-in models: `Mu` (default), `Ones`, `Mixture`. Custom models are loaded by providing a dotted Python path (e.g. `--poiModel mymod.MyModel`); the module must be on `$PYTHONPATH` with an `__init__.py`.

### Mappings: `rabbit/mappings/`
Base class `Mapping` in `mapping.py` defines `compute_flat(params, observables)`, which is a differentiable transformation of the flat bin vector. The framework propagates uncertainties through it via automatic differentiation (`tf.GradientTape`). Built-in mappings (`Select`, `Project`, `Normalize`, `Ratio`, `Normratio`) live in `project.py` and `ratio.py`. Custom mappings follow the same pattern as POI models.

### Bin scripts: `bin/`
Entry points registered in `pyproject.toml`. The main one is `rabbit_fit.py`; others are diagnostic/plotting scripts. All use `rabbit/parsing.py` for shared CLI arguments.

## Custom Extensions

Custom mappings and POI models must:
1. Subclass `Mapping` or `POIModel` respectively
2. Implement `compute_flat` (mapping) or `compute` (POI model) as TF-differentiable functions
3. Be importable from `$PYTHONPATH` (directory needs `__init__.py`)
4. Be referenced with dotted module path: `-m my_package.MyMapping` or `--poiModel my_package.MyModel`

See `tests/my_custom_mapping.py` and `tests/my_custom_model.py` for examples.
