<p align="center">
  <img src="data/logo/logo.png" alt="Framework logo" width="180"/>
</p>

rabbit is a Python package for binned profile likelihood fits in high-energy physics, exploiting state-of-the-art differential programming.
Computations are based on the TensorFlow 2 library for the multithreading support on CPU and GPU, interfaced to SciPy minimizers.
Implemented approximations in the limit of large sample size simplify intensive computations.

## Tutorials and talks

Jupyter notebook tutorials are available in [notebooks/](notebooks/):
- [Tutorial 1: Getting started](notebooks/tutorial_1_getting_started.ipynb)
- [Tutorial 2: Advanced topics (mappings, masked channels, POI models)](notebooks/tutorial_2_advanced.ipynb)

Talks given about rabbit:
- [ACAT 2025](https://indico.cern.ch/event/1488410/contributions/6561542/)
- [PyHEP 2025](https://indico.cern.ch/event/1566263/timetable/#1-efficient-binned-profile-lik)

## Install

You can install rabbit via pip. It can be installed with the core functionality:
```bash
pip install rabbit-fit
```
Or with optional dependencies to use the plotting scripts
```bash
pip install rabbit-fit[plotting]
```

### Get the code

If you want to have more control or want to develop rabbit you can check it out as a (sub) module.

```bash
MY_GIT_USER=$(git config user.github)
git clone git@github.com:$MY_GIT_USER/rabbit.git
cd rabbit/
git remote add upstream git@github.com:WMass/rabbit.git
```

Get updates from the central repository (and main branch)
```bash
git pull upstream main
git push origin main
```

It can be run within a comprehensive singularity (recommended) or in an environment set up by yourself. 
It makes use of the [wums](https://pypi.org/project/wums) package for storing hdf5 files in compressed format.

### In a python virtual environment
The simplest is to make a python virtual environment. It depends on the python version you are working with (tested with 3.9.18).
First, make a python virtual environment, e.g. in the rabbit base directory (On some machines you have to use `python3`):
```bash
python -m venv env
```
Then activate it and install the necessary packages
```bash
source env/bin/activate
pip install wums[pickling,plotting] tensorflow tensorflow-probability numpy h5py hist scipy matplotlib mplhep seaborn pandas plotly kaleido
```
The packages `matplotlib`, `mplhep`, `seaborn`, `pandas`, `plotly`, and `kaleido` are only needed for the plotting scripts. 
For the `text2hdf5.py` conversion also the `uproot` package is needed.
In case you want to contribute to the development, please also install the linters `isort`, `flake8`, `autoflake`, `black`, and `pylint` used in the pre-commit hooks and the github CI
Deactivate the environment with `deactivate`.

### In singularity
The singularity includes a comprehensive set of packages. 
But the singularity is missing the `wums` package, you have to check it out as a submodule.
It also comes with custom optimized builds that for example enable numpy and scipy to be run with more than 64 threads (the limit in the standard build).
Activate the singularity image (to be done every time before running code). 
```bash
singularity run /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling\:latest
```

### Run the code
Setting up environment variables and python path (to be done every time before running code).
```bash
source setup.sh
```

## Making the input tensor
An example can be found in `tests/make_tensor.py`. Run it with:
```bash
python tests/make_tensor.py -o test_tensor.hdf5
```

### Systematic uncertainties
Systematic uncertainties are implemented by default using a log-normal probability density (with a multiplicative effect on the event yield).  Gaussian uncertainties with an additive effect on the event yield can also be used.  This is configured through the `systematic_type` parameter of the `TensorWriter`.

### Sparse tensor
By setting `sparse=True` in the `TensorWriter` constructor the tensor is stored in the sparse representation. 
This is useful when working with a sparse tensor, e.g. having many bins/processes/systematics where each bin/process/systematic only contributes to a small number of bins/processes/systematics. 
This is often the case in the standard profile likelihood unfolding. 

### Symmetrization
By default, systematic variations are asymmetric. 
However, defining only symmetric variations can be beneficial as a fully symmetric tensor has reduced memory consumption, simplifications in the likelihood function in the fit, and is usually numerically more stable. 
Different symmetrization options are supported:
 * `"average"` (default): Symmetrize by taking the average of the up and down variations.
 * `"conservative"`: Symmetrize by taking the larger of the two variations by magnitude.
 * `"linear"`: Split the asymmetric variation into two symmetric ones (average and half-difference), where the difference term models a piecewise linear dependence on the nuisance parameter. Produces two systematics: `<name>SymAvg` and `<name>SymDiff`.
 * `"quadratic"`: Like `"linear"` but the difference term is scaled by `sqrt(3)`, modeling a quadratic dependence on the nuisance parameter.
If a systematic variation is added by providing a single histogram, the variation is mirrored.

### Masked channels
Masked channels can be added that don't contribute to the likelihood but are evaluated as any other channel. 
This is done by defining `masked=True` in the `tensorwriter` `add_channel` function. 
(Pseudo) Data histograms for masked channels are not supported.
This is useful for example to compute unfolded (differential) cross sections and their uncertainties, including global impacts, taking into account all nuisance parameters that affect these channels.

### text2hdf5
The input tensor can also be generated from the input used for the [Combine tool](https://link.springer.com/article/10.1007/s41781-024-00121-4)  using the `text2hdf5.py` command.
This script is mainly intended for users that have these inputs already and want to perform some cross checks.
Only basic functionality is supported and for complex models the conversion can take long, it is thus recommended to directly produce the input tensor using the provided interface as explained above. 

### Diagnostics
Scripts for diagnosing the input tensor are available:
Running some checks for empty bins etc.
```bash
debug_inputdata.py test_tensor.hdf5
```
Plotting the histograms that are actually used in the fit, supporting adding of systematic variations in the plot:
```bash
rabbit_plot_inputdata.py test_tensor.hdf5 -o results/
```


## Run the fit
For example:
```bash
rabbit_fit.py test_tensor.hdf5 -o results/fitresult.hdf5 -t 0 --doImpacts --globalImpacts --saveHists --computeHistErrors
```

### Bin-by-bin statistical uncertainties
Bin-by-bin statistical uncertainties on the templates are added by default and can be disabled at runtime using the `--noBinByBinStat` option. The Barlow-Beeston lite method is used to add implicit nuisance parameters for each template bin.  By default this is implemented using a gamma distribution for the probability density, but Gaussian uncertainties can also be used with `--binByBinStatType normal`.

### Mappings
Perform mappings on the parameters and observables (the histogram bins in the (masked) channels). 
Baseline mappings are defined in `rabbit/mappings/` and can be called in `rabbit_fit` with the `--mapping` or `-m` option e.g. `-m Select ch0 -m Project ch1 b`. 
The first argument is the mapping name followed by arguments passed into the mapping.
Available mappings are:
 * `BaseMapping`: Compute histograms in all bins and all channels.
 * `Select`: To select histograms of a channel, and perform a selection of processes and bins, supporting rebinning.
 * `Project`: To project histograms to lower dimensions, respecting the covariance matrix across bins.
 * `Normalize`: To normalize histograms to their sum (and project them) e.g. to compute normalized differential cross sections.
 * `Ratio`: To compute the ratio between channels, processes, or histogram bins.
 * `Normratio`: To compute the ratio of normalized histograms.

Mappings can be specified on the command line and can feature different parsing syntax.
A convention is set up for parsing process and axes selections (e.g. in the `Select` and `Ratio` mappings). For selecting processes a comma separated list, e.g. <process_0>,<process_1>...
and for axes selections <axis_name_0>:<selection_0>,<axis_name_1>:<selection_1>,... i.e. a comma separated list of axis names and selections separated by ":".
Selections can be
- integers for bin indices,
- `slice()` objects e.g. `slice(0j,2,2)` where `j` can be used to index by axis value,
- `sum` to sum all bins of an axis,
- `rebin()` to rebin an axis with new edges,
- `None:None` for which `None` is returned, indicating no selection
Multiple selection per axis can be specified, e.g. `x:slice(2,8),x:sum`.

Custom mappings can be defined.
They can be specified with the full path to the custom mapping e.g. `-m custom_mapping.MyCustomMapping`. 
The path must be accessible from your `$PYTHONPATH` variable and an `__init__.py` file must be in the directory.

### POI models
POI models can be used to introduce parameters of interest (POIs) and modify the number of predicted events in the fit.
Baseline models are defined in `rabbit/poi_models/` and can be called in `rabbit_fit` with the `--poiModel` option, e.g. `--poiModel Mu`.
Only one POI model can be used at a time.
Available POI models are:
* `Mu`: Scale the number of events for each signal process with an unconstrained parameter, and background processes with 1. This is the default model.
* `Ones`: Return ones, i.e. leave the number of predicted events the same.
* `Mixture`: Scale the `primary` processes by `x` and the `complementary` processes by `1-x`

Custom POI models can be defined.
They can be specified with the full path to the custom mapping e.g. `--poiModel custom_model.MyCustomModel`. 
The path must be accessible from your `$PYTHONPATH` variable and an `__init__.py` file must be in the directory.

## Fit diagnostics

Parameter values and their uncertainties:
```bash
rabbit_print_pulls_and_constraints.py results/fitresult.hdf5
```

Uncertainty breakdown for parameter of interest, sometimes referred to nuisance parameter impacts:
```bash
rabbit_print_impacts results/fitresult.hdf5
```


## Contributing to the code

We use pre-commit hooks and linters in the CI. Activate git pre-commit hooks (only need to do this once when checking out)
```
git config --local include.path ../.gitconfig
```
In case rabbit is included as a submodule, use instead:
```
git config --local include.path "$(git rev-parse --show-superproject-working-tree)/.gitconfig"
```
