# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Jupyter notebooks (plus one standalone script) for analysing and plotting outputs of
[PROTEUS](https://github.com/FormingWorlds/PROTEUS) coupled planet-evolution simulations,
mostly grids of models and Bayesian-optimisation (BayesOpt) inference runs. It produces
publication figures for the author's papers. There is no build system, package metadata,
test suite or linter config; work is done interactively in notebooks.

Licence is GPLv3; the README asks that the author be contacted before the code is reused in
a publication.

## Environment

Use the `proteus` conda env (`~/miniforge3/envs/proteus`). The system `python3` cannot import
`proteus`. The env provides editable installs of sibling repos in `~/Projects/`
(`PROTEUS`, `CALLIOPE`, `InferAGNI`), plus `netCDF4`, `cmcrameri`, `toml`, `feathers`, `lisa`.

```bash
~/miniforge3/envs/proteus/bin/python l9859_proposal_grids.py     # standalone grid analysis
~/miniforge3/envs/proteus/bin/jupyter lab                        # notebooks
```

Run notebooks and scripts from the repo root: imports use `import utils.load_grid as lg`, and
data/output paths are relative (`data/...`, `output/...`). There is no `__init__.py` in
`utils/`; it works as a namespace package from the repo root only.

## Data layout (gitignored, symlinked)

- `data` -> `../PROTEUS/output` (the live PROTEUS output tree).
- `data_extract` -> a frozen extract of BayesOpt paper data under `~/Documents/My_articles/...`.
- `output/` holds generated figures (PDF/PNG). Also gitignored, as are `*.nc`, `*.csv`, `*.dat`,
  `*.tsv` and anything matching `nogit*`.

A PROTEUS **grid** directory contains `case_NNNNN/` subfolders. Each case (and each single run)
has:
- `init_coupler.toml`: the resolved config, read through `proteus.config.read_config_object`.
- `runtime_helpfile.csv`: whitespace-separated time series of scalar quantities, one row per
  coupling step. Most scalar analysis reads this file.
- `status`: integer code on the first line. `utils/load_grid.get_cases(only_completed=True)`
  treats codes 10–19 as completed.
- `data/<year>_atm.nc`: atmosphere profiles (NetCDF); `data/<year>.json`: interior snapshots.
- `offchem/`, `observe/` (e.g. `obs_synth_transit.csv`), `plots/`.

BayesOpt runs are named `bayesopt_<planet>/` (ground truth: `se`, `sn`, `tr`) and
`bayesopt_infer_<planet>_<kernel>_<acqf>_<workers>_<steps>/` (inference runs).

Many notebooks hard-code `pgrid_dir`/`SIM_DIR`/`outdir` paths into `data/shared/...`,
`data/article_data/...` or `/data/hen28/...` that no longer exist on this machine. Check that a
path exists before assuming a notebook will run, and ask which dataset to use rather than guessing.

## Code structure

- `utils/load_grid.py`: shared helpers for grids. `get_cases`/`get_statuses` find case dirs;
  `load_configs` + `access_configs(cfgs, "section.key")` pull config values across cases
  (`descend_get` walks dotted keys); `load_helpfiles` returns per-case DataFrames plus `hvars`,
  a dict of ragged object arrays indexed `[case][timestep]`, sliced with `access_hvars`;
  `readncdf`/`readjson` read single snapshots; `interp_2d` interpolates over 2D parameter
  planes, switching to log scaling when a range spans more than 2 orders of magnitude. It
  `import *`s `proteus.utils.constants`, so notebooks pick up constants such as `R_earth` from it.
- `utils/ppr.py`: reader for post-processed `ppr.nc` spectral output and a Planck function.
- `l9859_proposal_grids.py`: self-contained script for the L 98-59 c/d grids. Paths are anchored
  to the script dir, and the target planet is switched by editing the module-level constants
  (`grid_name`, `GRID_KEYS`, observed mass/radius/temperature). `main()` writes
  `output/<grid_name>_analysis/summary_grid.csv` and figures. It uses PROTEUS APIs
  (`proteus.atmos_clim.common.read_ncdf_profile`, `proteus.utils.plot.get_colour`) rather than
  `utils/load_grid.py`.
- Current BayesOpt paper work: `bayesopt.ipynb` (inference runs and convergence),
  `bayesopt_static.ipynb` (InferAGNI static retrievals compared with evolution; reads
  `data_extract/`), `bayesopt_sensitivity.ipynb` (CALLIOPE outgassing parameter sweeps).
- `old/`: retired notebooks, kept for reference.

## Conventions

- Species colours and labels come from `proteus.utils.plot` (`get_colour`, `latexify`) to match
  the PROTEUS ecosystem; colormaps come from `cmcrameri`.
- Figures are saved to `output/` as PDF with `bbox_inches='tight'`.

## Known defects in `utils/load_grid.py`

Found by reading the code, not by running it:
- `load_netcdfs_end` calls `read_nc`, which is undefined (the reader is `readncdf`).
- `descend_set` indexes `bits[3]` at depth 2 (should be `bits[2]`).
- `get_common_years` passes a `set` to `np.array`, which gives a 0-d object array, so the
  indexing that follows does not work.
- `composition_bars.ipynb` imports `utils.load_cmaps`, which does not exist.
