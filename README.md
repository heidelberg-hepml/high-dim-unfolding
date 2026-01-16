<div align="center">

## Generative Unfolding of Jets and Their Substructure

[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![high-dim-unfolding](http://img.shields.io/badge/paper-arxiv.2510.19906-B31B1B.svg)](https://arxiv.org/abs/2510.19906)

</div>

This project implements the framework introduced in *Generative Unfolding of Jets and Their Substructure*. The code trains conditional flow-matching (CFM) generative models with Lorentz-equivariant transformers to unfold detector-level observations to particle level in several hundred dimensions. The pipeline factorises the task into three specialised stages — multiplicity, jet kinematics, and constituent structure — and can be chained into an end-to-end generator that produces fully unfolded jets ready for physics analysis. The codebase is forked from the original [Lorentz-GATr repository](https://github.com/heidelberg-hepml/lorentz-gatr.git) and uses the newer [`lgatr` library](https://github.com/heidelberg-hepml/lgatr.git) for L-GATr blocks.

## Citation

If you find this code useful in your research, please cite our paper

```bibtex
@article{Petitjean:2025tgk,
    author = "Petitjean, Antoine and Butter, Anja and Greif, Kevin and Palacios Schweitzer, Sofia and Plehn, Tilman and Spinner, Jonas and Whiteson, Daniel",
    title = "{Generative Unfolding of Jets and Their Substructure}",
    eprint = "2510.19906",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "10",
    year = "2025"
}
```

## Highlights
- Use of `pytorch-geometric` graph objects for variable-length events with jets and constituents information at detector-level and particle-level.
- Conditional flow-matching training of transformer-based architectures.
- Lorentz-equivariant CFM models via the `lgatr` library.
- Modular experiments (`Multiplicity`, `JetKinematics`, `Kinematics`,
  `Chain`).

## Repository Layout
- `experiments/`: experiment base classes, data handling, embedding,
  conditional flow-matching models, utilities, and baseline networks.
  - `experiments/baselines`: baseline models, as opposed to the models imported from the `lgatr` package
  - `experiments/multiplicity`: multiplicity experiment files
  - `experiments/kinematics`: CFM-based experiments files, split into jet kinematics unfolding and constituents unfolding
  - `experiments/chain`: Sequential sampling files calling for the other experiments
- `config/`: Hydra configuration trees for experiments and models
- `runs/`: default output location for trained checkpoints, configs, plots,
  and optional MLflow metadata.

## Setup
1. **Environment**
   ```bash
   uv venv
   source .venv/bin/activate
   uv sync
   uv run pre-commit install   # install git hooks for ruff
   ```
   Ensure the PyTorch build matches your CUDA toolkit.  Installing `xformers`
   may require wheels specific to your platform.

2. **Code quality and tests**
   ```bash
   uv run ruff format          # format in place
   uv run ruff check           # lint
   uv run pytest tests         # unit tests
   ```

3. **Datasets**
   Place the datasets under `data/`. Existing configs
   expect the EnergyFlow `zplusjet` dataset or our generated top dataset, available upon request.
   See `experiments/dataset.py` to add new datasets.

4. **FastJet (optional)**
   Some substructure observables rely on `fastjet`/`fastjet contribs`. Our custom python bindings for `fastjet contribs` are available [here](https://github.com/AntoinePTJ/pybind_fastjet_contribs). If this package is missing from the python venv, the code will skip related imports and plots.
   A example script to easily install `fastjet`, `fastjet contribs` and the python bindings is provided in `fastjet_script.sh`. Make sure to set the correct path at the start of the script.

## Running Experiments

Runs parameters are set via Hydra configs. There are different configuration files for each experiment type.
```bash
uv run python run.py --config-name multiplicity
uv run python run.py --config-name jets
uv run python run.py --config-name constituents
```

### Individual runs
```bash
uv run python run.py -cn constituents \
    exp_name=z_constituents \
    run_name=lgatr_200k \
    data.dataset=zplusjet \
    training.iterations=200000 \
    model=cond_lgatr_constituents
```
The model config is loaded from `config/model/<model_name>.yaml`. It has to correspond to the chosen experiment.

Outputs are stored in `runs/<exp_name>/<run_name>/`, including:
- `config.yaml`, `config_<run_idx>.yaml`: frozen configs.
- `models/model_run*.pt`: checkpoints (model, optimizer, EMA, scheduler).
- `out_<run_idx>.log`: aggregated log.
- `plots/`: PDFs summarising losses and chosen observables.
- `source.zip` (optional): zipped source code at the beginning of the run.

### Chained Generation
Provide paths to a previous run for each experiment:
```bash
uv run python run.py --config-name chain \
    experiment_paths.multiplicity=/path/to/mult/run_dir \
    experiment_paths.jets=/path/to/jets/run_dir \
    experiment_paths.constituents=/path/to/const/run_dir \
```
The chain will load the specified checkpoints, sample multiplicities, jet
kinematics, then constituents, and generate evaluation plots.
