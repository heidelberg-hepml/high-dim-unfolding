# Generative Unfolding of Jets and Their Substructure

This project implements the framework introduced in *Generative Unfolding of
Jets and Their Substructure*. The code trains
conditional flow-matching (CFM) generative models with Lorentz-equivariant
transformers to unfold detector-level observations to particle level in
several hundred dimensions. The pipeline factorises the task into three
specialised stages—multiplicity, jet kinematics, and constituent structure—
and can be chained into an end-to-end generator that produces fully unfolded
jets ready for physics analysis. The codebase is forked from the original [Lorentz-GATr repository](https://github.com/heidelberg-hepml/lorentz-gatr.git) and uses the newer [`lgatr` library](https://github.com/heidelberg-hepml/lgatr.git) for L-GATr blocks.

## Highlights
- Use of `pytorch-geometric` graph objects for variable-length events with jets, constituents, scalar information for at detector-level and particle-level.
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
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   Ensure the PyTorch build matches your CUDA toolkit.  Installing `xformers`
   may require wheels specific to your platform.

2. **Datasets**
   Place the datasets under `data/`. Existing configs
   expect the EnergyFlow `zplusjet` dataset or our generated top dataset, available upon request.
   See `experiments/dataset.py` to add new datasets.

3. **FastJet (optional)**
   Some substructure observables rely on `fastjet`/`fastjet contribs`. Our custom python bindings for `fastjet contribs` are available [here](https://github.com/AntoinePTJ/pybind_fastjet_contribs). If this package is missing from the python venv, the code will skip related imports and plots.

## Running Experiments

Runs parameters are set via Hydra configs. There are different configuration files for each experiment type.
```bash
python run.py --config-name multiplicity
python run.py --config-name jets
python run.py --config-name constituents
```

### Individual runs
```bash
python run.py -cn constituents \
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
python run.py --config-name chain \
    experiment_paths.multiplicity=/path/to/mult/run_dir \
    experiment_paths.jets=/path/to/jets/run_dir \
    experiment_paths.constituents=/path/to/const/run_dir \
```
The chain will load the specified checkpoints, sample multiplicities, jet
kinematics, then constituents, and generate evaluation plots.