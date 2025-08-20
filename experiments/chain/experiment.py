import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

import os, time
from omegaconf import open_dict, OmegaConf

from experiments.base_experiment import BaseExperiment
from experiments.dataset import Dataset, load_dataset
from experiments.logger import LOGGER
from experiments.multiplicity.experiment import MultiplicityExperiment
from experiments.kinematics.jet_experiment import JetKinematicsExperiment
from experiments.kinematics.experiment import KinematicsExperiment
from experiments.kinematics.plots import plot_kinematics
from experiments.coordinates import fourmomenta_to_jetmomenta
from experiments.utils import get_device, GaussianFourierProjection


class ChainExperiment(BaseExperiment):
    """
    Chain experiment that runs multiplicity -> jets -> constituents sampling in sequence.
    """

    def __init__(self, cfg, rank=0, world_size=1):
        super().__init__(cfg, rank, world_size)

        self.multiplicity_exp = None
        self.jet_exp = None
        self.constituents_exp = None

    def init_physics(self):
        """Load configurations from previous experiment directories and set up subexperiments"""
        self._init_multiplicity_config()
        self._init_jet_config()
        self._init_constituents_config()

    def _init_multiplicity_config(self):
        """Load multiplicity experiment config from directory and set up for chaining"""
        mult_path = self.cfg.experiment_paths.multiplicity
        mult_config_path = os.path.join(
            mult_path, f"config_{self.cfg.model_run_indices.multiplicity}.yaml"
        )

        LOGGER.info(f"Loading multiplicity config from {mult_config_path}")
        mult_cfg = OmegaConf.load(mult_config_path)

        with open_dict(mult_cfg):
            mult_cfg.warm_start_idx = self.cfg.model_run_indices.multiplicity
            mult_cfg.original_run_dir = mult_cfg.run_dir
            mult_cfg.new_run_dir = os.path.join(self.cfg.run_dir, "multiplicity")
            mult_cfg.run_name = f"chained_mult_{self.cfg.run_idx}"
            mult_cfg.run_idx = self.cfg.run_idx
            mult_cfg.train = False
            mult_cfg.evaluate = True
            mult_cfg.evaluation.sample = True
            mult_cfg.evaluation.save_samples = True
            mult_cfg.evaluation.load_samples = False
            mult_cfg.plot = True

        self.multiplicity_exp = MultiplicityExperiment(
            mult_cfg, self.rank, self.world_size
        )

    def _init_jet_config(self):
        """Load jets experiment config from directory and set up for chaining"""
        jets_path = self.cfg.experiment_paths.jets
        jets_config_path = os.path.join(
            jets_path, f"config_{self.cfg.model_run_indices.jets}.yaml"
        )

        LOGGER.info(f"Loading jets config from {jets_config_path}")
        jet_cfg = OmegaConf.load(jets_config_path)

        with open_dict(jet_cfg):
            jet_cfg.warm_start_idx = self.cfg.model_run_indices.jets
            jet_cfg.new_run_dir = os.path.join(self.cfg.run_dir, "jets")
            jet_cfg.run_name = f"chained_jets_{self.cfg.run_idx}"
            jet_cfg.run_idx = self.cfg.run_idx

            jet_cfg.train = False
            jet_cfg.evaluate = True
            jet_cfg.evaluation.sample = True
            jet_cfg.evaluation.save_samples = True
            jet_cfg.evaluation.load_samples = False
            jet_cfg.plot = True

        self.jet_exp = JetKinematicsExperiment(jet_cfg, self.rank, self.world_size)

    def _init_constituents_config(self):
        """Load constituents experiment config from directory and set up for chaining"""
        const_path = self.cfg.experiment_paths.constituents
        const_config_path = os.path.join(
            const_path, f"config_{self.cfg.model_run_indices.constituents}.yaml"
        )

        LOGGER.info(f"Loading constituents config from {const_config_path}")
        const_cfg = OmegaConf.load(const_config_path)

        with open_dict(const_cfg):
            const_cfg.warm_start_idx = self.cfg.model_run_indices.constituents
            const_cfg.original_run_dir = const_cfg.run_dir
            const_cfg.new_run_dir = os.path.join(self.cfg.run_dir, "constituents")
            const_cfg.run_name = f"chained_constituents_{self.cfg.run_idx}"
            const_cfg.run_idx = self.cfg.run_idx

            const_cfg.train = False
            const_cfg.evaluate = True
            const_cfg.evaluation.sample = True
            const_cfg.evaluation.save_samples = True
            const_cfg.evaluation.load_samples = False
            const_cfg.plot = True

        self.constituents_exp = KinematicsExperiment(
            const_cfg, self.rank, self.world_size
        )

    def init_data(self):
        """Data initialization handled by individual subexperiments"""
        pass

    def init_subexperiments(self):
        """Initialize all sub-experiments with their models"""
        LOGGER.info("Initializing multiplicity experiment...")
        self.multiplicity_exp._init()
        self.multiplicity_exp.init_physics()
        self.multiplicity_exp.init_model()
        self.multiplicity_exp.init_data()
        self.multiplicity_exp._init_dataloader()

        with open_dict(self.multiplicity_exp.cfg):
            self.multiplicity_exp.cfg.run_dir = self.multiplicity_exp.cfg.new_run_dir
            self.multiplicity_exp.cfg.run_idx = self.cfg.run_idx

        LOGGER.info("Initializing jet experiment...")
        self.jet_exp._init()
        self.jet_exp.init_physics()
        self.jet_exp.init_model()
        self.jet_exp.init_data()
        self.jet_exp._init_dataloader()

        with open_dict(self.jet_exp.cfg):
            self.jet_exp.cfg.run_dir = self.jet_exp.cfg.new_run_dir
            self.jet_exp.cfg.run_idx = self.cfg.run_idx

        LOGGER.info("Initializing constituents experiment...")
        self.constituents_exp._init()
        self.constituents_exp.init_physics()
        self.constituents_exp.init_model()
        self.constituents_exp.init_data()
        self.constituents_exp._init_dataloader()

        with open_dict(self.constituents_exp.cfg):
            self.constituents_exp.cfg.run_dir = self.constituents_exp.cfg.new_run_dir
            self.constituents_exp.cfg.run_idx = self.cfg.run_idx

    def sample_chain(self):
        """
        Run the full sampling chain: multiplicity -> jets -> constituents
        Each experiment uses its own data pipeline, but we patch the gen-level data
        """
        LOGGER.info("Starting chained sampling...")

        LOGGER.info("Step 1: Sampling multiplicities...")
        self.multiplicity_exp.evaluate()

        mult_samples_path = os.path.join(
            self.cfg.run_dir, "multiplicity", f"samples_{self.cfg.run_idx}"
        )
        mult_samples = torch.load(os.path.join(mult_samples_path, "samples.pt"))
        self.sampled_multiplicities = mult_samples[:, :1].to(dtype=torch.int64)

        LOGGER.info("Step 2: Sampling jet kinematics...")

        self.jet_exp.evaluate(self.sampled_multiplicities)

        jet_samples_path = os.path.join(
            self.cfg.run_dir, "jets", f"samples_{self.cfg.run_idx}"
        )
        self.sampled_jets = fourmomenta_to_jetmomenta(
            torch.load(
                os.path.join(jet_samples_path, "samples.pt"), weights_only=False
            ).jet_gen
        )

        LOGGER.info("Step 3: Sampling constituents...")

        self.constituents_exp.evaluate(self.sampled_multiplicities, self.sampled_jets)

        LOGGER.info("Chained sampling complete.")

    def evaluate(self):
        """Run the full chained evaluation"""
        self.init_subexperiments()
        self.sample_chain()

    def plot(self):
        """Create plots for chained experiment"""

        LOGGER.info("Plotting multiplicity...")
        self.multiplicity_exp.plot()
        LOGGER.info("Plotting jets...")
        self.jet_exp.plot()
        LOGGER.info("Plotting constituents...")
        self.constituents_exp.plot()
        LOGGER.info("Plotting complete.")

    def _init_loss(self):
        """Not used in chained experiment"""
        pass

    def _batch_loss(self, batch):
        """Not used in chained experiment"""
        pass

    def _init_metrics(self):
        """Not used in chained experiment"""
        return {}

    def init_geometric_algebra(self):
        """Initialize geometric algebra settings - delegate to sub-experiments"""
        pass

    def init_model(self):
        """Models are loaded separately for each sub-experiment"""
        pass

    def _init_dataloader(self):
        """Data loaders handled per sub-experiment"""
        pass

    def _init_backend(self):
        """Backend handled per sub-experiment"""
        self.device = get_device()

    def _init_optimizer(self):
        """Not used in chained experiment"""
        pass

    def _init_scheduler(self):
        """Not used in chained experiment"""
        pass

    def train(self):
        """Not used in chained experiment"""
        pass

    def _save_model(self):
        """Not used in chained experiment"""
        pass
