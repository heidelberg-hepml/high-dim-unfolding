import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

import os, time
from omegaconf import open_dict

from experiments.base_experiment import BaseExperiment
from experiments.dataset import Dataset, load_dataset
from experiments.logger import LOGGER
from experiments.multiplicity.experiment import MultiplicityExperiment
from experiments.kinematics.jet_experiment import JetKinematicsExperiment
from experiments.kinematics.experiment import KinematicsExperiment
from experiments.utils import get_device


class ChainedExperiment(BaseExperiment):
    """
    Chained experiment that runs multiplicity -> jets -> constituents sampling in sequence.
    """

    def __init__(self, cfg, rank=0, world_size=1):
        super().__init__(cfg, rank, world_size)

        # Initialize the three sub-experiments
        self.multiplicity_exp = None
        self.jet_exp = None
        self.constituents_exp = None

        # Storage for chained samples
        self.sampled_multiplicities = None
        self.sampled_jets = None
        self.sampled_constituents = None

    def init_physics(self):
        with open_dict(self.cfg):
            # Load dataset info once
            max_num_particles, diff, pt_min, masked_dims, load_fn = load_dataset(
                self.cfg.data.dataset
            )

            self.cfg.data.max_num_particles = max_num_particles
            self.cfg.data.diff = diff
            self.cfg.data.pt_min = pt_min
            self.cfg.cfm.masked_dims = masked_dims
            self.load_fn = load_fn

            # Initialize sub-experiment configurations
            self._init_multiplicity_config()
            self._init_jet_config()
            self._init_constituents_config()

    def _init_multiplicity_config(self):
        """Initialize multiplicity experiment with saved model"""
        mult_cfg = self.cfg.multiplicity.copy()
        # Configure warm start from original model path
        with open_dict(mult_cfg):
            mult_cfg.warm_start_idx = self.cfg.multiplicity.run_idx
            mult_cfg.run_name = f"chained_mult_{self.cfg.run_idx}"
            mult_cfg.run_dir = os.path.join(self.cfg.run_dir, "multiplicity")
            mult_cfg.run_idx = 0  # New run for this chained experiment
            mult_cfg.evaluation.load_samples = False
            mult_cfg.evaluation.sample = True
            mult_cfg.evaluation.save_samples = True
            mult_cfg.train = False
            mult_cfg.evaluate = True
            # Store original run directory for model loading
            mult_cfg.original_run_dir = self.cfg.multiplicity.model_path

        self.multiplicity_exp = MultiplicityExperiment(
            mult_cfg, self.rank, self.world_size
        )

    def _init_jet_config(self):
        """Initialize jet experiment with saved model"""
        jet_cfg = self.cfg.jets.copy()
        # Configure warm start from original model path
        with open_dict(jet_cfg):
            jet_cfg.warm_start_idx = self.cfg.jets.run_idx
            jet_cfg.run_name = f"chained_jets_{self.cfg.run_idx}"
            jet_cfg.run_dir = os.path.join(self.cfg.run_dir, "jets")
            jet_cfg.run_idx = 0  # New run for this chained experiment
            jet_cfg.evaluation.load_samples = False
            jet_cfg.evaluation.sample = True
            jet_cfg.evaluation.save_samples = True
            jet_cfg.train = False
            jet_cfg.evaluate = True
            # Store original run directory for model loading
            jet_cfg.original_run_dir = self.cfg.jets.model_path

        self.jet_exp = JetKinematicsExperiment(jet_cfg, self.rank, self.world_size)

    def _init_constituents_config(self):
        """Initialize constituents experiment with saved model"""
        const_cfg = self.cfg.constituents.copy()
        # Configure warm start from original model path
        with open_dict(const_cfg):
            const_cfg.warm_start_idx = self.cfg.constituents.run_idx
            const_cfg.run_name = f"chained_constituents_{self.cfg.run_idx}"
            const_cfg.run_dir = os.path.join(self.cfg.run_dir, "constituents")
            const_cfg.run_idx = 0  # New run for this chained experiment
            const_cfg.evaluation.load_samples = False
            const_cfg.evaluation.sample = True
            const_cfg.evaluation.save_samples = True
            const_cfg.train = False
            const_cfg.evaluate = True
            # Store original run directory for model loading
            const_cfg.original_run_dir = self.cfg.constituents.model_path

        self.constituents_exp = KinematicsExperiment(
            const_cfg, self.rank, self.world_size
        )

    def init_data(self):
        """Data initialization handled by individual subexperiments"""
        pass

    def init_subexperiments(self):
        """Initialize all sub-experiments with their models"""
        LOGGER.info("Initializing multiplicity experiment...")
        self.multiplicity_exp.init_physics()
        self.multiplicity_exp.init_data()
        self.multiplicity_exp._init_dataloader()
        # Model loading handled by warm_start mechanism

        LOGGER.info("Initializing jet experiment...")
        self.jet_exp.init_physics()
        self.jet_exp.init_data()
        self.jet_exp._init_dataloader()
        # Model loading handled by warm_start mechanism

        LOGGER.info("Initializing constituents experiment...")
        self.constituents_exp.init_physics()
        self.constituents_exp.init_data()
        self.constituents_exp._init_dataloader()
        # Model loading handled by warm_start mechanism

    def sample_chain(self):
        """
        Run the full sampling chain: multiplicity -> jets -> constituents
        Each experiment uses its own data pipeline, but we patch the gen-level data
        """
        LOGGER.info("Starting chained sampling...")

        # Step 1: Sample multiplicities using original data pipeline
        LOGGER.info("Step 1: Sampling multiplicities...")
        self.multiplicity_exp.evaluate()
        # Get sampled multiplicities from the experiment's results
        mult_samples_path = os.path.join(self.cfg.run_dir, "multiplicity", f"samples_0")
        mult_samples = torch.load(os.path.join(mult_samples_path, "samples.pt"))
        self.sampled_multiplicities = mult_samples["samples"][:, 0]  # First column is samples

        # Step 2: Sample jets using original data pipeline but patch multiplicities
        LOGGER.info("Step 2: Sampling jet kinematics...")
        self._patch_jet_data_with_multiplicities(self.sampled_multiplicities)
        self.jet_exp.evaluate()
        # Get sampled jets from the experiment's results
        jet_samples_path = os.path.join(self.cfg.run_dir, "jets", f"samples_0")
        self.sampled_jets = torch.load(os.path.join(jet_samples_path, "samples.pt"))

        # Step 3: Sample constituents using original data pipeline but patch jets and multiplicities
        LOGGER.info("Step 3: Sampling constituents...")
        self._patch_constituents_data_with_samples(self.sampled_jets, self.sampled_multiplicities)
        self.constituents_exp.evaluate()
        # Get sampled constituents from the experiment's results
        const_samples_path = os.path.join(self.cfg.run_dir, "constituents", f"samples_0")
        self.sampled_constituents = torch.load(os.path.join(const_samples_path, "samples.pt"))

        LOGGER.info("Chained sampling complete!")

    def _patch_jet_data_with_multiplicities(self, sampled_multiplicities):
        """Patch the jet experiment's dataset to use sampled multiplicities as gen_mults"""
        # Replace gen_mults in test data with sampled multiplicities
        for i, data_point in enumerate(self.jet_exp.test_data):
            if i < len(sampled_multiplicities):
                data_point.mult_gen = sampled_multiplicities[i].item()

    def _patch_constituents_data_with_samples(self, sampled_jets, sampled_multiplicities):
        """Patch the constituents experiment's dataset to use sampled jets and multiplicities"""
        # Extract jet momenta from the jet samples batch
        jet_momenta = sampled_jets.jet_gen if hasattr(sampled_jets, 'jet_gen') else sampled_jets
        
        # Replace gen_jets and gen_mults in test data with samples
        for i, data_point in enumerate(self.constituents_exp.test_data):
            if i < len(sampled_multiplicities):
                data_point.mult_gen = sampled_multiplicities[i].item()
            if i < len(jet_momenta):
                data_point.jet_gen = jet_momenta[i]

    def evaluate(self):
        """Run the full chained evaluation"""
        self.init_subexperiments()
        self.sample_chain()

        # Results are already saved by individual experiments
        LOGGER.info("Chained evaluation complete. Individual experiment samples saved to run_dir.")

    def _save_samples(self):
        """Samples are saved by individual experiments in their evaluate() methods"""
        pass

    def plot(self):
        """Create plots comparing truth vs chained samples"""
        # Implementation would create comparison plots between:
        # - Truth multiplicities vs sampled multiplicities
        # - Truth jets vs sampled jets
        # - Truth constituents vs sampled constituents
        LOGGER.info("Plotting for chained experiment not yet implemented")

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
