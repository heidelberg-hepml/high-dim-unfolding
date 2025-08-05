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
    Uses detector-level conditions consistently across all three stages.
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
        mult_cfg.data = self.cfg.data.copy()
        mult_cfg.run_dir = self.cfg.multiplicity.model_path
        mult_cfg.warm_start_idx = self.cfg.multiplicity.run_idx
        mult_cfg.evaluation.load_samples = False
        mult_cfg.evaluation.sample = True
        mult_cfg.train = False
        mult_cfg.evaluate = True
        
        self.multiplicity_exp = MultiplicityExperiment(mult_cfg, self.rank, self.world_size)

    def _init_jet_config(self):
        """Initialize jet experiment with saved model"""  
        jet_cfg = self.cfg.jets.copy()
        jet_cfg.data = self.cfg.data.copy()
        jet_cfg.run_dir = self.cfg.jets.model_path
        jet_cfg.warm_start_idx = self.cfg.jets.run_idx
        jet_cfg.evaluation.load_samples = False
        jet_cfg.evaluation.sample = True
        jet_cfg.train = False
        jet_cfg.evaluate = True
        
        self.jet_exp = JetKinematicsExperiment(jet_cfg, self.rank, self.world_size)

    def _init_constituents_config(self):
        """Initialize constituents experiment with saved model"""
        const_cfg = self.cfg.constituents.copy()
        const_cfg.data = self.cfg.data.copy() 
        const_cfg.run_dir = self.cfg.constituents.model_path
        const_cfg.warm_start_idx = self.cfg.constituents.run_idx
        const_cfg.evaluation.load_samples = False
        const_cfg.evaluation.sample = True
        const_cfg.train = False
        const_cfg.evaluate = True
        
        self.constituents_exp = KinematicsExperiment(const_cfg, self.rank, self.world_size)

    def init_data(self):
        """Initialize data for the chained experiment"""
        t0 = time.time()
        data_path = os.path.join(self.cfg.data.data_dir, f"{self.cfg.data.dataset}")
        LOGGER.info(f"Creating chained experiment data from {data_path}")
        
        # Initialize device and dtype
        self.device = get_device()
        self.dtype = torch.float32
        
        # Load the original data
        data = self.load_fn(data_path, self.cfg.data, self.dtype)
        
        # Store detector-level data that will be used as condition throughout
        self.det_particles = data["det_particles"]
        self.det_mults = data["det_mults"]
        self.det_pids = data["det_pids"]
        self.det_jets = data["det_jets"]
        
        # Store generator-level truth for comparison
        self.gen_particles = data["gen_particles"]
        self.gen_mults = data["gen_mults"]
        self.gen_pids = data["gen_pids"]
        self.gen_jets = data["gen_jets"]
        
        size = len(self.det_particles)
        split = self.cfg.data.train_val_test
        train_idx, val_idx, test_idx = np.cumsum([int(s * size) for s in split])
        self.test_idx = (val_idx, test_idx)
        
        LOGGER.info(f"Initialized chained experiment data in {time.time() - t0:.2f} seconds")

    def init_subexperiments(self):
        """Initialize all sub-experiments with their models"""
        LOGGER.info("Initializing multiplicity experiment...")
        self.multiplicity_exp.init_physics()
        self.multiplicity_exp.init_data()
        self.multiplicity_exp.init_distributed()
        self.multiplicity_exp.load_model()
        
        LOGGER.info("Initializing jet experiment...")
        self.jet_exp.init_physics()
        self.jet_exp.init_data()
        self.jet_exp.init_distributed() 
        self.jet_exp.load_model()
        
        LOGGER.info("Initializing constituents experiment...")
        self.constituents_exp.init_physics()
        self.constituents_exp.init_data()
        self.constituents_exp.init_distributed()
        self.constituents_exp.load_model()

    def sample_chain(self, n_events=None):
        """
        Run the full sampling chain: multiplicity -> jets -> constituents
        """
        if n_events is None:
            n_events = self.test_idx[1] - self.test_idx[0]
            
        LOGGER.info(f"Starting chained sampling for {n_events} events")
        
        # Use test data indices
        start_idx, end_idx = self.test_idx
        det_particles_test = self.det_particles[start_idx:end_idx]
        det_jets_test = self.det_jets[start_idx:end_idx]
        det_mults_test = self.det_mults[start_idx:end_idx]
        
        # Step 1: Sample multiplicities conditioned on detector data
        LOGGER.info("Step 1: Sampling multiplicities...")
        self.sampled_multiplicities = self._sample_multiplicities(
            det_particles_test, det_jets_test, det_mults_test
        )
        
        # Step 2: Sample jet kinematics conditioned on detector jets and sampled multiplicities  
        LOGGER.info("Step 2: Sampling jet kinematics...")
        self.sampled_jets = self._sample_jets(
            det_jets_test, self.sampled_multiplicities
        )
        
        # Step 3: Sample constituents conditioned on detector data, sampled jets, and sampled multiplicities
        LOGGER.info("Step 3: Sampling constituents...")
        self.sampled_constituents = self._sample_constituents(
            det_particles_test, det_jets_test, det_mults_test,
            self.sampled_jets, self.sampled_multiplicities
        )
        
        LOGGER.info("Chained sampling complete!")

    def _sample_multiplicities(self, det_particles, det_jets, det_mults):
        """Sample multiplicities using the multiplicity model"""
        # Create dataset for multiplicity sampling
        mult_dataset = Dataset(self.dtype, pos_encoding_dim=self.cfg.data.pos_encoding_dim)
        mult_dataset.create_data_list(
            det_particles=det_particles,
            det_pids=self.det_pids[self.test_idx[0]:self.test_idx[1]],
            det_mults=det_mults,
            det_jets=det_jets.sum(dim=1, keepdim=True),  # Sum for total jet momentum
            gen_particles=torch.zeros_like(det_particles),  # Dummy
            gen_pids=torch.zeros_like(self.det_pids[self.test_idx[0]:self.test_idx[1]]),  # Dummy
            gen_mults=torch.zeros_like(det_mults),  # Dummy - will be replaced by samples
            gen_jets=torch.zeros_like(det_jets.sum(dim=1, keepdim=True)),  # Dummy
        )
        
        mult_loader = DataLoader(
            dataset=mult_dataset,
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
            follow_batch=["x_gen", "x_det"],
        )
        
        sampled_mults = []
        self.multiplicity_exp.model.eval()
        with torch.no_grad():
            for batch in mult_loader:
                batch = batch.to(self.device)
                output = self.multiplicity_exp.model(batch)
                
                # Process output through distribution using experiment's method
                from experiments.multiplicity.distributions import process_params
                params = process_params(output)
                predicted_dist = self.multiplicity_exp.distribution(params)
                
                # Sample from distribution
                sample = predicted_dist.sample()
                
                # Handle differential case if needed
                if self.multiplicity_exp.cfg.dist.diff:
                    sample = batch.x_det_ptr.diff() + sample
                    
                sampled_mults.append(sample.cpu())
        
        return torch.cat(sampled_mults, dim=0)

    def _sample_jets(self, det_jets, sampled_mults):
        """Sample jet kinematics using the jet model and sampled multiplicities"""
        # Create modified dataset where gen_mults are replaced with sampled multiplicities
        jet_dataset = Dataset(
            self.dtype, 
            pos_encoding_dim=self.cfg.data.pos_encoding_dim,
            mult_encoding_dim=getattr(self.cfg.data, 'mult_encoding_dim', 0)
        )
        
        # Create dummy particles based on sampled multiplicities
        max_particles = sampled_mults.max().item()
        dummy_particles = torch.zeros(len(sampled_mults), max_particles, 4, dtype=self.dtype)
        dummy_pids = torch.zeros(len(sampled_mults), max_particles, 6, dtype=self.dtype)
        
        jet_dataset.create_data_list(
            det_particles=dummy_particles,  # Not used in jet experiment
            det_pids=dummy_pids,  # Not used in jet experiment  
            det_mults=sampled_mults,  # Use sampled multiplicities
            det_jets=det_jets,
            gen_particles=dummy_particles,  # Dummy
            gen_pids=dummy_pids,  # Dummy
            gen_mults=sampled_mults,  # Use sampled multiplicities
            gen_jets=torch.zeros_like(det_jets),  # Will be replaced by samples
        )
        
        jet_loader = DataLoader(
            dataset=jet_dataset,
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
            follow_batch=["x_gen", "x_det"],
        )
        
        sampled_jets = []
        self.jet_exp.model.eval()
        with torch.no_grad():
            for batch in jet_loader:
                batch = batch.to(self.device)
                sample_batch, _ = self.jet_exp.model.sample(batch, self.device, self.dtype)
                sampled_jets.append(sample_batch.jet_gen.cpu())
        
        return torch.cat(sampled_jets, dim=0)

    def _sample_constituents(self, det_particles, det_jets, det_mults, sampled_jets, sampled_mults):
        """Sample constituents using the constituents model, sampled jets, and sampled multiplicities"""
        # Create modified dataset where gen_jets and gen_mults are replaced with samples
        const_dataset = Dataset(self.dtype, pos_encoding_dim=self.cfg.data.pos_encoding_dim)
        const_dataset.create_data_list(
            det_particles=det_particles,
            det_pids=self.det_pids[self.test_idx[0]:self.test_idx[1]],
            det_mults=det_mults, 
            det_jets=det_jets,
            gen_particles=torch.zeros_like(det_particles),  # Will be replaced by samples
            gen_pids=torch.zeros_like(self.det_pids[self.test_idx[0]:self.test_idx[1]]),  # Will be replaced
            gen_mults=sampled_mults,  # Use sampled multiplicities
            gen_jets=sampled_jets,  # Use sampled jets
        )
        
        const_loader = DataLoader(
            dataset=const_dataset,
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
            follow_batch=["x_gen", "x_det"],
        )
        
        sampled_constituents = []
        self.constituents_exp.model.eval()
        with torch.no_grad():
            for batch in const_loader:
                batch = batch.to(self.device)
                sample_batch, _ = self.constituents_exp.model.sample(batch, self.device, self.dtype)
                sampled_constituents.append(sample_batch.x_gen.cpu())
        
        return torch.cat(sampled_constituents, dim=0)

    def evaluate(self):
        """Run the full chained evaluation"""
        self.init_subexperiments()
        self.sample_chain()
        
        # Save the results
        if self.cfg.evaluation.save_samples:
            self._save_samples()

    def _save_samples(self):
        """Save all sampled quantities"""
        path = os.path.join(self.cfg.run_dir, f"chained_samples_{self.cfg.run_idx}")
        os.makedirs(path, exist_ok=True)
        
        LOGGER.info(f"Saving chained samples to {path}")
        
        torch.save({
            'multiplicities': self.sampled_multiplicities,
            'jets': self.sampled_jets,  
            'constituents': self.sampled_constituents,
            'det_particles': self.det_particles[self.test_idx[0]:self.test_idx[1]],
            'det_jets': self.det_jets[self.test_idx[0]:self.test_idx[1]],
            'det_mults': self.det_mults[self.test_idx[0]:self.test_idx[1]],
            'gen_particles': self.gen_particles[self.test_idx[0]:self.test_idx[1]],
            'gen_jets': self.gen_jets[self.test_idx[0]:self.test_idx[1]], 
            'gen_mults': self.gen_mults[self.test_idx[0]:self.test_idx[1]],
        }, os.path.join(path, 'chained_samples.pt'))

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