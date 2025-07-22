import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

import os, time
from omegaconf import open_dict

from experiments.kinematics.experiment import KinematicsExperiment
from experiments.logger import LOGGER


class JetTokensKinematicsExperiment(KinematicsExperiment):
    """
    Kinematics experiment that treats each jet's 4 coordinates (E, px, py, pz) 
    as 4 separate tokens of dimension 1, similar to how constituents are handled.
    
    This experiment reuses all the data objects from the constituents experiment
    but the model learns on jet coordinates instead of constituent coordinates.
    """

    def init_physics(self):
        """Initialize physics - same as constituents but adapted for jet tokens"""
        # Call parent's init_physics to get all the standard setup
        super().init_physics()
        
        # Override specific settings for jet tokens
        with open_dict(self.cfg):
            # For jets, we don't fix mass and don't use PIDs
            self.cfg.data.add_pid = False
            # Each jet has exactly 4 tokens (E, px, py, pz)
            self.cfg.data.max_constituents = 4
            
            # Update channel dimensions for jet tokens (4 tokens of dim 1 each)
            if self.cfg.modelname == "ConditionalTransformer":
                # Base channels: 1 (single coordinate per token)
                base_channels = 1
                self.cfg.model.net.in_channels = (
                    base_channels + self.cfg.cfm.embed_t_dim + self.cfg.data.pos_encoding_dim
                )
                self.cfg.model.net_condition.in_channels = (
                    base_channels + self.cfg.data.pos_encoding_dim
                )
                if self.cfg.cfm.add_jet:
                    self.cfg.model.net.in_channels += 1
                    self.cfg.model.net_condition.in_channels += 1
                if self.cfg.cfm.self_condition_prob > 0.0:
                    self.cfg.model.net.in_channels += base_channels

            elif self.cfg.modelname == "ConditionalLGATr":
                # For LGATr, we still work with 4-vectors but in token representation
                # Keep the same channel configuration as regular constituents
                pass

    def _init_data(self, data_path):
        """Initialize data - same as constituents experiment"""
        # Use the parent's data initialization which handles all the data loading
        # and coordinate transforms properly
        super()._init_data(data_path)
        
        LOGGER.info("Jet tokens experiment will learn on jet coordinates as tokens")

    def _sample_events(self, loader):
        """Sample events - same as constituents but using jets"""
        samples = []
        targets = []
        self.data_raw = {}
        it = iter(loader)
        n_batches = self.cfg.evaluation.n_batches
        if n_batches > len(loader):
            LOGGER.warning(
                f"Requested {n_batches} batches for sampling, but only {len(loader)} batches available in test dataset."
            )
            n_batches = len(loader)
        elif n_batches == -1:
            n_batches = len(loader)
        LOGGER.info(f"Sampling {n_batches} batches for evaluation")

        for i in range(n_batches):
            batch = next(it).to(self.device)

            sample_batch, sample = self.model.sample(
                batch,
                self.device,
                self.dtype,
            )

            if i == 0:
                # Plot the first batch for visualization
                from experiments.kinematics.plots import plot_kinematics
                plot_kinematics(
                    self.cfg.run_dir,
                    batch.jet_det.detach().cpu(),  # Use jets for plotting
                    batch.jet_gen.detach().cpu(),
                    sample_batch.jet_gen.detach().cpu(),
                    f"post_jet_tokens.pdf",
                )

            samples.extend(sample_batch.detach().to_data_list())
            targets.extend(batch.detach().to_data_list())

        self.data_raw["samples"] = Batch.from_data_list(
            samples, follow_batch=["x_gen", "x_det"]
        )

        self.data_raw["truth"] = Batch.from_data_list(
            targets, follow_batch=["x_gen", "x_det"]
        )

        # convert the list into a dataloader
        sampler = torch.utils.data.DistributedSampler(
            samples,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )
        self.sample_loader = DataLoader(
            dataset=samples,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            sampler=sampler,
            follow_batch=["x_gen", "x_det"],
        )

        if self.cfg.evaluation.save_samples:
            path = os.path.join(self.cfg.run_dir, f"samples_{self.cfg.run_idx}")
            os.makedirs(os.path.join(path), exist_ok=True)
            LOGGER.info(f"Saving samples in {path}")
            t0 = time.time()
            torch.save(self.data_raw["samples"], os.path.join(path, "samples.pt"))
            torch.save(self.data_raw["truth"], os.path.join(path, "truth.pt"))
            LOGGER.info(f"Saved samples in {time.time() - t0:.2f}s")

    def plot(self):
        """Plotting - same as parent but with different model label"""
        path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(os.path.join(path), exist_ok=True)
        LOGGER.info(f"Creating plots in {path}")
        t0 = time.time()

        if self.cfg.modelname == "ConditionalTransformer":
            model_label = "CondTr-JetTokens"
        elif self.cfg.modelname == "ConditionalLGATr":
            model_label = "CondLGATr-JetTokens"
        else:
            model_label = f"{self.cfg.modelname}-JetTokens"
            
        kwargs = {
            "exp": self,
            "model_label": model_label,
        }

        # Use parent's plotting functionality
        import experiments.kinematics.plotter as plotter
        
        if self.cfg.train:
            filename = os.path.join(path, "training.pdf")
            plotter.plot_losses(filename=filename, **kwargs)

        if not self.cfg.evaluate:
            return

        weights, mask_dict = None, None

        if (
            self.cfg.evaluation.sample
            or self.cfg.evaluation.load_samples
            or self.cfg.evaluation.overfit
        ):
            if self.cfg.plotting.fourmomenta:
                filename = os.path.join(path, "fourmomenta.pdf")
                plotter.plot_fourmomenta(
                    filename=filename, **kwargs, weights=weights, mask_dict=mask_dict
                )

            if self.cfg.plotting.jetmomenta:
                filename = os.path.join(path, "jetmomenta.pdf")
                plotter.plot_jetmomenta(
                    filename=filename, **kwargs, weights=weights, mask_dict=mask_dict
                )

            if self.cfg.plotting.preprocessed:
                filename = os.path.join(path, "preprocessed.pdf")
                plotter.plot_preprocessed(
                    filename=filename, **kwargs, weights=weights, mask_dict=mask_dict
                )
                
            if self.cfg.plotting.jetscaled:
                filename = os.path.join(path, "jetscaled.pdf")
                plotter.plot_jetscaled(
                    filename=filename, **kwargs, weights=weights, mask_dict=mask_dict
                )
                
            if len(self.obs.keys()) > 0:
                filename = os.path.join(path, "observables.pdf")
                plotter.plot_observables(
                    filename=filename, **kwargs, weights=weights, mask_dict=mask_dict
                )
        LOGGER.info(f"Plotting done in {time.time() - t0:.2f} seconds")

    def define_process_specifics(self):
        """Define process specifics for jet tokens"""
        super().define_process_specifics()
        self.plot_title = "Jet Tokens"