import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

import os, time, glob
from omegaconf import open_dict
from itertools import chain

from experiments.base_experiment import BaseExperiment
from experiments.dataset import (
    Dataset,
    load_dataset,
    positional_encoding,
    load_ttbar_file,
)
import experiments.kinematics.plotter as plotter
from experiments.kinematics.plots import plot_kinematics
from experiments.logger import LOGGER
from experiments.kinematics.observables import create_partial_jet
from experiments.coordinates import fourmomenta_to_jetmomenta, jetmomenta_to_fourmomenta
from experiments.utils import GaussianFourierProjection


class JetKinematicsExperiment(BaseExperiment):
    def init_physics(self):

        with open_dict(self.cfg):
            self.cfg.modelname = self.cfg.model._target_.rsplit(".", 1)[-1][:-3]
            self.cfg.cfm.run_dir = self.cfg.run_dir

            if self.cfg.evaluation.load_samples:
                self.cfg.train = False
                self.cfg.evaluation.sample = False
                self.cfg.evaluation.save_samples = False

            max_num_particles, diff, pt_min, jet_pt_min, masked_dims, load_fn = (
                load_dataset(self.cfg.data.dataset)
            )

            self.cfg.cfm.jet_coordinates_options.pt_min = jet_pt_min
            self.cfg.cfm.const_coordinates_options.pt_min = pt_min
            self.cfg.cfm.const_coordinates_options.fixed_dims = masked_dims
            self.cfg.cfm.masked_dims = []
            self.load_fn = load_fn

            self.cfg.cfm.mult_encoding_dim = self.cfg.data.mult_encoding_dim
            self.cfg.cfm.pe_dim = self.cfg.data.pos_encoding_dim

            if self.cfg.modelname == "JetConditionalTransformer":
                self.cfg.cfm.transpose = True
                if self.cfg.cfm.transpose:
                    base_in_channels = 1
                else:
                    base_in_channels = 4
                self.cfg.model.net.in_channels = (
                    base_in_channels
                    + self.cfg.cfm.embed_t_dim
                    + self.cfg.data.mult_encoding_dim
                )
                if self.cfg.cfm.transpose:
                    self.cfg.model.net.in_channels += self.cfg.data.pos_encoding_dim
                if self.cfg.cfm.add_constituents:
                    self.cfg.model.net_condition.in_channels = (
                        base_in_channels + self.cfg.data.pos_encoding_dim + 1
                    )
                else:
                    self.cfg.model.net_condition.in_channels = (
                        base_in_channels + self.cfg.data.mult_encoding_dim
                    )
                    if self.cfg.cfm.transpose:
                        self.cfg.model.net_condition.in_channels += (
                            self.cfg.data.pos_encoding_dim
                        )
                self.cfg.model.net_condition.out_channels = (
                    self.cfg.model.net.hidden_channels
                )
                self.cfg.model.net.out_channels = base_in_channels

                if self.cfg.cfm.self_condition_prob > 0.0:
                    self.cfg.model.net.in_channels += base_in_channels

            elif self.cfg.modelname == "JetConditionalLGATr":
                self.cfg.cfm.transpose = False
                self.cfg.model.net.in_s_channels = (
                    self.cfg.cfm.embed_t_dim + self.cfg.data.mult_encoding_dim
                )
                if self.cfg.cfm.add_constituents:
                    self.cfg.model.net_condition.in_s_channels = (
                        self.cfg.data.pos_encoding_dim + 1
                    )
                else:
                    self.cfg.model.net_condition.in_s_channels = (
                        self.cfg.data.mult_encoding_dim
                    )
                self.cfg.model.net_condition.out_mv_channels = (
                    self.cfg.model.net.hidden_mv_channels
                )
                self.cfg.model.net.condition_mv_channels = (
                    self.cfg.model.net_condition.out_mv_channels
                )
                self.cfg.model.net_condition.out_s_channels = (
                    self.cfg.model.net.hidden_s_channels
                )
                self.cfg.model.net.condition_s_channels = (
                    self.cfg.model.net_condition.out_s_channels
                )
                if self.cfg.cfm.self_condition_prob > 0.0:
                    self.cfg.model.net.in_s_channels += 4

            elif self.cfg.modelname == "JetMLP":
                base_in_channels = 4
                self.cfg.model.net.in_shape = (
                    base_in_channels
                    + self.cfg.cfm.embed_t_dim
                    + self.cfg.data.mult_encoding_dim
                )
                self.cfg.model.net.out_shape = base_in_channels

                if not self.cfg.cfm.unconditional:
                    self.cfg.model.net.in_shape += (
                        base_in_channels + self.cfg.data.mult_encoding_dim
                    )
                if self.cfg.cfm.self_condition_prob > 0.0:
                    self.cfg.model.net.in_channels += base_in_channels

            # copy model-specific parameters
            self.cfg.model.odeint = self.cfg.odeint
            self.cfg.model.cfm = self.cfg.cfm

        self.define_process_specifics()

    def init_data(self):
        if self.cfg.evaluation.load_samples:
            # if we load samples, we do not need to initialize the data
            self.train_data = None
            self.val_data = None
            self.test_data = None
            return
        t0 = time.time()
        data_path = os.path.join(self.cfg.data.data_dir, f"{self.cfg.data.dataset}")
        LOGGER.info(f"Creating {self.cfg.data.dataset} from {data_path}")
        if self.cfg.data.dataset == "ttbar":
            self._init_data2(data_path)
        else:
            self._init_data(data_path)
        LOGGER.info(
            f"Created {self.cfg.data.dataset} with {len(self.train_data)} training jets, {len(self.val_data)} validation jets, and {len(self.test_data)} test jets in {time.time() - t0:.2f} seconds"
        )

    def _init_data(self, data_path):
        t0 = time.time()
        data = self.load_fn(data_path, self.cfg.data, self.dtype)
        det_particles = data["det_particles"]
        det_mults = data["det_mults"]
        det_pids = data["det_pids"]
        det_jets = data["det_jets"]
        gen_particles = data["gen_particles"]
        gen_mults = data["gen_mults"]
        gen_pids = data["gen_pids"]
        gen_jets = data["gen_jets"]
        size = len(gen_jets)

        LOGGER.info(f"Loaded {size} events in {time.time() - t0:.2f} seconds")

        det_jets = jetmomenta_to_fourmomenta(det_jets)
        gen_jets = jetmomenta_to_fourmomenta(gen_jets)

        if self.cfg.data.max_constituents > 0:
            det_mults = torch.clamp(det_mults, max=self.cfg.data.max_constituents)
            gen_mults = torch.clamp(gen_mults, max=self.cfg.data.max_constituents)

        split = self.cfg.data.train_val_test
        train_idx, val_idx, test_idx = np.cumsum([int(s * size) for s in split])

        # initialize cfm (requires data)
        self.model.init_coordinates()

        # initialize geometry
        self.model.init_geometry()

        # For jet-level learning, we fit on the jet momenta directly
        # We create a simple mask where each jet is treated as a single particle
        train_gen_mask = torch.ones(train_idx, 1, dtype=torch.bool)
        self.model.coordinates.init_fit(
            gen_jets[:train_idx].unsqueeze(1),  # Add sequence dimension
            mask=train_gen_mask,
            jet=gen_jets[:train_idx],
        )

        jet_train_det_mask = torch.ones(train_idx, 1, dtype=torch.bool)
        self.model.condition_coordinates.init_fit(
            det_jets[:train_idx].unsqueeze(1),
            mask=jet_train_det_mask,
            jet=det_jets[:train_idx],
        )

        det_jets = self.model.condition_coordinates.fourmomenta_to_x(det_jets)

        gen_jets = self.model.coordinates.fourmomenta_to_x(gen_jets)

        if self.cfg.cfm.add_constituents:
            train_det_mask = (
                torch.arange(det_particles.shape[1])[None, :]
                < det_mults[:train_idx, None]
            )
            self.model.constituents_condition_coordinates.init_fit(
                det_particles[:train_idx],
                mask=train_det_mask,
                jet=torch.repeat_interleave(
                    det_jets[:train_idx], det_mults[:train_idx], dim=0
                ),
            )

            det_mask = (
                torch.arange(det_particles.shape[1])[None, :] < det_mults[:, None]
            )
            det_particles[det_mask] = (
                self.model.constituents_condition_coordinates.fourmomenta_to_x(
                    det_particles[det_mask],
                    jet=torch.repeat_interleave(det_jets, det_mults, dim=0),
                    ptr=torch.cumsum(
                        torch.cat(
                            [torch.zeros(1, dtype=torch.int64), det_mults], dim=0
                        ),
                        dim=0,
                    ),
                )
            )

        pos_encoding = positional_encoding(pe_dim=self.cfg.data.pos_encoding_dim)

        if self.cfg.data.mult_encoding_dim > 0:
            mult_encoding = self.model.mult_encoding.to(pos_encoding.device)
        else:
            mult_encoding = None

        self.train_data = Dataset(
            self.dtype,
            pos_encoding=pos_encoding,
            mult_encoding=mult_encoding,
        )
        self.val_data = Dataset(
            self.dtype,
            pos_encoding=pos_encoding,
            mult_encoding=mult_encoding,
        )
        self.test_data = Dataset(
            self.dtype,
            pos_encoding=pos_encoding,
            mult_encoding=mult_encoding,
        )

        self.train_data.append(
            det_particles=det_particles[:train_idx],
            det_pids=det_pids[:train_idx],
            det_mults=det_mults[:train_idx],
            det_jets=det_jets[:train_idx],
            gen_particles=gen_particles[:train_idx],
            gen_pids=gen_pids[:train_idx],
            gen_mults=gen_mults[:train_idx],
            gen_jets=gen_jets[:train_idx],
        )
        self.val_data.append(
            det_particles=det_particles[train_idx:val_idx],
            det_pids=det_pids[train_idx:val_idx],
            det_mults=det_mults[train_idx:val_idx],
            det_jets=det_jets[train_idx:val_idx],
            gen_particles=gen_particles[train_idx:val_idx],
            gen_pids=gen_pids[train_idx:val_idx],
            gen_mults=gen_mults[train_idx:val_idx],
            gen_jets=gen_jets[train_idx:val_idx],
        )
        self.test_data.append(
            det_particles=det_particles[val_idx:test_idx],
            det_pids=det_pids[val_idx:test_idx],
            det_mults=det_mults[val_idx:test_idx],
            det_jets=det_jets[val_idx:test_idx],
            gen_particles=gen_particles[val_idx:test_idx],
            gen_pids=gen_pids[val_idx:test_idx],
            gen_mults=gen_mults[val_idx:test_idx],
            gen_jets=gen_jets[val_idx:test_idx],
        )

    def _init_data2(self, data_path):
        t0 = time.time()

        pos_encoding = positional_encoding(pe_dim=self.cfg.data.pos_encoding_dim)

        if self.cfg.data.mult_encoding_dim > 0:
            mult_encoding = self.model.mult_encoding.to(pos_encoding.device)
        else:
            mult_encoding = None

        self.train_data = Dataset(
            self.dtype,
            pos_encoding=pos_encoding,
            mult_encoding=mult_encoding,
        )
        self.val_data = Dataset(
            self.dtype,
            pos_encoding=pos_encoding,
            mult_encoding=mult_encoding,
        )
        self.test_data = Dataset(
            self.dtype,
            pos_encoding=pos_encoding,
            mult_encoding=mult_encoding,
        )

        # initialize cfm (requires data)
        self.model.init_coordinates()

        # initialize geometry
        self.model.init_geometry()

        files = sorted(glob.glob(os.path.join(data_path, "new_ttbar*.parquet")))
        num_events = self.cfg.data.length
        for i in range(len(files)):
            data = self.process_one_file(files[i], init=(i == 0), num_events=num_events)

            t0 = time.time()

            size = data["det_particles"].shape[0]
            split = self.cfg.data.train_val_test
            train_idx, val_idx, test_idx = np.cumsum([int(s * size) for s in split])

            self.train_data.append(
                det_particles=data["det_particles"][:train_idx],
                det_pids=data["det_pids"][:train_idx],
                det_mults=data["det_mults"][:train_idx],
                det_jets=data["det_jets"][:train_idx],
                gen_particles=data["gen_particles"][:train_idx],
                gen_pids=data["gen_pids"][:train_idx],
                gen_mults=data["gen_mults"][:train_idx],
                gen_jets=data["gen_jets"][:train_idx],
            )
            self.val_data.append(
                det_particles=data["det_particles"][train_idx:val_idx],
                det_pids=data["det_pids"][train_idx:val_idx],
                det_mults=data["det_mults"][train_idx:val_idx],
                det_jets=data["det_jets"][train_idx:val_idx],
                gen_particles=data["gen_particles"][train_idx:val_idx],
                gen_pids=data["gen_pids"][train_idx:val_idx],
                gen_mults=data["gen_mults"][train_idx:val_idx],
                gen_jets=data["gen_jets"][train_idx:val_idx],
            )
            self.test_data.append(
                det_particles=data["det_particles"][val_idx:test_idx],
                det_pids=data["det_pids"][val_idx:test_idx],
                det_mults=data["det_mults"][val_idx:test_idx],
                det_jets=data["det_jets"][val_idx:test_idx],
                gen_particles=data["gen_particles"][val_idx:test_idx],
                gen_pids=data["gen_pids"][val_idx:test_idx],
                gen_mults=data["gen_mults"][val_idx:test_idx],
                gen_jets=data["gen_jets"][val_idx:test_idx],
            )

            if num_events > 0:
                num_events -= data["det_particles"].shape[0]
                if num_events <= 0:
                    break
            LOGGER.info(
                f"Created {train_idx} training graphs, {val_idx - train_idx} validation graphs, {test_idx - val_idx} test graphs in {time.time() - t0:.2f} seconds"
            )

    def process_one_file(self, file, num_events, init=False):
        t0 = time.time()
        data = load_ttbar_file(file, self.cfg.data, self.dtype, num_events)
        det_particles = data["det_particles"]
        det_mults = data["det_mults"]
        det_pids = data["det_pids"]
        det_jets = data["det_jets"]
        gen_particles = data["gen_particles"]
        gen_mults = data["gen_mults"]
        gen_pids = data["gen_pids"]
        gen_jets = data["gen_jets"]
        size = len(gen_particles)

        LOGGER.info(
            f"Loaded {size} events from {file} in {time.time() - t0:.2f} seconds"
        )
        t1 = time.time()

        if self.cfg.data.max_constituents > 0:
            det_mults = torch.clamp(det_mults, max=self.cfg.data.max_constituents)
            gen_mults = torch.clamp(gen_mults, max=self.cfg.data.max_constituents)

        size = len(gen_particles)

        det_jets = jetmomenta_to_fourmomenta(det_jets)
        gen_jets = jetmomenta_to_fourmomenta(gen_jets)

        split = self.cfg.data.train_val_test
        train_idx, val_idx, test_idx = np.cumsum([int(s * size) for s in split])

        if init:
            # For jet-level learning, we fit on the jet momenta directly
            # We create a simple mask where each jet is treated as a single particle
            train_gen_mask = torch.ones(train_idx, 1, dtype=torch.bool)
            self.model.coordinates.init_fit(
                gen_jets[:train_idx].unsqueeze(1),  # Add sequence dimension
                mask=train_gen_mask,
                jet=gen_jets[:train_idx],
            )

            jet_train_det_mask = torch.ones(train_idx, 1, dtype=torch.bool)
            self.model.condition_coordinates.init_fit(
                det_jets[:train_idx].unsqueeze(1),
                mask=jet_train_det_mask,
                jet=det_jets[:train_idx],
            )

        det_jets = self.model.condition_coordinates.fourmomenta_to_x(det_jets)

        gen_jets = self.model.coordinates.fourmomenta_to_x(gen_jets)

        LOGGER.info(
            f"Preprocessed {size} events from {file} in {time.time() - t1:.2f} seconds"
        )
        return {
            "det_particles": det_particles,
            "det_mults": det_mults,
            "det_pids": det_pids,
            "det_jets": det_jets,
            "gen_particles": gen_particles,
            "gen_mults": gen_mults,
            "gen_pids": gen_pids,
            "gen_jets": gen_jets,
        }

    def _init_dataloader(self):

        if self.cfg.evaluation.load_samples:
            self.train_loader = None
            self.val_loader = None
            self.test_loader = None
            return
        train_sampler = torch.utils.data.DistributedSampler(
            self.train_data,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )
        self.train_loader = DataLoader(
            dataset=self.train_data,
            batch_size=self.cfg.training.batchsize // self.world_size,
            sampler=train_sampler,
            follow_batch=["x_gen", "x_det"],
        )
        test_sampler = torch.utils.data.DistributedSampler(
            self.test_data,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )
        self.test_loader = DataLoader(
            dataset=self.test_data,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            sampler=test_sampler,
            follow_batch=["x_gen", "x_det"],
        )
        val_sampler = torch.utils.data.DistributedSampler(
            self.val_data,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )
        self.val_loader = DataLoader(
            dataset=self.val_data,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            sampler=val_sampler,
            follow_batch=["x_gen", "x_det"],
        )

        LOGGER.info(
            f"Constructed dataloaders with "
            f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )

    @torch.no_grad()
    def evaluate(self, sampled_mults=None):
        # EMA-evaluation not implemented
        loaders = {
            "train": self.train_loader,
            "test": self.test_loader,
            "val": self.val_loader,
        }
        self.model.eval()
        if self.cfg.evaluation.sample:
            t0 = time.time()
            self._sample_events(loaders["test"], sampled_mults)
            loaders["gen"] = self.sample_loader
            dt = time.time() - t0
            LOGGER.info(f"Finished sampling after {dt/60:.2f}min")
        elif self.cfg.evaluation.load_samples:
            self._load_samples()
            loaders["gen"] = self.sample_loader
        else:
            LOGGER.info("Skip sampling")

    def _sample_events(self, loader, sampled_mults=None):
        samples = []
        targets = []
        self.data_raw = {}
        n_batches = self.cfg.evaluation.n_batches
        if n_batches > len(loader):
            LOGGER.warning(
                f"Requested {n_batches} batches for sampling, but only {len(loader)} batches available in test dataset."
            )
            n_batches = len(loader)
        elif n_batches == -1:
            n_batches = len(loader)
        LOGGER.info(f"Sampling {n_batches} batches for evaluation")
        it = iter(loader)

        for i in range(n_batches):
            batch = next(it).to(self.device)
            new_batch = batch.clone()

            if sampled_mults is not None:
                new_batch.jet_scalars_gen = self.model.mult_encoding(
                    sampled_mults[
                        self.cfg.evaluation.batchsize
                        * i : self.cfg.evaluation.batchsize
                        * (i + 1)
                    ].to(dtype=self.dtype)
                ).to(self.device)

            sample_batch, base = self.model.sample(
                new_batch,
                self.device,
                self.dtype,
            )

            sample_batch.jet_det = self.model.condition_coordinates.x_to_fourmomenta(
                sample_batch.jet_det
            )

            sample_batch.jet_gen = self.model.coordinates.x_to_fourmomenta(
                sample_batch.jet_gen
            )

            batch.jet_det = self.model.condition_coordinates.x_to_fourmomenta(
                batch.jet_det
            )

            batch.jet_gen = self.model.coordinates.x_to_fourmomenta(batch.jet_gen)

            if self.cfg.cfm.add_constituents:
                sample_batch.x_det = (
                    self.model.constituents_condition_coordinates.x_to_fourmomenta(
                        sample_batch.x_det
                    )
                )
                batch.x_det = (
                    self.model.constituents_condition_coordinates.x_to_fourmomenta(
                        batch.x_det
                    )
                )

            samples.extend(sample_batch.detach().to_data_list())
            targets.extend(batch.detach().to_data_list())

            LOGGER.info(f"Sampled batch {i+1}/{n_batches}")

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
            torch.save(self.data_raw["samples"], os.path.join(path, "samples.pt"))
            torch.save(self.data_raw["truth"], os.path.join(path, "truth.pt"))

    def _load_samples(self):
        path = os.path.join(self.cfg.run_dir, f"samples_{self.cfg.warm_start_idx}")
        LOGGER.info(f"Loading samples from {path}")
        t0 = time.time()
        self.data_raw = {}
        self.data_raw["samples"] = torch.load(
            os.path.join(path, "samples.pt"),
            weights_only=False,
            map_location=self.device,
        )
        self.data_raw["truth"] = torch.load(
            os.path.join(path, "truth.pt"), weights_only=False, map_location=self.device
        )
        LOGGER.info(f"Loaded samples with {len(self.data_raw['samples'])} jets")

        samples = self.data_raw["samples"].to_data_list()
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

        LOGGER.info(f"Loaded samples in {time.time() - t0:.2f}s")

    def plot(self):
        path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(os.path.join(path), exist_ok=True)
        LOGGER.info(f"Creating plots in {path}")
        t0 = time.time()

        if self.cfg.modelname == "JetConditionalTransformer":
            model_label = "Transformer"
        elif self.cfg.modelname == "JetConditionalLGATr":
            model_label = "L-GATr"
        elif self.cfg.modelname == "JetMLP":
            model_label = "MLP"
        else:
            model_label = self.cfg.modelname
        kwargs = {
            "exp": self,
            "model_label": model_label,
        }

        if self.cfg.train:
            filename = os.path.join(path, "training.pdf")
            plotter.plot_losses(filename=filename, **kwargs)

        if not self.cfg.evaluate:
            return

        weights, mask_dict = None, None

        if self.cfg.evaluation.sample or self.cfg.evaluation.load_samples:
            if self.cfg.plotting.fourmomenta:
                filename = os.path.join(path, "fourmomenta.pdf")
                plotter.plot_fourmomenta(
                    filename=filename,
                    **kwargs,
                    jet=True,
                    weights=weights,
                    mask_dict=mask_dict,
                )

            if self.cfg.plotting.jetmomenta:
                filename = os.path.join(path, "jetmomenta.pdf")
                plotter.plot_jetmomenta(
                    filename=filename,
                    **kwargs,
                    jet=True,
                    weights=weights,
                    mask_dict=mask_dict,
                )
        LOGGER.info(f"Plotting done in {time.time() - t0:.2f} seconds")

    def _init_loss(self):
        # loss defined manually within the model
        pass

    def _batch_loss(self, batch):
        batch = batch.to(self.device)
        loss, component_loss = self.model.batch_loss(batch)
        mse = loss.cpu().item()
        assert torch.isfinite(loss).all()
        metrics = {"mse": mse}
        for k in range(4):
            metrics[f"mse_{k}"] = component_loss[k].cpu().item()
        return loss, metrics

    def _init_metrics(self):
        metrics = {"mse": []}
        for k in range(4):
            metrics[f"mse_{k}"] = []
        return metrics

    def define_process_specifics(self):
        self.plot_title = None

        self.obs_coords = {}
        self.obs_coords[r"\text{jet}"] = create_partial_jet(0.0, 1.0)

        self.obs = {}
