import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch.distributions import Categorical
import numpy as np

import os, time
from omegaconf import open_dict

from experiments.base_experiment import BaseExperiment
from experiments.dataset import Dataset, load_dataset, positional_encoding
from experiments.multiplicity.distributions import (
    GammaMixture,
    GaussianMixture,
    cross_entropy,
)
from experiments.multiplicity.plots import plot_mixer
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow
from experiments.utils import GaussianFourierProjection

MODEL_TITLE_DICT = {"LGATr": "L-GATr", "Transformer": "Tr"}


class MultiplicityExperiment(BaseExperiment):
    def _init_loss(self):
        self.loss = lambda dist, target: cross_entropy(dist, target).mean()

    def init_physics(self):

        with open_dict(self.cfg):
            self.cfg.modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]

            if self.cfg.evaluation.load_samples:
                self.cfg.train = False
                self.cfg.evaluation.sample = False
                self.cfg.evaluation.save_samples = False

            max_num_particles, diff, pt_min, jet_pt_min, masked_dims, load_fn = (
                load_dataset(self.cfg.data.dataset)
            )

            self.cfg.data.max_num_particles = max_num_particles
            self.cfg.data.diff = diff
            self.cfg.data.pt_min = pt_min
            self.load_fn = load_fn

            self.cfg.model.distribution = self.cfg.dist.type
            self.cfg.model.range = (
                self.cfg.data.min_mult,
                self.cfg.data.max_num_particles,
            )

            if self.cfg.modelname == "Transformer":
                self.cfg.model.net.in_channels = 4
                if self.cfg.data.add_pid:
                    self.cfg.model.net.in_channels += 6
                if self.cfg.data.pos_encoding_dim > 0:
                    self.cfg.model.net.in_channels += self.cfg.data.pos_encoding_dim
                if self.cfg.dist.type == "GammaMixture":
                    self.distribution = GammaMixture
                    self.cfg.model.net.out_channels = 3 * self.cfg.dist.n_components
                    assert (
                        self.cfg.dist.diff == False
                    ), "GammaMixture requires non-negative integers"
                elif self.cfg.dist.type == "GaussianMixture":
                    self.distribution = GaussianMixture
                    self.cfg.model.net.out_channels = 3 * self.cfg.dist.n_components
                elif self.cfg.dist.type == "Categorical":
                    self.distribution = Categorical
                    if self.cfg.dist.diff:
                        self.cfg.model.net.out_channels = (
                            self.cfg.data.diff[1] - self.cfg.data.diff[0] + 1
                        )
                        self.cfg.model.range = self.cfg.data.diff
                    else:
                        self.cfg.model.net.out_channels = (
                            self.cfg.data.max_num_particles - self.cfg.data.min_mult + 1
                        )
                        self.cfg.model.range = (
                            self.cfg.data.min_mult,
                            self.cfg.data.max_num_particles,
                        )
            elif self.cfg.modelname == "LGATr":
                if self.cfg.dist.type == "GammaMixture":
                    self.distribution = GammaMixture
                    assert (
                        self.cfg.dist.diff == False
                    ), "GammaMixture requires non-negative integers"
                    self.cfg.model.net.out_mv_channels = 3 * self.cfg.dist.n_components
                elif self.cfg.dist.type == "GaussianMixture":
                    self.distribution = GaussianMixture
                    self.cfg.model.net.out_mv_channels = 3 * self.cfg.dist.n_components
                elif self.cfg.dist.type == "Categorical":
                    self.distribution = Categorical
                    if self.cfg.dist.diff:
                        self.cfg.model.net.out_mv_channels = (
                            self.cfg.data.diff[1] - self.cfg.data.diff[0] + 1
                        )
                        self.cfg.model.range = self.cfg.data.diff
                    else:
                        self.cfg.model.net.out_mv_channels = (
                            self.cfg.data.max_num_particles + 1
                        )
                        self.cfg.model.range = (
                            self.cfg.data.min_mult,
                            self.cfg.data.max_num_particles,
                        )

                # scalar channels
                self.cfg.model.net.in_s_channels = 0
                if self.cfg.data.add_pid:
                    self.cfg.model.net.in_s_channels += 6
                if self.cfg.data.pos_encoding_dim > 0:
                    self.cfg.model.net.in_s_channels += self.cfg.data.pos_encoding_dim

                # mv channels for beam_reference and time_reference
                self.cfg.model.net.in_mv_channels = 1

            else:
                raise ValueError(f"Model not implemented: {self.cfg.modelname}")

    def init_data(self):
        data_path = os.path.join(self.cfg.data.data_dir, f"{self.cfg.data.dataset}")
        LOGGER.info(f"Creating MultiplicityDataset from {data_path}")
        t0 = time.time()
        self._init_data(data_path)
        LOGGER.info(f"Created MultiplicityDataset in {time.time() - t0:.2f} seconds")

    def _init_data(self, data_path):
        if self.cfg.evaluation.load_samples:
            # if we load samples, we do not need to initialize the data
            self.train_data = None
            self.val_data = None
            self.test_data = None
            return

        data = self.load_fn(data_path, self.cfg.data, self.dtype)

        det_particles = data["det_particles"]
        det_pids = data["det_pids"]
        det_mults = data["det_mults"]
        gen_particles = data["gen_particles"]
        gen_pids = data["gen_pids"]
        gen_mults = data["gen_mults"]

        det_jets = det_particles.sum(dim=1, keepdim=True)
        gen_jets = gen_particles.sum(dim=1, keepdim=True)

        size = len(det_particles)
        split = self.cfg.data.train_val_test
        train_idx, val_idx, test_idx = np.cumsum([int(s * size) for s in split])

        if self.cfg.data.pos_encoding_dim > 0:
            pos_encoding = positional_encoding(pe_dim=self.cfg.data.pos_encoding_dim)
        else:
            pos_encoding = None

        self.train_data = Dataset(
            self.dtype,
            pos_encoding=pos_encoding,
        )
        self.val_data = Dataset(
            self.dtype,
            pos_encoding=pos_encoding,
        )
        self.test_data = Dataset(
            self.dtype,
            pos_encoding=pos_encoding,
        )

        self.train_data.create_data_list(
            det_particles=det_particles[:train_idx],
            det_pids=det_pids[:train_idx],
            det_mults=det_mults[:train_idx],
            det_jets=det_jets[:train_idx],
            gen_particles=gen_particles[:train_idx],
            gen_pids=gen_pids[:train_idx],
            gen_mults=gen_mults[:train_idx],
            gen_jets=gen_jets[:train_idx],
        )
        self.val_data.create_data_list(
            det_particles=det_particles[train_idx:val_idx],
            det_pids=det_pids[train_idx:val_idx],
            det_mults=det_mults[train_idx:val_idx],
            det_jets=det_jets[train_idx:val_idx],
            gen_particles=gen_particles[train_idx:val_idx],
            gen_pids=gen_pids[train_idx:val_idx],
            gen_mults=gen_mults[train_idx:val_idx],
            gen_jets=gen_jets[train_idx:val_idx],
        )
        self.test_data.create_data_list(
            det_particles=det_particles[val_idx:test_idx],
            det_pids=det_pids[val_idx:test_idx],
            det_mults=det_mults[val_idx:test_idx],
            det_jets=det_jets[val_idx:test_idx],
            gen_particles=gen_particles[val_idx:test_idx],
            gen_pids=gen_pids[val_idx:test_idx],
            gen_mults=gen_mults[val_idx:test_idx],
            gen_jets=gen_jets[val_idx:test_idx],
        )

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
            num_workers=2,
            pin_memory=True,
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
            num_workers=2,
            pin_memory=True,
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
            num_workers=2,
            pin_memory=True,
        )

        LOGGER.info(
            f"Constructed dataloaders with "
            f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )

    @torch.no_grad()
    def evaluate(self):
        if self.cfg.evaluation.load_samples:
            self.results_test = self._load_samples()
        else:
            if self.ema is not None:
                with self.ema.average_parameters():
                    # self.results_train = self._evaluate_single(
                    #     self.train_loader, "train"
                    # )
                    # self.results_val = self._evaluate_single(self.val_loader, "val")
                    self.results_test = self._evaluate_single(self.test_loader, "test")

                # also evaluate without ema to see the effect
                # self._evaluate_single(self.train_loader, "train_noema")
                # self._evaluate_single(self.val_loader, "val_noema")
                self._evaluate_single(self.test_loader, "test_noema")

            else:
                # self.results_train = self._evaluate_single(self.train_loader, "train")
                # self.results_val = self._evaluate_single(self.val_loader, "val")
                self.results_test = self._evaluate_single(self.test_loader, "test")
            if self.cfg.evaluation.save_samples:
                tensor_path = os.path.join(
                    self.cfg.run_dir, f"samples_{self.cfg.run_idx}"
                )
                os.makedirs(tensor_path, exist_ok=True)
                torch.save(
                    self.results_test["samples"],
                    f"{tensor_path}/samples.pt",
                )
            if self.cfg.evaluation.save_params > 0:
                torch.save(
                    self.results_test["params"][: self.cfg.evaluation.save_params],
                    f"{tensor_path}/params.pt",
                )

    def _evaluate_single(self, loader, title, step=None):
        LOGGER.info(
            f"### Starting to evaluate model on {title} dataset with "
            f"{len(loader.dataset)} elements, batchsize {loader.batch_size} ###"
        )
        outputs = {}
        self.model.eval()
        nll = []
        params = []
        samples = []
        with torch.no_grad():
            for batch in loader:
                batch_samples, batch_params, batch_nll = self.sample(batch)
                nll.append(batch_nll)
                params.append(batch_params)
                samples.append(batch_samples)
        nll = torch.tensor(nll)
        LOGGER.info(f"NLL on {title} dataset: {nll.mean():.4f}")

        outputs["loss"] = nll.mean()
        outputs["params"] = torch.cat(params)
        outputs["samples"] = torch.cat(samples)
        if self.cfg.use_mlflow:
            for key, value in outputs.items():
                name = f"{title}"
                log_mlflow(f"{name}.{key}", value, step=step)
        return outputs

    def _load_samples(self):
        path = os.path.join(self.cfg.run_dir, f"samples_{self.cfg.warm_start_idx}")
        LOGGER.info(f"Loading samples from {path}")
        saved_samples = {}
        saved_samples["samples"] = torch.load(f"{path}/samples.pt")
        if os.path.isfile(f"{path}/params.pt"):
            saved_samples["params"] = torch.load(f"{path}/params.pt")
        return saved_samples

    def plot(self):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path, exist_ok=True)
        LOGGER.info(f"Creating plots in {plot_path}")

        plot_dict = {}
        if self.cfg.evaluate:
            # plot_dict["results_train"] = self.results_train
            # plot_dict["results_val"] = self.results_val
            plot_dict["results_test"] = self.results_test
        if self.cfg.train:
            plot_dict["train_loss"] = self.train_loss
            plot_dict["val_loss"] = self.val_loss
            plot_dict["train_lr"] = self.train_lr
        plot_mixer(self.cfg, plot_path, plot_dict)

    def _batch_loss(self, batch):
        batch = batch.to(self.device, non_blocking=True)

        loss = self.model.batch_loss(batch, diff=self.cfg.dist.diff)
        return loss, {"nll": loss.detach()}

    def sample(self, batch):
        batch = batch.to(self.device)

        return self.model.sample(
            batch,
            range=(self.cfg.data.min_mult, self.cfg.data.max_num_particles),
            diff=self.cfg.dist.diff,
        )

    def _init_metrics(self):
        return {"nll": []}
