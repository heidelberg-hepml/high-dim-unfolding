import torch
from torch_geometric.loader import DataLoader
import numpy as np

import os, time
from omegaconf import open_dict

from experiments.base_experiment import BaseExperiment
from experiments.dataset import (
    Dataset,
    load_zplusjet,
    load_ttbar,
)
from experiments.distributions import (
    GammaMixture,
    CategoricalDistribution,
    GaussianMixture,
    cross_entropy,
    smooth_cross_entropy,
)
from experiments.multiplicity.plots import plot_mixer
from experiments.multiplicity.embedding import (
    embed_data_into_ga,
    compute_scalar_features_from_jetmomenta,
)
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow
from gatr.interface import get_num_spurions

MODEL_TITLE_DICT = {"GATr": "GATr", "Transformer": "Tr"}


class MultiplicityExperiment(BaseExperiment):
    def _init_loss(self):
        if self.cfg.loss.type == "cross_entropy":
            self.loss = lambda dist, target: cross_entropy(dist, target).mean()
        elif self.cfg.loss.type == "smooth_cross_entropy":
            self.loss = lambda dist, target: smooth_cross_entropy(
                dist, target, self.cfg.data.max_num_particles, self.cfg.loss.smoothness
            ).mean()

    def init_physics(self):

        with open_dict(self.cfg):
            self.cfg.modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]

            if self.cfg.data.dataset == "zplusjet":
                self.cfg.data.max_num_particles = 152
                self.cfg.data.diff = [-53, 78]
            elif self.cfg.data.dataset == "ttbar":
                self.cfg.data.max_num_particles = 238
                self.cfg.data.diff = [-35, 101]

            if self.cfg.modelname == "Transformer":
                self.cfg.model.net.in_channels = 4
                if self.cfg.data.add_pid:
                    self.cfg.model.net.in_channels += 6
                if self.cfg.dist.type == "GammaMixture":
                    self.distribution = GammaMixture
                    self.cfg.model.net.out_channels = 3 * self.cfg.dist.n_components
                elif self.cfg.dist.type == "GaussianMixture":
                    self.distribution = GaussianMixture
                    self.cfg.model.net.out_channels = 3 * self.cfg.dist.n_components
                elif self.cfg.dist.type == "Categorical":
                    self.distribution = CategoricalDistribution
                    if self.cfg.dist.diff:
                        self.cfg.model.net.out_channels = (
                            self.cfg.data.diff[1] - self.cfg.data.diff[0] + 1
                        )
                    else:
                        self.cfg.model.net.out_channels = (
                            self.cfg.data.max_num_particles + 1
                        )
            elif self.cfg.modelname == "GATr":
                if self.cfg.dist.type == "GammaMixture":
                    self.distribution = GammaMixture
                    self.cfg.model.net.out_mv_channels = 3 * self.cfg.dist.n_components
                elif self.cfg.dist.type == "GaussianMixture":
                    self.distribution = GaussianMixture
                    self.cfg.model.net.out_mv_channels = 3 * self.cfg.dist.n_components
                elif self.cfg.dist.type == "Categorical":
                    self.distribution = CategoricalDistribution
                    if self.cfg.dist.diff:
                        self.cfg.model.net.out_mv_channels = (
                            self.cfg.data.diff[1] - self.cfg.data.diff[0] + 1
                        )
                    else:
                        self.cfg.model.net.out_mv_channels = (
                            self.cfg.data.max_num_particles + 1
                        )

                # scalar channels
                self.cfg.model.net.in_s_channels = 0
                if self.cfg.data.add_pid:
                    self.cfg.model.net.in_s_channels += 6

                # mv channels for beam_reference and time_reference
                self.cfg.model.net.in_mv_channels = 1
                if not self.cfg.data.beam_token:
                    self.cfg.model.net.in_mv_channels += get_num_spurions(
                        self.cfg.data.beam_reference,
                        self.cfg.data.add_time_reference,
                        self.cfg.data.two_beams,
                        self.cfg.data.add_xzplane,
                        self.cfg.data.add_yzplane,
                    )

                # reinsert channels
                if self.cfg.data.reinsert_channels:
                    self.cfg.model.net.reinsert_mv_channels = list(
                        range(self.cfg.model.net.in_mv_channels)
                    )
                    self.cfg.model.net.reinsert_s_channels = list(
                        range(self.cfg.model.net.in_s_channels)
                    )

            else:
                raise ValueError(f"Model not implemented: {self.cfg.modelname}")

    def init_data(self):
        data_path = os.path.join(self.cfg.data.data_dir, f"{self.cfg.data.dataset}")
        LOGGER.info(f"Creating MultiplicityDataset from {data_path}")
        t0 = time.time()
        self._init_data(data_path)
        LOGGER.info(f"Created MultiplicityDataset in {time.time() - t0:.2f} seconds")

    def _init_data(self, data_path):

        if self.cfg.data.dataset == "zplusjet":
            data = load_zplusjet(data_path, self.cfg.data, self.dtype)
        elif self.cfg.data.dataset == "ttbar":
            data = load_ttbar(data_path, self.cfg.data, self.dtype)
        else:
            raise ValueError(f"Dataset not implemented: {self.cfg.data.dataset}")

        det_particles = data["det_particles"]
        det_pids = data["det_pids"]
        det_mults = data["det_mults"]
        gen_particles = data["gen_particles"]
        gen_pids = data["gen_pids"]
        gen_mults = data["gen_mults"]

        size = len(det_particles)
        split = self.cfg.data.train_val_test
        train_idx, val_idx, test_idx = np.cumsum([int(s * size) for s in split])

        if self.cfg.data.embed_det_in_GA and self.cfg.data.add_spurions:
            self.spurions = None
        else:
            self.spurions = None

        self.train_data = Dataset(
            self.dtype,
            self.cfg.data.add_jet,
            self.cfg.data.embed_det_in_GA,
            self.spurions,
        )
        self.val_data = Dataset(
            self.dtype,
            self.cfg.data.add_jet,
            self.cfg.data.embed_det_in_GA,
            self.spurions,
        )
        self.test_data = Dataset(
            self.dtype,
            self.cfg.data.add_jet,
            self.cfg.data.embed_det_in_GA,
            self.spurions,
        )
        self.train_data.create_data_list(
            det_particles[:train_idx],
            det_pids[:train_idx],
            det_mults[:train_idx],
            gen_particles[:train_idx],
            gen_pids[:train_idx],
            gen_mults[:train_idx],
        )
        self.val_data.create_data_list(
            det_particles[train_idx:val_idx],
            det_pids[train_idx:val_idx],
            det_mults[train_idx:val_idx],
            gen_particles[train_idx:val_idx],
            gen_pids[train_idx:val_idx],
            gen_mults[train_idx:val_idx],
        )
        self.test_data.create_data_list(
            det_particles[val_idx:test_idx],
            det_pids[val_idx:test_idx],
            det_mults[val_idx:test_idx],
            gen_particles[val_idx:test_idx],
            gen_pids[val_idx:test_idx],
            gen_mults[val_idx:test_idx],
        )

    def _init_dataloader(self):
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

    def evaluate(self):
        with torch.no_grad():
            if self.ema is not None:
                with self.ema.average_parameters():
                    self.results_train = self._evaluate_single(
                        self.train_loader, "train"
                    )
                    self.results_val = self._evaluate_single(self.val_loader, "val")
                    self.results_test = self._evaluate_single(self.test_loader, "test")

                # also evaluate without ema to see the effect
                self._evaluate_single(self.train_loader, "train_noema")
                self._evaluate_single(self.val_loader, "val_noema")
                self._evaluate_single(self.test_loader, "test_noema")

            else:
                self.results_train = self._evaluate_single(self.train_loader, "train")
                self.results_val = self._evaluate_single(self.val_loader, "val")
                self.results_test = self._evaluate_single(self.test_loader, "test")
            if self.cfg.evaluation.save != 0:
                tensor_path = os.path.join(
                    self.cfg.run_dir, f"tensors_{self.cfg.run_idx}"
                )
                os.makedirs(tensor_path, exist_ok=True)
                torch.save(
                    self.results_test["samples"][: self.cfg.evaluation.save],
                    f"{tensor_path}/samples.pt",
                )
                torch.save(
                    self.results_test["params"][: self.cfg.evaluation.save],
                    f"{tensor_path}/params.pt",
                )

    def _evaluate_single(self, loader, title, step=None):
        LOGGER.info(
            f"### Starting to evaluate model on {title} dataset with "
            f"{len(loader.dataset)} elements, batchsize {loader.batch_size} ###"
        )
        metrics = {}
        self.model.eval()
        loss = []
        params = []
        samples = []
        with torch.no_grad():
            for batch in loader:
                batch_loss, batch_metrics = self._batch_loss(batch)
                loss.append(batch_loss)
                params.append(batch_metrics["params"])
                samples.append(batch_metrics["samples"])
        loss = torch.tensor(loss)
        LOGGER.info(f"NLL on {title} dataset: {loss.mean():.4f}")

        metrics["loss"] = loss.mean()
        metrics["params"] = torch.cat(params)
        metrics["samples"] = torch.cat(samples)
        if self.cfg.use_mlflow:
            for key, value in metrics.items():
                name = f"{title}"
                log_mlflow(f"{name}.{key}", value, step=step)
        return metrics

    def plot(self):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path, exist_ok=True)
        LOGGER.info(f"Creating plots in {plot_path}")

        plot_dict = {}
        if self.cfg.evaluate:
            plot_dict["results_train"] = self.results_train
            plot_dict["results_val"] = self.results_val
            plot_dict["results_test"] = self.results_test
        if self.cfg.train:
            plot_dict["train_loss"] = self.train_loss
            plot_dict["val_loss"] = self.val_loss
            plot_dict["train_lr"] = self.train_lr
        plot_mixer(self.cfg, plot_path, plot_dict)

    def _batch_loss(self, batch):
        predicted_dist, label = self._get_predicted_dist_and_label(batch)
        params = predicted_dist.params.cpu().detach()
        if self.cfg.dist.diff:
            if self.cfg.dist.type == "Categorical":
                # Rescale to have only positive indices
                loss = self.loss(
                    predicted_dist,
                    label - batch.x_det_ptr.diff() - self.cfg.data.diff[0],
                )
                # Rescale back to original range
                sample = (
                    (
                        batch.x_det_ptr.diff()
                        + predicted_dist.sample()
                        + self.cfg.data.diff[0]
                    )
                    .cpu()
                    .detach()
                )
            else:
                loss = self.loss(predicted_dist, label - batch.x_det_ptr.diff())
                sample = (
                    (batch.x_det_ptr.diff() + predicted_dist.sample()).cpu().detach()
                )
        else:
            loss = self.loss(predicted_dist, label)
            sample = predicted_dist.sample().cpu().detach()
        assert torch.isfinite(loss).all()
        det_mult = batch.x_det_ptr.diff().cpu()
        metrics = {
            "params": params,
            "samples": torch.stack([sample, label.cpu(), det_mult], dim=-1),
        }
        return loss, metrics

    def _get_predicted_dist_and_label(self, batch, min_arg=-10.0, max_arg=5.0):
        batch = batch.to(self.device)
        if self.cfg.modelname == "Transformer":
            scalars = batch.scalars_det
            input = torch.cat([batch.x_det, scalars], dim=-1)
            output = self.model(input, batch.x_det_batch)
        elif self.cfg.modelname == "GATr":
            embedding = embed_data_into_ga(
                batch.x_det,
                batch.scalars_det,
                batch.x_det_ptr,
                self.cfg.data,
            )
            output = self.model(embedding)

        params = torch.clamp(output, min=min_arg, max=max_arg)  # avoid inf and 0
        params = torch.exp(params)  # ensure positive params
        if self.distribution == "Categorical":
            params = torch.nn.functional.softmax(params, dim=-1)

        predicted_dist = self.distribution(params)  # batch of mixtures

        return predicted_dist, batch.x_gen_ptr.diff()

    def _init_metrics(self):
        return {"params": [], "samples": []}
