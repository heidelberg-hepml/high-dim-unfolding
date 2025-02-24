import numpy as np
import einops
import torch
from torch_geometric.loader import DataLoader
import energyflow

import os, time
from omegaconf import open_dict

from experiments.base_experiment import BaseExperiment
from experiments.multiplicity.dataset import MultiplicityDataset
from experiments.multiplicity.distributions import (
    GammaMixture,
    CategoricalDistribution,
    GaussianMixture,
    cross_entropy,
    smooth_cross_entropy,
)
from experiments.multiplicity.plots import plot_mixer
from experiments.multiplicity.utils import (
    ensure_angle,
    jetmomenta_to_fourmomenta,
    pid_encoding,
)
from experiments.multiplicity.embedding import (
    embed_data_into_ga,
    compute_scalar_features,
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
            if self.cfg.modelname == "Transformer":
                self.cfg.model.net.in_channels = 4
                if self.cfg.data.pid_encoding:
                    self.cfg.model.net.in_channels += 6
                if self.cfg.data.add_scalar_features:
                    self.cfg.model.net.in_channels += 7
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
                        self.cfg.model.net.out_channels = (
                            self.cfg.data.diff[1] - self.cfg.data.diff[0] + 1
                        )
                    else:
                        self.cfg.model.net.out_channels = (
                            self.cfg.data.max_num_particles + 1
                        )

                # scalar channels
                self.cfg.model.net.in_s_channels = 0
                if self.cfg.data.pid_encoding:
                    self.cfg.model.net.in_s_channels += 6
                if self.cfg.data.add_scalar_features:
                    self.cfg.model.net.in_s_channels += 7

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
        data = energyflow.zjets_delphes.load(
            "Herwig",
            num_data=self.cfg.data.length,
            pad=True,
            cache_dir=data_path,
            include_keys=["particles", "mults", "jets"],
        )

        split = self.cfg.data.split
        size = len(self.data["sim_particles"])
        train_idx = int(split[0] * size)
        val_idx = int(split[1] * size)

        det_particles = torch.tensor(self.data["sim_particles"], dtype=self.dtype)
        det_jets = torch.tensor(self.data["sim_jets"], dtype=self.dtype)
        det_mults = torch.tensor(self.data["sim_mults"], dtype=torch.int)
        gen_mults = torch.tensor(self.data["gen_mults"], dtype=torch.int)

        # undo the dataset scaling
        det_particles[..., 1:3] = det_particles[..., 1:3] + det_jets[:, None, 1:3]
        det_particles[..., 2] = ensure_angle(det_particles[..., 2])
        det_particles[..., 0] = det_particles[..., 0] * 100

        # swap eta and phi for consistency
        det_particles[..., [1, 2]] = det_particles[..., [2, 1]]

        # save pids before replacing with mass
        det_pids = det_particles[..., 3].clone().unsqueeze(-1)
        if self.cfg.data.pid_encoding:
            det_pids = pid_encoding(det_pids)
        det_particles[..., 3] = self.cfg.data.mass

        if self.cfg.modelname == "GATr":
            det_particles = jetmomenta_to_fourmomenta(det_particles)

        if self.cfg.data.standardize:
            mask = (
                torch.arange(det_particles.shape[1])[None, :]
                < det_mults[:train_idx, None]
            )
            flattened_particles = det_particles[:train_idx][mask]

            if self.cfg.modelname == "GATr":
                # For GATr, same standardization for all components
                mean = flattened_particles.mean().unsqueeze(0).expand(1, 4)
                std = flattened_particles.std().unsqueeze(0).expand(1, 4)
            elif self.cfg.modelname == "Transformer":
                # Otherwise, standardization done separately for each component
                mean = flattened_particles.mean(dim=0, keepdim=True)
                mean[..., -1] = 0
                std = flattened_particles.std(dim=0, keepdim=True)
                std[..., -1] = 1
            det_particles = (det_particles - mean) / std

        self.train_data = MultiplicityDataset(self.cfg.data.pid_encoding, self.dtype)
        self.val_data = MultiplicityDataset(self.cfg.data.pid_encoding, self.dtype)
        self.test_data = MultiplicityDataset(self.cfg.data.pid_encoding, self.dtype)

        self.train_data.create_data_list(
            det_particles[:train_idx],
            det_pids[:train_idx],
            det_mults[:train_idx],
            gen_mults[:train_idx],
        )
        self.val_data.create_data_list(
            det_particles[train_idx : train_idx + val_idx],
            det_pids[train_idx : train_idx + val_idx],
            det_mults[train_idx : train_idx + val_idx],
            gen_mults[train_idx : train_idx + val_idx],
        )
        self.test_data.create_data_list(
            det_particles[train_idx + val_idx :],
            det_pids[train_idx + val_idx :],
            det_mults[train_idx + val_idx :],
            gen_mults[train_idx + val_idx :],
        )

    def _init_dataloader(self):
        self.train_loader = DataLoader(
            dataset=self.train_data,
            batch_size=self.cfg.training.batchsize,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            dataset=self.val_data,
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )
        self.test_loader = DataLoader(
            dataset=self.test_data,
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
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
        loss = torch.tensor(loss)  # .detach().cpu()
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
                    predicted_dist, label - batch.det_mult - self.cfg.data.diff[0]
                )
                # Rescale back to original range
                sample = (
                    (batch.det_mult + predicted_dist.sample() + self.cfg.data.diff[0])
                    .cpu()
                    .detach()
                )
            else:
                loss = self.loss(predicted_dist, label - batch.det_mult)
                sample = (batch.det_mult + predicted_dist.sample()).cpu().detach()
        else:
            loss = self.loss(predicted_dist, label)
            sample = predicted_dist.sample().cpu().detach()
        assert torch.isfinite(loss).all()
        det_mult = batch.det_mult.cpu()
        metrics = {
            "params": params,
            "samples": torch.stack([sample, label.cpu(), det_mult], dim=-1),
        }
        return loss, metrics

    def _get_predicted_dist_and_label(self, batch, min_arg=-10.0, max_arg=5.0):
        batch = batch.to(self.device)
        if self.cfg.modelname == "Transformer":
            if self.cfg.data.add_scalar_features:
                scalar_features = compute_scalar_features(
                    batch.x, batch.ptr, self.cfg.data
                )
                scalars = torch.cat([batch.scalars, scalar_features], dim=-1)
            else:
                scalars = batch.scalars
            input = torch.cat([batch.x, scalars], dim=-1)
            output = self.model(input, batch.batch)
        elif self.cfg.modelname == "GATr":
            embedding = embed_data_into_ga(
                batch.x,
                batch.scalars,
                batch.ptr,
                self.cfg.data,
            )
            output = self.model(embedding)

        params = torch.clamp(output, min=min_arg, max=max_arg)  # avoid inf and 0
        params = torch.exp(params)  # ensure positive params
        if self.distribution == "Categorical":
            params = torch.nn.functional.softmax(params, dim=-1)

        predicted_dist = self.distribution(params)  # batch of mixtures

        return predicted_dist, batch.label

    def _init_metrics(self):
        return {"params": [], "samples": []}
