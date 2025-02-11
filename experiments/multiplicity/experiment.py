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
from experiments.multiplicity.utils import jetmomenta_to_fourmomenta
from experiments.multiplicity.embedding import embed_data_into_ga
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
                if self.cfg.data.pid_raw:
                    self.cfg.model.net.in_channels += 1
                elif self.cfg.data.pid_encoding:
                    self.cfg.model.net.in_channels += 6
                if self.cfg.dist.type == "GammaMixture":
                    self.distribution = GammaMixture
                    self.cfg.model.net.out_channels = 3 * self.cfg.dist.n_components
                elif self.cfg.dist.type == "GaussianMixture":
                    self.distribution = GaussianMixture
                    self.cfg.model.net.out_channels = 3 * self.cfg.dist.n_components
                elif self.cfg.dist.type == "Categorical":
                    self.distribution = CategoricalDistribution
                    self.cfg.model.net.out_channels = self.cfg.data.max_num_particles
            elif self.cfg.modelname == "GATr":
                if self.cfg.dist.type == "GammaMixture":
                    self.distribution = GammaMixture
                    self.cfg.model.net.out_mv_channels = 3 * self.cfg.dist.n_components
                elif self.cfg.dist.type == "GaussianMixture":
                    self.distribution = GaussianMixture
                    self.cfg.model.net.out_mv_channels = 3 * self.cfg.dist.n_components
                elif self.cfg.dist.type == "Categorical":
                    self.distribution = CategoricalDistribution
                    self.cfg.model.net.out_mv_channels = self.cfg.data.max_num_particles

                # no global token for the embedding
                self.cfg.data.include_global_token = False
                self.cfg.data.num_global_tokens = 0

                # scalar channels
                self.cfg.model.net.in_s_channels = 0
                if self.cfg.data.pid_raw:
                    self.cfg.model.net.in_s_channels += 1
                elif self.cfg.data.pid_encoding:
                    self.cfg.model.net.in_s_channels += 6
                if self.cfg.data.add_scalar_features:
                    self.cfg.model.net.in_s_channels += 7
                if self.cfg.data.include_global_token:
                    self.cfg.model.net.in_s_channels += self.cfg.data.num_global_tokens

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
        dataset = MultiplicityDataset(data_path=data_path, cfg=self.cfg)
        LOGGER.info(f"Created MultiplicityDataset in {time.time() - t0:.2f} seconds")
        self.data_train = dataset.train_data_list
        self.data_val = dataset.val_data_list
        self.data_test = dataset.test_data_list

    def _init_dataloader(self):
        self.train_loader = DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.training.batchsize,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )
        self.test_loader = DataLoader(
            dataset=self.data_test,
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
        loss = np.array(loss)
        LOGGER.info(
            f"Loss on {title} dataset: mean {loss.mean():.4f} , std {loss.std():.4f}"
        )
        metrics["loss"] = loss / len(loader.dataset)
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
        loss = self.loss(predicted_dist, label)
        assert torch.isfinite(loss).all()
        params = predicted_dist.params.cpu().detach()
        sample = predicted_dist.sample().cpu().detach()
        det_mult = torch.tensor(batch.det_mult).cpu()
        metrics = {
            "params": params,
            "samples": torch.stack([sample, label.cpu(), det_mult], dim=-1),
        }
        return loss, metrics

    def _get_predicted_dist_and_label(self, batch, min_arg=-10.0, max_arg=5.0):
        batch = batch.to(self.device)
        if self.cfg.modelname == "Transformer":
            input = torch.cat([batch.x, batch.scalars], dim=-1)
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

        predicted_dist = self.distribution(params)  # batch of mixtures

        return predicted_dist, batch.label

    def _init_metrics(self):
        return {"params": [], "samples": []}
