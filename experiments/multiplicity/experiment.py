import numpy as np
import einops
import torch
from torch_geometric.loader import DataLoader
import energyflow

import os, time
from omegaconf import open_dict

from experiments.base_experiment import BaseExperiment
from experiments.multiplicity.dataset import MultiplicityDataset
from experiments.multiplicity.distributions import GammaMixture, CategoricalDistribution,cross_entropy, smooth_cross_entropy
from experiments.multiplicity.plots import plot_mixer
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow

MODEL_TITLE_DICT = {"GATr": "GATr", "Transformer": "Tr"}


class MultiplicityExperiment(BaseExperiment):
    def _init_loss(self):
        if self.cfg.loss.type == "cross_entropy":
            self.loss = lambda dist, target: cross_entropy(dist,target).sum()
        elif self.cfg.loss.type == "smooth_cross_entropy":
            self.loss = lambda dist, target: smooth_cross_entropy(dist,target, self.cfg.data.max_num_particles, self.cfg.loss.smoothness).sum()

    def init_physics(self):
        with open_dict(self.cfg):
            if self.cfg.dist.type == "GammaMixture":
                self.distribution = GammaMixture
                self.cfg.model.net.out_channels = 3 * self.cfg.dist.n_components
            elif self.cfg.dist.type == "Categorical":
                self.distribution = CategoricalDistribution
                self.cfg.model.net.out_channels = self.cfg.data.max_num_particles

    def init_data(self):
        data_path = os.path.join(self.cfg.data.data_dir, f"{self.cfg.data.dataset}")
        self._init_data(MultiplicityDataset, data_path)

    def _init_data(self, Dataset, data_path):
        LOGGER.info(f"Creating {Dataset.__name__} from {data_path}")
        t0 = time.time()
        kwargs = {"rescale_data": self.cfg.data.rescale_data}

        data = energyflow.zjets_delphes.load(
            "Herwig",
            num_data=self.cfg.data.length,
            pad=True,
            cache_dir=data_path,
            include_keys=["particles", "mults"],
        )

        shuffle_indices = torch.randperm(len(data["sim_particles"]))
        data["sim_particles"] = data["sim_particles"][shuffle_indices]
        data["sim_mults"] = data["sim_mults"][shuffle_indices]
        data["gen_mults"] = data["gen_mults"][shuffle_indices]

        self.data_train = Dataset(**kwargs)
        self.data_test = Dataset(**kwargs)
        self.data_val = Dataset(**kwargs)
        self.data_train.load_data(
            data,
            mode="train",
            split=self.cfg.data.split,
        )
        self.data_test.load_data(
            data,
            mode="test",
            split=self.cfg.data.split,
        )
        self.data_val.load_data(
            data,
            mode="val",
            split=self.cfg.data.split,
        )
        dt = time.time() - t0
        LOGGER.info(f"Finished creating datasets after {dt:.2f} s = {dt/60:.2f} min")
        del data

    def _init_dataloader(self):
        self.train_loader = DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.training.batchsize,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )
        self.val_loader = DataLoader(
            dataset=self.data_val,
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
        loss = 0.0
        params = []
        samples = []
        with torch.no_grad():
            for batch in loader:
                batch_loss, batch_metrics = self._batch_loss(batch)
                loss += batch_loss
                params.append(batch_metrics["params"])
                samples.append(batch_metrics["samples"])
        metrics["loss"] = (loss / len(loader.dataset)).cpu().item()
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
        model_title = MODEL_TITLE_DICT[type(self.model.net).__name__]
        title = model_title
        LOGGER.info(f"Creating plots in {plot_path}")

        plot_dict = {}
        if self.cfg.evaluate:
            plot_dict["results_train"] = self.results_train
            plot_dict["results_test"] = self.results_test
            plot_dict["results_val"] = self.results_val
        if self.cfg.train:
            plot_dict["train_loss"] = self.train_loss
            plot_dict["val_loss"] = self.val_loss
            plot_dict["train_lr"] = self.train_lr
        plot_mixer(self.cfg, plot_path, title, plot_dict)

    def _batch_loss(self, batch):
        predicted_dist, label = self._get_predicted_dist_and_label(batch)
        loss = self.loss(predicted_dist, label)
        assert torch.isfinite(loss).all()
        params = predicted_dist.params.cpu().detach()
        sample = predicted_dist.sample().cpu().detach()
        sim_mult = torch.tensor(batch.sim_mult)
        metrics = {
            "params": params,
            "samples": torch.stack([sample, label.cpu(), sim_mult], dim=-1),
        }
        return loss, metrics

    def _get_predicted_dist_and_label(self, batch, min_sigmaarg=-10, max_sigmaarg=5.0):
        batch = batch.to(self.device)
        output = self.model(batch.x, batch.batch)
        # avoid inf and 0 (unstable)
        sigmaarg = torch.clamp(output, min=min_sigmaarg, max=max_sigmaarg)
        sigma = torch.exp(sigmaarg)
        assert torch.isfinite(sigma).all()
        predicted_dist = self.distribution(sigma)
        return predicted_dist, batch.label

    def _init_metrics(self):
        return {"params": [], "samples": []}
