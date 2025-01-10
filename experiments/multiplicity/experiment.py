import numpy as np
import einops
import torch
from torch_geometric.loader import DataLoader

import os, time
from omegaconf import open_dict

from experiments.base_experiment import BaseExperiment
from experiments.multiplicity.dataset import MultiplicityDataset
from experiments.multiplicity.utils import smoothCELoss
from experiments.multiplicity.plots import plot_mixer
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow

MODEL_TITLE_DICT = {"GATr": "GATr", "Transformer": "Tr"}


class MultiplicityExperiment(BaseExperiment):
    def _init_loss(self):
        self.loss = smoothCELoss(self.cfg.data.max_num_particles)

    def init_physics(self):
        pass

    def init_data(self):
        data_path = os.path.join(self.cfg.data.data_dir, f"{self.cfg.data.dataset}")
        self._init_data(MultiplicityDataset, data_path)

    def _init_data(self, Dataset, data_path):
        LOGGER.info(f"Creating {Dataset.__name__} from {data_path}")
        t0 = time.time()
        kwargs = {"rescale_data": self.cfg.data.rescale_data}
        self.data_train = Dataset(**kwargs)
        self.data_test = Dataset(**kwargs)
        self.data_val = Dataset(**kwargs)
        self.data_train.load_data(
            data_path,
            n_elements=self.cfg.data.length,
            mode="train",
            split=self.cfg.data.split,
        )
        self.data_test.load_data(
            data_path,
            n_elements=self.cfg.data.length,
            mode="test",
            split=self.cfg.data.split,
        )
        self.data_val.load_data(
            data_path,
            n_elements=self.cfg.data.length,
            mode="val",
            split=self.cfg.data.split,
        )
        dt = time.time() - t0
        LOGGER.info(f"Finished creating datasets after {dt:.2f} s = {dt/60:.2f} min")

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

    def _evaluate_single(self, loader, title):
        LOGGER.info(
            f"### Starting to evaluate model on {title} dataset with "
            f"{len(loader.dataset)} elements, batchsize {loader.batch_size} ###"
        )
        metrics = {}
        self.model.eval()
        if self.cfg.training.optimizer == "ScheduleFree":
            self.optimizer.eval()
        loss = 0.0
        with torch.no_grad():
            for batch in loader:
                loss += self._batch_loss(batch).sum()
        metrics["loss"] = loss / len(loader.dataset)

        if self.cfg.use_mlflow:
            for key, value in metrics.items():
                name = f"{title}"
                log_mlflow(f"{name}.{key}", value, step=step)
        return metrics

    def plot(self):
        raise NotImplementedError

    def _batch_loss(self, batch):
        predicted_dist, label = self._get_predicted_dist_and_label(batch)
        loss = self.loss(predicted_dist, label)
        assert torch.isfinite(loss).all()

        metrics = {}
        return loss, metrics

    def _get_predicted_dist_and_label(self, batch):
        dist_params = torch.nn.functional.softplus(self.model(batch.x, batch.batch))
        dist_params = einops.rearrange(
            dist_params, "b (n_mix n_params) -> b n_mix n_params", n_params=3
        )
        gammas = torch.distributions.Gamma(dist_params[:, :, 0], dist_params[:, :, 1])
        mix = torch.distributions.Categorical(dist_params[:, :, 2])
        predicted_dist = torch.distributions.MixtureSameFamily(mix, gammas)
        return predicted_dist, batch.label.to(self.dtype)

    def _init_metrics(self):
        return {}
