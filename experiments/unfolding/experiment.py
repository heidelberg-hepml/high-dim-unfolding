import numpy as np
import torch
from torch_geometric.loader import DataLoader

import os, time
from omegaconf import open_dict
from hydra.utils import instantiate
from tqdm import trange, tqdm
import energyflow

from experiments.base_experiment import BaseExperiment
from experiments.unfolding.dataset import ZplusJetDataset, collate
import experiments.unfolding.plotter as plotter
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow

from experiments.unfolding.plots import plot_kinematics


class UnfoldingExperiment(BaseExperiment):
    def init_physics(self):

        # dynamically set wrapper properties
        self.modeltype = "CFM"

        with open_dict(self.cfg):
            self.cfg.modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]
            # dynamically set channel dimensions
            if self.cfg.modelname == "ConditionalGATr":
                self.cfg.model.net.in_s_channels = self.cfg.cfm.embed_t_dim
                self.cfg.model.net.condition_s_channels = 0
                if self.cfg.data.pid_raw:
                    self.cfg.model.net.in_s_channels += 1
                    self.cfg.model.net.condition_s_channels += 1
                elif self.cfg.data.pid_encoding:
                    self.cfg.model.net.in_s_channels += 6
                    self.cfg.model.net.condition_s_channels += 6
                if self.cfg.data.add_scalar_features:
                    self.cfg.model.net.condition_s_channels += 7
                if not self.cfg.data.beam_token:
                    self.cfg.model.net.condition_mv_channels += (
                        2
                        if (
                            self.cfg.data.two_beams
                            and self.cfg.data.beam_reference != "xyplane"
                        )
                        else 1
                    )
                    if self.cfg.data.add_time_reference:
                        self.cfg.model.net.condition_mv_channels += 1
                self.cfg.model.cfg_data = self.cfg.data

            elif self.cfg.modelname == "ConditionalTransformer":
                self.cfg.model.net.in_channels = 4 + self.cfg.cfm.embed_t_dim
                self.cfg.model.net.condition_channels = 4
                if self.cfg.data.pid_raw:
                    self.cfg.model.net.in_channels += 1
                    self.cfg.model.net.condition_channels += 1
                if self.cfg.data.pid_encoding:
                    self.cfg.model.net.in_channels += 6
                    self.cfg.model.net.condition_channels += 6

            # copy model-specific parameters
            self.cfg.model.odeint = self.cfg.odeint
            self.cfg.model.cfm = self.cfg.cfm

    def init_data(self):
        data_path = os.path.join(self.cfg.data.data_dir, f"{self.cfg.data.dataset}")
        self._init_data(ZplusJetDataset, data_path)

    def _init_data(self, Dataset, data_path):
        self.dataset = Dataset(data_path, self.cfg)

        # initialize cfm (might require data)
        self.model.init_physics(
            self.cfg.data.units,
            self.cfg.data.pt_min,
            self.cfg.data.train_gen_mean,
            self.cfg.data.train_gen_std,
            self.cfg.data.base_type,
            self.cfg.data.mass,
            self.device,
        )
        self.model.init_distribution()
        self.model.init_coordinates()
        self.model.init_geometry()

    def _init_dataloader(self):
        self.train_loader = DataLoader(
            dataset=self.dataset.train_data_list,
            batch_size=self.cfg.training.batchsize,
            shuffle=True,
            collate_fn=collate,
        )
        self.test_loader = DataLoader(
            dataset=self.dataset.test_data_list,
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
            collate_fn=collate,
        )
        self.val_loader = DataLoader(
            dataset=self.dataset.val_data_list,
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
            collate_fn=collate,
        )

        LOGGER.info(
            f"Constructed dataloaders with "
            f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )

    @torch.no_grad()
    def evaluate(self):
        # EMA-evaluation not implemented
        loaders = {
            "train": self.train_loader,
            "test": self.test_loader,
            "val": self.val_loader,
        }
        if self.cfg.evaluation.sample:
            samples, samples_ptr, targets, targets_ptr = self._sample_events(
                loaders["train"], self.cfg.evaluation.n_batches
            )
            plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
            os.makedirs(plot_path, exist_ok=True)
            plot_kinematics(plot_path, samples, targets)

        else:
            LOGGER.info("Skip sampling")

        if self.cfg.evaluation.classifier:
            self.classifiers = []
            for ijet, n_jets in enumerate(self.cfg.data.n_jets):
                self.classifiers.append(self._evaluate_classifier_metric(ijet, n_jets))

        for key in self.cfg.evaluation.eval_loss:
            if key in loaders.keys():
                self._evaluate_loss_single(loaders[key], key)

    def _evaluate_classifier_metric(self):
        pass

    def _evaluate_loss_single(self, loader, title):
        self.model.eval()
        losses = []
        LOGGER.info(f"Starting to evaluate loss for model on {title} dataset")
        t0 = time.time()
        for i, data in enumerate(loader):
            loss = 0.0
            data[0], data[1] = data[0].to(self.device), data[1].to(self.device)
            loss = self.model.batch_loss(data)[0]
            losses.append(loss.cpu().item())
        dt = time.time() - t0
        LOGGER.info(
            f"Finished evaluating loss for {title} dataset after {dt/60:.2f}min"
        )

        if self.cfg.use_mlflow:
            log_mlflow(f"eval.{title}.loss", np.mean(losses))

    def _evaluate_log_prob_single(self, loader, title):
        pass

    def _sample_events(self, loader, n_batches):
        samples = torch.empty((0, 4), device=self.device)
        targets = torch.empty((0, 4), device=self.device)
        samples_ptr = torch.zeros((1,), device=self.device)
        targets_ptr = torch.zeros((1,), device=self.device)
        it = iter(loader)
        for i in range(n_batches):
            batch = next(it)
            batch[0], batch[1] = batch[0].to(self.device), batch[1].to(self.device)
            batch = (batch[0], batch[1])

            target = batch[0].x
            target_ptr = batch[0].ptr
            targets = torch.cat([targets, target], dim=0)
            targets_ptr = torch.cat(
                [
                    targets_ptr,
                    torch.tensor(target_ptr[1:]) + targets_ptr[-1],
                ],
                dim=0,
            )

            sample, ptr = self.model.sample(
                batch,
                self.device,
                self.dtype,
            )
            samples = torch.cat([samples, sample], dim=0)
            samples_ptr = torch.cat(
                [samples_ptr, torch.tensor(ptr[1:]) + samples_ptr[-1]], dim=0
            )

        return samples, samples_ptr, targets, targets_ptr

    def plot(self):
        pass

    def _init_loss(self):
        # loss defined manually within the model
        pass

    def _batch_loss(self, batch):
        batch[0], batch[1] = batch[0].to(self.device), batch[1].to(self.device)
        loss, component_loss = self.model.batch_loss(batch)
        mse = loss.cpu().item()
        component_mse = [x.cpu().item() for x in component_loss]
        assert torch.isfinite(loss).all()
        metrics = {"mse": mse}
        for k in range(4):
            metrics[f"mse_{k}"] = component_mse[k]
        return loss, metrics

    def _init_metrics(self):
        metrics = {"mse": []}
        for k in range(4):
            metrics[f"mse_{k}"] = []
        return metrics
