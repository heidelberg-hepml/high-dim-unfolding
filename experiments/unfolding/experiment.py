import numpy as np
import torch
from torch_geometric.data import DataLoader

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


class UnfoldingExperiment(BaseExperiment):
    def init_physics(self):

        # dynamically set wrapper properties
        self.modeltype = "CFM"
        self.modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]

        with open_dict(self.cfg):

            # dynamically set channel dimensions
            if self.modelname == "ConditionalGATr":
                self.cfg.model.net.in_s_channels = self.cfg.cfm.embed_t_dim
                self.cfg.model.net.condition_in_s_channels = self.cfg.cfm.embed_t_dim
                if self.cfg.data.save_pid:
                    self.cfg.model.net.in_s_channels += 1
                    self.cfg.model.net.condition_in_s_channels += 1
                if self.cfg.data.pid_encoding:
                    self.cfg.model.net.in_s_channels += 6
                    self.cfg.model.net.condition_in_s_channels += 6
                if self.cfg.model.beam_reference is not None:
                    self.cfg.model.net.condition_in_mv_channels += (
                        2
                        if (
                            self.cfg.model.two_beams
                            and self.cfg.model.beam_reference != "xyplane"
                        )
                        else 1
                    )
                if self.cfg.model.add_time_reference:
                    self.cfg.model.net.condition_in_mv_channels += 1

            elif self.modelname == "ConditionalTransformer":
                self.cfg.model.net.in_channels = 4 + self.cfg.cfm.embed_t_dim
                self.cfg.model.net.condition_in_channels = 4
                if self.cfg.data.save_pid:
                    self.cfg.model.net.in_channels += 1
                    self.cfg.model.net.condition_in_channels += 1
                if self.cfg.data.pid_encoding:
                    self.cfg.model.net.in_channels += 6
                    self.cfg.model.net.condition_in_channels += 6

            # copy model-specific parameters
            self.cfg.model.odeint = self.cfg.odeint
            self.cfg.model.cfm = self.cfg.cfm

    def init_data(self):
        data_path = os.path.join(self.cfg.data.data_dir, f"{self.cfg.data.dataset}")
        self._init_data(ZplusJetDataset, data_path)

    def _init_data(self, Dataset, data_path):
        LOGGER.info(f"Creating {Dataset.__name__} from {data_path}")
        t0 = time.time()

        data = energyflow.zjets_delphes.load(
            "Herwig",
            num_data=self.cfg.data.length,
            pad=True,
            cache_dir=data_path,
            include_keys=["particles", "jets", "mults"],
        )

        data["sim_particles"] = torch.tensor(data["sim_particles"])
        data["sim_jets"] = torch.tensor(data["sim_jets"])
        data["gen_particles"] = torch.tensor(data["gen_particles"])
        data["gen_jets"] = torch.tensor(data["gen_jets"])
        data["sim_mults"] = torch.tensor(data["sim_mults"])
        data["gen_mults"] = torch.tensor(data["gen_mults"])

        # undo dataset preprocessing
        data["sim_particles"][..., 1:3] = data["sim_particles"][..., 1:3] + data[
            "sim_jets"
        ][..., 1:3].unsqueeze(1)
        data["gen_particles"][..., 1:3] = data["gen_particles"][..., 1:3] + data[
            "gen_jets"
        ][..., 1:3].unsqueeze(1)

        train_idx = round(self.cfg.data.split[0] * data["sim_particles"].shape[0])
        val_idx = train_idx + round(
            self.cfg.data.split[1] * data["sim_particles"].shape[0]
        )

        # compute mean and std
        # train_det_particles = data["sim_particles"][0:train_idx]
        # n_train_det = torch.sum(data["sim_mults"][0:train_idx])
        train_gen_particles = data["gen_particles"][0:train_idx]
        n_train_gen = torch.sum(data["gen_mults"][0:train_idx])

        # train_det_mean = train_det_particles.sum(0, 1) / n_train_det
        # train_det_std = torch.sqrt(
        #     ((train_det_particles - train_det_mean) ** 2).sum(0, 1) / n_train_det
        # )
        train_gen_mean = train_gen_particles.sum(dim=[0, 1]) / n_train_gen
        train_gen_std = torch.sqrt(
            ((train_gen_particles - train_gen_mean.view(1, 1, 4)) ** 2).sum(dim=[0, 1])
            / n_train_gen
        )
        # remove it for the fixed mass
        # train_det_mean[..., 3] = 0.0
        # train_det_std[..., 3] = 1.0
        train_gen_mean[..., 3] = 0.0
        train_gen_std[..., 3] = 1.0

        with open_dict(self.cfg):
            # self.cfg.data.train_det_mean = train_det_mean.unsqueeze(0)
            # self.cfg.data.train_det_std = train_det_std.unsqueeze(0)
            self.cfg.data.train_gen_mean = [mean.item() for mean in train_gen_mean]
            self.cfg.data.train_gen_std = [std.item() for std in train_gen_std]

        self.train_dataset = ZplusJetDataset(self.cfg.data)
        self.test_dataset = ZplusJetDataset(self.cfg.data)
        self.val_dataset = ZplusJetDataset(self.cfg.data)

        self.train_dataset.load_data(data, (0, train_idx))
        self.test_dataset.load_data(data, (train_idx, val_idx))
        self.val_dataset.load_data(data, (val_idx, -1))

        # initialize cfm (might require data)
        self.model.init_physics(
            self.cfg.data.units,
            self.cfg.data.pt_min,
            self.cfg.data.train_gen_mean,
            self.cfg.data.train_gen_std,
            self.cfg.data.base_type,
            self.cfg.data.onshell_mass,
            self.device,
        )
        self.model.init_distribution()
        self.model.init_coordinates()
        self.model.init_geometry()
        print(self.train_dataset[0][0].x.dtype, self.train_dataset[0][0].scalars.dtype)

    def _init_dataloader(self):
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.training.batchsize,
            shuffle=True,
            collate_fn=collate,
        )
        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
            collate_fn=collate,
        )
        self.val_loader = DataLoader(
            dataset=self.val_dataset,
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
            self._sample_events()
            loaders["gen"] = self.sample_loader
        else:
            LOGGER.info("Skip sampling")

        if self.cfg.evaluation.classifier:
            self.classifiers = []
            for ijet, n_jets in enumerate(self.cfg.data.n_jets):
                self.classifiers.append(self._evaluate_classifier_metric(ijet, n_jets))

        for key in self.cfg.evaluation.eval_loss:
            if key in loaders.keys():
                self._evaluate_loss_single(loaders[key], key)

    def _evaluate_classifier_metric(self, ijet, n_jets):
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

    def _sample_events(self):
        pass

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
