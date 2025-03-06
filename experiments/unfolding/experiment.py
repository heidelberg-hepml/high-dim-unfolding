import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

import os, time
from omegaconf import open_dict
from hydra.utils import instantiate
from tqdm import trange, tqdm
import energyflow

from experiments.base_experiment import BaseExperiment
from experiments.unfolding.dataset import ZplusJetDataset, collate
from experiments.unfolding.utils import ensure_angle, pid_encoding, get_batch_from_ptr
from experiments.unfolding.coordinates import PtPhiEtaM2
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
        data = energyflow.zjets_delphes.load(
            "Herwig",
            num_data=self.cfg.data.length,
            pad=True,
            cache_dir=data_path,
            include_keys=["particles", "mults", "jets"],
        )

        split = self.cfg.data.split
        size = len(data["sim_particles"])
        train_idx = int(split[0] * size)
        val_idx = int(split[1] * size)

        det_particles = torch.tensor(data["sim_particles"], dtype=self.dtype)
        det_jets = torch.tensor(data["sim_jets"], dtype=self.dtype)
        det_mults = torch.tensor(data["sim_mults"], dtype=torch.int)

        gen_particles = torch.tensor(data["gen_particles"], dtype=self.dtype)
        gen_jets = torch.tensor(data["gen_jets"], dtype=self.dtype)
        gen_mults = torch.tensor(data["gen_mults"], dtype=torch.int)

        # undo the dataset scaling
        det_particles[..., 1:3] = det_particles[..., 1:3] + det_jets[:, None, 1:3]
        det_particles[..., 2] = ensure_angle(det_particles[..., 2])
        det_particles[..., 0] = det_particles[..., 0] * 100

        gen_particles[..., 1:3] = gen_particles[..., 1:3] + gen_jets[:, None, 1:3]
        gen_particles[..., 2] = ensure_angle(gen_particles[..., 2])
        gen_particles[..., 0] = gen_particles[..., 0] * 100

        # swap eta and phi for consistency
        det_particles[..., [1, 2]] = det_particles[..., [2, 1]]
        gen_particles[..., [1, 2]] = gen_particles[..., [2, 1]]

        # save pids before replacing with mass
        if self.cfg.data.pid_encoding:
            det_pids = det_particles[..., 3].clone().unsqueeze(-1)
            det_pids = pid_encoding(det_pids)
            gen_pids = gen_particles[..., 3].clone().unsqueeze(-1)
            gen_pids = pid_encoding(gen_pids)
        else:
            det_pids = torch.empty(*det_particles.shape[:-1], 0, dtype=self.dtype)
            gen_pids = torch.empty(*gen_particles.shape[:-1], 0, dtype=self.dtype)

        det_particles[..., 3] = self.cfg.data.mass**2
        gen_particles[..., 3] = self.cfg.data.mass**2

        DatasetCoordinates = PtPhiEtaM2()
        det_particles = DatasetCoordinates.x_to_fourmomenta(det_particles)
        gen_particles = DatasetCoordinates.x_to_fourmomenta(gen_particles)

        if self.cfg.data.standardize:
            mask = (
                torch.arange(det_particles.shape[1])[None, :]
                < det_mults[:train_idx, None]
            )
            flattened_particles = det_particles[:train_idx][mask]

            if self.cfg.modelname == "ConditionalGATr":
                # For GATr, same standardization for all components
                # mean = flattened_particles.mean().unsqueeze(0).expand(1, 4)
                mean = torch.zeros(1, 4, dtype=self.dtype)
                std = flattened_particles.std().unsqueeze(0).expand(1, 4)
            elif self.cfg.modelname == "ConditionalTransformer":
                # Otherwise, standardization done separately for each component
                mean = flattened_particles.mean(dim=0, keepdim=True)
                mean[..., -1] = 0
                std = flattened_particles.std(dim=0, keepdim=True)
                std[..., -1] = 1
            det_particles = (det_particles - mean) / std

        self.train_data = Dataset(self.dtype)
        self.val_data = Dataset(self.dtype)
        self.test_data = Dataset(self.dtype)

        self.train_data.create_data_list(
            det_particles[:train_idx],
            det_pids[:train_idx],
            det_mults[:train_idx],
            gen_particles[:train_idx],
            gen_pids[:train_idx],
            gen_mults[:train_idx],
        )
        self.val_data.create_data_list(
            det_particles[train_idx : train_idx + val_idx],
            det_pids[train_idx : train_idx + val_idx],
            det_mults[train_idx : train_idx + val_idx],
            gen_particles[train_idx : train_idx + val_idx],
            gen_pids[train_idx : train_idx + val_idx],
            gen_mults[train_idx : train_idx + val_idx],
        )
        self.test_data.create_data_list(
            det_particles[train_idx + val_idx :],
            det_pids[train_idx + val_idx :],
            det_mults[train_idx + val_idx :],
            gen_particles[train_idx + val_idx :],
            gen_pids[train_idx + val_idx :],
            gen_mults[train_idx + val_idx :],
        )

        # initialize cfm (might require data)
        self.model.init_physics(
            self.cfg.data.units,
            self.cfg.data.pt_min,
            self.cfg.data.base_type,
            self.cfg.data.mass,
            self.device,
        )
        self.model.init_distribution()
        self.model.init_coordinates()
        mask = (
            torch.arange(gen_particles.shape[1])[None, :] < gen_mults[:train_idx, None]
        )
        fit_data = gen_particles[:train_idx][mask]
        self.model.coordinates.init_fit(fit_data)
        if hasattr(self.model, "distribution"):
            self.model.distribution.coordinates.init_fit(fit_data)
        self.model.init_geometry()

    def _init_dataloader(self):
        train_sampler = torch.utils.data.DistributedSampler(
            self.train_data,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )
        self.train_loader = DataLoader(
            dataset=self.train_data,
            batch_size=self.cfg.training.batchsize,
            sampler=train_sampler,
            collate_fn=collate,
        )
        test_sampler = torch.utils.data.DistributedSampler(
            self.test_data,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            dataset=self.test_data,
            batch_size=self.cfg.evaluation.batchsize,
            sampler=test_sampler,
            collate_fn=collate,
        )
        val_sampler = torch.utils.data.DistributedSampler(
            self.val_data,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            dataset=self.val_data,
            batch_size=self.cfg.evaluation.batchsize,
            sampler=val_sampler,
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
            LOGGER.info(
                f"Sampling {self.cfg.evaluation.n_batches} batches for evaluation"
            )
            t0 = time.time()
            samples, samples_ptr, targets, targets_ptr, base_samples = (
                self._sample_events(loaders["train"], self.cfg.evaluation.n_batches)
            )
            dt = time.time() - t0
            LOGGER.info(f"Finished sampling after {dt/60:.2f}min")
            samples_batches = get_batch_from_ptr(samples_ptr)
            targets_batches = get_batch_from_ptr(targets_ptr)
            jet_samples = scatter(samples, samples_batches, dim=0, reduce="sum")
            jet_base_samples = scatter(
                base_samples, samples_batches, dim=0, reduce="sum"
            )
            jet_targets = scatter(targets, targets_batches, dim=0, reduce="sum")
            plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
            os.makedirs(plot_path, exist_ok=True)
            plot_kinematics(plot_path, jet_samples, jet_targets, jet_base_samples)

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
            f"Finished evaluating loss for {title} dataset after {dt/60:.2f}min: {np.mean(losses):.4f}"
        )

        if self.cfg.use_mlflow:
            log_mlflow(f"eval.{title}.loss", np.mean(losses))

    def _evaluate_log_prob_single(self, loader, title):
        pass

    def _sample_events(self, loader, n_batches):
        samples = torch.empty((0, 4), device=self.device)
        targets = torch.empty((0, 4), device=self.device)
        base_samples = torch.empty((0, 4), device=self.device)
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
                    target_ptr[1:] + targets_ptr[-1],
                ],
                dim=0,
            )
            sample, base_sample, ptr = self.model.sample(
                batch,
                self.device,
                self.dtype,
            )
            samples = torch.cat([samples, sample], dim=0)
            base_samples = torch.cat([base_samples, base_sample], dim=0)
            samples_ptr = torch.cat([samples_ptr, ptr[1:] + samples_ptr[-1]], dim=0)

        return samples, samples_ptr, targets, targets_ptr, base_samples

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
