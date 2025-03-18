import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from torch_geometric.data import Batch

import os, time
from omegaconf import open_dict
from hydra.utils import instantiate
from tqdm import trange, tqdm
import energyflow

from gatr.interface import embed_spurions, extract_vector
from experiments.base_experiment import BaseExperiment
from experiments.unfolding.dataset import ZplusJetDataset
from experiments.unfolding.utils import (
    ensure_angle,
    pid_encoding,
    get_ptr_from_batch,
)
from experiments.unfolding.coordinates import PtPhiEtaM2
import experiments.unfolding.plotter as plotter
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow


class UnfoldingExperiment(BaseExperiment):
    def init_physics(self):

        self.define_process_specifics()

        # dynamically set wrapper properties
        self.modeltype = "CFM"

        with open_dict(self.cfg):
            self.cfg.modelname = self.cfg.model._target_.rsplit(".", 1)[-1][:-3]
            # dynamically set channel dimensions
            if self.cfg.modelname == "ConditionalGATr":
                self.cfg.data.embed_det_with_spurions = True
                self.cfg.model.net.in_s_channels = self.cfg.cfm.embed_t_dim
                self.cfg.model.net_condition.in_s_channels = 0
                self.cfg.model.net_condition.out_mv_channels = (
                    self.cfg.model.net.hidden_mv_channels
                )
                self.cfg.model.net_condition.out_s_channels = (
                    self.cfg.model.net.hidden_s_channels
                )
                if self.cfg.data.pid_raw:
                    self.cfg.model.net.in_s_channels += 1
                    self.cfg.model.net_condition.in_s_channels += 1
                elif self.cfg.data.pid_encoding:
                    self.cfg.model.net.in_s_channels += 6
                    self.cfg.model.net_condition.in_s_channels += 6
                if self.cfg.data.add_scalar_features:
                    self.cfg.model.net_condition.in_s_channels += 7
                if not self.cfg.data.beam_token:
                    self.cfg.model.net_condition.in_mv_channels += (
                        2
                        if (
                            self.cfg.data.two_beams
                            and self.cfg.data.beam_reference != "xyplane"
                        )
                        else 1
                    )
                    if self.cfg.data.add_time_reference:
                        self.cfg.model.net_condition.in_mv_channels += 1
                self.cfg.model.cfg_data = self.cfg.data

            elif self.cfg.modelname == "ConditionalTransformer":
                self.cfg.data.embed_det_with_spurions = False
                self.cfg.model.net.in_channels = 4 + self.cfg.cfm.embed_t_dim
                self.cfg.model.net_condition.in_channels = 4
                self.cfg.model.net_condition.out_channels = (
                    self.cfg.model.net.hidden_channels
                )
                if self.cfg.data.pid_raw:
                    self.cfg.model.net.in_channels += 1
                    self.cfg.model.net_condition.in_channels += 1
                if self.cfg.data.pid_encoding:
                    self.cfg.model.net.in_channels += 6
                    self.cfg.model.net_condition.in_channels += 6
            elif self.cfg.modelname == "ConditionalAutoregressiveTransformer":
                self.cfg.data.embed_det_with_spurions = False
                self.cfg.model.autoregressive_tr.in_channels = 4
                self.cfg.model.net_condition.in_channels = 4
                self.cfg.model.net_condition.out_channels = (
                    self.cfg.model.autoregressive_tr.hidden_channels
                )
                self.cfg.model.autoregressive_tr.out_channels = (
                    self.cfg.model.autoregressive_tr.hidden_channels
                )
                self.cfg.model.mlp.in_shape = (
                    4
                    + self.cfg.cfm.embed_t_dim
                    + self.cfg.model.autoregressive_tr.out_channels
                )
                self.cfg.model.mlp.out_shape = 4
                self.cfg.model.mlp.hidden_channels = (
                    self.cfg.model.autoregressive_tr.hidden_channels
                )

            # copy model-specific parameters
            self.cfg.model.odeint = self.cfg.odeint
            self.cfg.model.cfm = self.cfg.cfm

    def init_data(self):
        t0 = time.time()
        data_path = os.path.join(self.cfg.data.data_dir, f"{self.cfg.data.dataset}")
        LOGGER.info(f"Creating ZplusJetDataset from {data_path}")
        self._init_data(ZplusJetDataset, data_path)
        LOGGER.info(
            f"Created ZplusJetDataset with {len(self.train_data)} training events, {len(self.val_data)} validation events, and {len(self.test_data)} test events in {time.time() - t0:.2f} seconds"
        )

    def _init_data(self, Dataset, data_path):
        t0 = time.time()
        data = energyflow.zjets_delphes.load(
            "Herwig",
            num_data=self.cfg.data.length,
            pad=True,
            cache_dir=data_path,
            include_keys=["particles", "mults", "jets"],
        )
        LOGGER.info(f"Loaded data in {time.time() - t0:.2f} seconds")
        split = self.cfg.data.train_test_val
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

        if self.cfg.data.embed_det_with_spurions:
            self.spurions = embed_spurions(
                self.cfg.data.beam_reference,
                self.cfg.data.add_time_reference,
                self.cfg.data.two_beams,
                self.cfg.data.add_xzplane,
                self.cfg.data.add_yzplane,
            )
        else:
            self.spurions = None

        self.train_data = Dataset(
            self.cfg.data.max_constituents,
            self.dtype,
            self.cfg.data.embed_det_with_spurions,
            self.spurions,
        )
        self.val_data = Dataset(
            self.cfg.data.max_constituents,
            self.dtype,
            self.cfg.data.embed_det_with_spurions,
            self.spurions,
        )
        self.test_data = Dataset(
            self.cfg.data.max_constituents,
            self.dtype,
            self.cfg.data.embed_det_with_spurions,
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
        gen_mask = (
            torch.arange(gen_particles.shape[1])[None, :] < gen_mults[:train_idx, None]
        )
        fit_gen_data = gen_particles[:train_idx][gen_mask]
        self.model.coordinates.init_fit(fit_gen_data)
        if hasattr(self.model, "distribution"):
            self.model.distribution.coordinates.init_fit(fit_gen_data)

        det_mask = (
            torch.arange(det_particles.shape[1])[None, :] < det_mults[:train_idx, None]
        )
        fit_det_data = det_particles[:train_idx][det_mask]
        self.model.condition_coordinates.init_fit(fit_det_data)

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
            self._sample_events(loaders["test"], self.cfg.evaluation.n_batches)
            loaders["gen"] = self.sample_loader
            dt = time.time() - t0
            LOGGER.info(f"Finished sampling after {dt/60:.2f}min")
        else:
            LOGGER.info("Skip sampling")

        if self.cfg.evaluation.classifier:
            self.classifier = self._evaluate_classifier_metric()

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
            data = data.to(self.device)
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
        samples = []
        targets = []
        self.data_raw = {}
        it = iter(loader)
        if n_batches > len(loader):
            LOGGER.warning(
                f"Requested {n_batches} batches for sampling, but only {len(loader)} batches available in test dataset."
            )
            n_batches = len(loader)
        for i in range(n_batches):
            batch = next(it).to(self.device)
            sample_batch = self.model.sample(
                batch,
                self.device,
                self.dtype,
            )
            samples.extend(sample_batch.to_data_list())
            targets.extend(batch.to_data_list())

        if self.cfg.data.embed_det_with_spurions:
            if len(self.spurions) > 0:
                for data in samples:
                    data.x_det = extract_vector(
                        data.x_det[: -len(self.spurions)]
                    ).squeeze(-2)
                for data in targets:
                    data.x_det = extract_vector(
                        data.x_det[: -len(self.spurions)]
                    ).squeeze(-2)

        self.data_raw["gen"] = Batch.from_data_list(
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

    def plot(self):
        path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(os.path.join(path), exist_ok=True)
        LOGGER.info(f"Creating plots in {path}")
        t0 = time.time()

        if self.cfg.modelname == "ConditionalTransformer":
            model_label = "CondTr"
        elif self.cfg.modelname == "ConditionalGATr":
            model_label = "CondGATr"
        elif self.cfg.modelname == "ConditionalAutoregressiveTransformer":
            model_label = "CondARTr"
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
        if (
            self.cfg.plotting.log_prob
            and len(self.cfg.evaluation.eval_log_prob) > 0
            and self.cfg.evaluate
        ):
            filename = os.path.join(path, "neg_log_prob.pdf")
            plotter.plot_log_prob(filename=filename, **kwargs)

        if self.cfg.evaluation.classifier and self.cfg.evaluate:
            filename = os.path.join(path, "classifier.pdf")
            plotter.plot_classifier(filename=filename, **kwargs)

        if self.cfg.evaluation.sample:
            if self.cfg.plotting.fourmomenta:
                filename = os.path.join(path, "fourmomenta.pdf")
                plotter.plot_fourmomenta(
                    filename=filename, **kwargs, weights=weights, mask_dict=mask_dict
                )

            if self.cfg.plotting.jetmomenta:
                filename = os.path.join(path, "jetmomenta.pdf")
                plotter.plot_jetmomenta(
                    filename=filename, **kwargs, weights=weights, mask_dict=mask_dict
                )

            if self.cfg.plotting.preprocessed:
                filename = os.path.join(path, "preprocessed.pdf")
                plotter.plot_preprocessed(
                    filename=filename, **kwargs, weights=weights, mask_dict=mask_dict
                )
        LOGGER.info(f"Plotting done in {time.time() - t0:.2f} seconds")

    def _init_loss(self):
        # loss defined manually within the model
        pass

    def _batch_loss(self, batch):
        batch = batch.to(self.device)
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

    def define_process_specifics(self):
        self.plot_title = "Z+Jet"

        self.obs = {}
        self.obs_ranges = {}

        if "jet" in self.cfg.evaluation.observables:

            def form_jet(constituents, batch_idx, other_batch_idx=None):
                jet = scatter(constituents, batch_idx, dim=0, reduce="sum")
                return jet.cpu().detach()

            self.obs["jet"] = form_jet
            self.obs_ranges["jet"] = {
                "fourmomenta": [[0, 1000], [-400, 400], [-400, 400], [-750, 750]],
                "jetmomenta": [[0, 600], [-torch.pi, torch.pi], [-3, 3], [0, 600]],
                "StandardLogPtPhiEtaLogM2": [[2, 3.5], [-2, 2], [-3, 3], [3, 9]],
            }

        if self.cfg.data.max_constituents > 0 or self.cfg.plotting.n_pt > 0:
            if self.cfg.plotting.n_pt > 0:
                if self.cfg.plotting.n_pt <= self.cfg.data.max_constituents:
                    n_pt = self.cfg.plotting.n_pt
                else:
                    n_pt = self.cfg.data.max_constituents
            else:
                n_pt = self.cfg.data.max_constituents

            for i in range(n_pt):

                def select_pt(i):
                    def ith_pt(constituents, batch_idx, other_batch_idx):
                        idx = []
                        batch_ptr = get_ptr_from_batch(batch_idx)
                        other_batch_ptr = get_ptr_from_batch(other_batch_idx)
                        for n in range(len(batch_ptr) - 1):
                            if i < batch_ptr[n + 1] - batch_ptr[n]:
                                if i < other_batch_ptr[n + 1] - other_batch_ptr[n]:
                                    idx.append(batch_ptr[n] + i)
                        selected_constituents = constituents[idx]
                        return selected_constituents.cpu().detach()

                    return ith_pt

                self.obs[str(i + 1) + " highest p_T"] = select_pt(i)
                self.obs_ranges[str(i + 1) + " highest p_T"] = {
                    "fourmomenta": [
                        [0, 400 - 50 * i],
                        [-200 + 25 * i, 200 - 25 * i],
                        [-200 + 25 * i, 200 - 25 * i],
                        [-400 + 50 * i, 400 - 50 * i],
                    ],
                    "jetmomenta": [
                        [0, 200 - 25 * i],
                        [-torch.pi, torch.pi],
                        [-3, 3],
                        [0, 0.02],
                    ],
                    "StandardLogPtPhiEtaLogM2": [
                        [0.5 - 0.25 * i, 3 - 0.25 * i],
                        [-2, 2],
                        [-3, 3],
                        [-5, -4],
                    ],
                }
