import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from torch_geometric.data import Batch

import os, time
from omegaconf import open_dict
import energyflow

from gatr.interface import embed_spurions, extract_vector
from experiments.base_experiment import BaseExperiment
from experiments.unfolding.dataset import ZplusJetDataset
from experiments.unfolding.utils import (
    ensure_angle,
    pid_encoding,
    get_ptr_from_batch,
    jetmomenta_to_fourmomenta,
)
import experiments.unfolding.plotter as plotter
from experiments.logger import LOGGER


class UnfoldingExperiment(BaseExperiment):
    def init_physics(self):

        self.define_process_specifics()

        with open_dict(self.cfg):
            self.cfg.modelname = self.cfg.model._target_.rsplit(".", 1)[-1][:-3]
            # dynamically set channel dimensions
            if self.cfg.modelname == "ConditionalGATr":
                self.cfg.data.embed_det_in_GA = True
                self.cfg.model.net.in_s_channels = self.cfg.cfm.embed_t_dim
                self.cfg.model.net_condition.in_s_channels = 0
                self.cfg.model.net_condition.out_mv_channels = (
                    self.cfg.model.net.hidden_mv_channels
                )
                self.cfg.model.net_condition.out_s_channels = (
                    self.cfg.model.net.hidden_s_channels
                )
                if self.cfg.data.pid_encoding:
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
                self.cfg.data.embed_det_in_GA = False
                self.cfg.model.net.in_channels = 4 + self.cfg.cfm.embed_t_dim
                self.cfg.model.net_condition.in_channels = 4
                self.cfg.model.net_condition.out_channels = (
                    self.cfg.model.net.hidden_channels
                )
                if self.cfg.data.pid_encoding:
                    self.cfg.model.net.in_channels += 6
                    self.cfg.model.net_condition.in_channels += 6
            elif self.cfg.modelname == "ConditionalMLP":
                self.cfg.data.embed_det_in_GA = False
                self.cfg.model.net.in_shape = 4 + self.cfg.cfm.embed_t_dim
                self.cfg.model.net.out_shape = 4
                if self.cfg.data.pid_encoding:
                    self.cfg.model.net.in_channels += 6
            elif self.cfg.modelname == "ConditionalAutoregressiveTransformer":
                self.cfg.data.embed_det_in_GA = False
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
        LOGGER.info(f"Creating {self.cfg.data.dataset} from {data_path}")
        if self.cfg.data.dataset == "zplusjet":
            self._init_data(ZplusJetDataset, data_path)
        else:
            self._init_data("jets", data_path)
        LOGGER.info(
            f"Created {self.cfg.data.dataset} with {len(self.train_data)} training events, {len(self.val_data)} validation events, and {len(self.test_data)} test events in {time.time() - t0:.2f} seconds"
        )

    def _init_data(self, Dataset, data_path):
        t0 = time.time()
        if Dataset == ZplusJetDataset:
            data = energyflow.zjets_delphes.load(
                "Herwig",
                num_data=self.cfg.data.num_data,
                pad=True,
                cache_dir=data_path,
                include_keys=["particles", "mults", "jets"],
            )
            size = len(data["sim_particles"])

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

            det_particles = jetmomenta_to_fourmomenta(det_particles)
            gen_particles = jetmomenta_to_fourmomenta(gen_particles)

        elif Dataset == "jets":
            Dataset = ZplusJetDataset
            gen_particles = (
                torch.from_numpy(
                    np.load(os.path.join(data_path, "gen_1725_delphes.npy"))
                )
                .to(self.dtype)
                .reshape(-1, 3, 4)
            )
            det_particles = (
                torch.from_numpy(
                    np.load(os.path.join(data_path, "rec_1725_delphes.npy"))
                )
                .to(self.dtype)
                .reshape(-1, 3, 4)
            )
            size = len(gen_particles)
            gen_mults = torch.ones(gen_particles.shape[0], dtype=torch.int)
            det_mults = torch.ones(det_particles.shape[0], dtype=torch.int)
            gen_pids = torch.empty(*gen_particles.shape[:-1], 0, dtype=self.dtype)
            det_pids = torch.empty(*det_particles.shape[:-1], 0, dtype=self.dtype)

        LOGGER.info(f"Loaded {size} events in {time.time() - t0:.2f} seconds")

        gen_particles /= self.cfg.data.units
        det_particles /= self.cfg.data.units

        if self.cfg.data.max_constituents > 0:
            if self.cfg.data.det_mult == 1:
                det_mults = torch.clamp(det_mults, max=self.cfg.data.max_constituents)
            elif self.cfg.data.det_mult == 2:
                det_mults = torch.clamp(
                    det_mults, max=2 * self.cfg.data.max_constituents
                )
            elif self.cfg.data.det_mult == -1:
                pass
            gen_mults = torch.clamp(gen_mults, max=self.cfg.data.max_constituents)

        split = self.cfg.data.train_test_val
        train_idx, val_idx, test_idx = np.cumsum([int(s * size) for s in split])

        # initialize cfm (might require data)
        self.model.init_physics(
            units=self.cfg.data.units,
            pt_min=0,
            base_type=self.cfg.data.base_type,
            onshell_mass=self.cfg.data.mass,
        )
        self.model.init_distribution()
        self.model.init_coordinates()

        gen_mask = (
            torch.arange(gen_particles.shape[1])[None, :] < gen_mults[:train_idx, None]
        )
        fit_gen_data = gen_particles[:train_idx][gen_mask]
        if self.cfg.cfm.coordinates[:2] == 'St':
            fit_gen_data = fit_gen_data.to(
                self.device, self.model.coordinates.transforms[-1].mean.dtype
            )
        self.model.coordinates.init_fit(fit_gen_data)
        self.model.distribution.coordinates.init_fit(fit_gen_data)

        det_mask = (
            torch.arange(det_particles.shape[1])[None, :] < det_mults[:train_idx, None]
        )
        fit_det_data = det_particles[:train_idx][det_mask]
        if self.cfg.cfm.coordinates[:2] == 'St':
            fit_det_data = fit_det_data.to(
                self.device, self.model.coordinates.transforms[-1].mean.dtype
            )
        self.model.condition_coordinates.init_fit(fit_det_data)

        self.model.init_geometry()

        if self.cfg.data.embed_det_in_GA and self.cfg.data.add_spurions:
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
            self.dtype,
            self.cfg.data.embed_det_in_GA,
            self.spurions,
        )
        self.val_data = Dataset(
            self.dtype,
            self.cfg.data.embed_det_in_GA,
            self.spurions,
        )
        self.test_data = Dataset(
            self.dtype,
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

    @torch.no_grad()
    def evaluate(self):
        # EMA-evaluation not implemented
        loaders = {
            "train": self.train_loader,
            "test": self.test_loader,
            "val": self.val_loader,
        }
        if self.cfg.evaluation.sample:
            t0 = time.time()
            self._sample_events(loaders["test"])
            loaders["gen"] = self.sample_loader
            dt = time.time() - t0
            LOGGER.info(f"Finished sampling after {dt/60:.2f}min")
        else:
            LOGGER.info("Skip sampling")

        if self.cfg.evaluation.classifier:
            self.classifier = self._evaluate_classifier_metric()

    def _evaluate_classifier_metric(self):
        raise NotImplementedError

    def _evaluate_log_prob_single(self, loader, title):
        raise NotImplementedError

    def _sample_events(self, loader):
        samples = []
        targets = []
        self.data_raw = {}
        it = iter(loader)
        n_batches = self.cfg.evaluation.n_batches
        if n_batches > len(loader):
            LOGGER.warning(
                f"Requested {n_batches} batches for sampling, but only {len(loader)} batches available in test dataset."
            )
            n_batches = len(loader)
        elif n_batches == -1:
            n_batches = len(loader)
        LOGGER.info(f"Sampling {n_batches} batches for evaluation")
        for i in range(n_batches):
            batch = next(it).to(self.device)

            sample_batch = self.model.sample(
                batch,
                self.device,
                self.dtype,
            )

            samples.extend(sample_batch.to_data_list())
            targets.extend(batch.to_data_list())

        if self.cfg.data.embed_det_in_GA:
            if self.spurions is not None and len(self.spurions) > 0:
                for data in samples:
                    data.x_det = extract_vector(
                        data.x_det[: -len(self.spurions)]
                    ).squeeze(-2)
                for data in targets:
                    data.x_det = extract_vector(
                        data.x_det[: -len(self.spurions)]
                    ).squeeze(-2)
            else:
                for data in samples:
                    data.x_det = extract_vector(data.x_det).squeeze(-2)
                for data in targets:
                    data.x_det = extract_vector(data.x_det).squeeze(-2)

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
        elif self.cfg.modelname == "ConditionalMLP":
            model_label = "MLP"
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
        if self.cfg.data.max_constituents == -1:
            n_const = "All"
        else:
            n_const = str(self.cfg.data.max_constituents)
        self.plot_title = n_const + " constituents"

        self.obs = {}
        self.obs_ranges = {}

        if "jet" in self.cfg.plotting.observables:

            def form_jet(constituents, batch_idx, other_batch_idx):
                jet = scatter(constituents, batch_idx, dim=0, reduce="sum")
                return jet * self.cfg.data.units

            self.obs[r"\text{ jet }"] = form_jet
            self.obs_ranges[r"\text{ jet }"] = {
                "fourmomenta": [[0, 1000], [-400, 400], [-400, 400], [-750, 750]],
                "jetmomenta": [[0, 600], [-torch.pi, torch.pi], [-3, 3], [0, 600]],
                "StandardLogPtPhiEtaLogM2": [[0, 4], [-2, 2], [-3, 3], [0, 10]],
            }

        if self.cfg.plotting.n_pt > 0:
            if self.cfg.data.max_constituents == -1:
                n_pt = self.cfg.plotting.n_pt
            else:
                n_pt = min(self.cfg.data.max_constituents, self.cfg.plotting.n_pt)

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
                        return selected_constituents * self.cfg.data.units

                    return ith_pt

                self.obs[str(i + 1) + r"\text{ highest } p_T"] = select_pt(i)
                self.obs_ranges[str(i + 1) + r"\text{ highest } p_T"] = {
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
