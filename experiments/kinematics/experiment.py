import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

import os, time
from omegaconf import open_dict


from experiments.base_experiment import BaseExperiment
from experiments.dataset import Dataset, load_dataset
from experiments.utils import fix_mass
from experiments.coordinates import fourmomenta_to_jetmomenta
import experiments.kinematics.plotter as plotter
from experiments.logger import LOGGER
import experiments.kinematics.observables as obs
from experiments.kinematics.observables import (
    FASTJET_AVAIL,
    create_partial_jet,
    compute_angles,
    tau1,
    tau2,
    dimass,
    deltaR,
    sd_mass,
    compute_zg,
    jet_mass,
    create_jet_norm,
)


class KinematicsExperiment(BaseExperiment):
    def init_physics(self):

        with open_dict(self.cfg):
            self.cfg.modelname = self.cfg.model._target_.rsplit(".", 1)[-1][:-3]
            self.cfg.cfm.run_dir = self.cfg.run_dir

            if self.cfg.evaluation.load_samples:
                self.cfg.train = False
                self.cfg.evaluation.sample = False
                self.cfg.evaluation.save_samples = False

            if self.cfg.evaluation.overfit:
                self.cfg.evaluation.sample = False
                self.cfg.evaluation.load_samples = False
                self.cfg.training.iterations = 100
                self.cfg.training.validate_every_n_steps = (
                    self.cfg.training.iterations + 1
                )
                self.cfg.data.length = 10000
                self.cfg.plotting.jetscaled = True
                self.cfg.evaluation.n_batches = 1

            max_num_particles, diff, pt_min, masked_dims, load_fn = load_dataset(
                self.cfg.data.dataset
            )

            self.cfg.data.max_num_particles = max_num_particles
            self.cfg.data.pt_min = pt_min
            self.cfg.cfm.masked_dims = masked_dims
            self.load_fn = load_fn

            if self.cfg.data.max_constituents == -1:
                self.cfg.data.max_constituents = self.cfg.data.max_num_particles

            if self.cfg.modelname == "ConditionalTransformer":
                self.cfg.model.net.in_channels = (
                    4 + self.cfg.cfm.embed_t_dim + self.cfg.data.pos_encoding_dim
                )
                self.cfg.model.net_condition.in_channels = (
                    4 + self.cfg.data.pos_encoding_dim
                )
                self.cfg.model.net_condition.out_channels = (
                    self.cfg.model.net.hidden_channels
                )
                if self.cfg.data.add_pid:
                    self.cfg.model.net.in_channels += 6
                    self.cfg.model.net_condition.in_channels += 6
                if self.cfg.cfm.add_jet:
                    self.cfg.model.net.in_channels += 1
                    self.cfg.model.net_condition.in_channels += 1
                if self.cfg.cfm.self_condition_prob > 0.0:
                    self.cfg.model.net.in_channels += 4

            elif self.cfg.modelname == "ConditionalLGATr":
                self.cfg.model.net.in_s_channels = (
                    self.cfg.cfm.embed_t_dim + self.cfg.data.pos_encoding_dim
                )
                self.cfg.model.net_condition.in_s_channels = (
                    self.cfg.data.pos_encoding_dim
                )
                self.cfg.model.net_condition.out_mv_channels = (
                    self.cfg.model.net.hidden_mv_channels
                )
                self.cfg.model.net.condition_mv_channels = (
                    self.cfg.model.net_condition.out_mv_channels
                )
                self.cfg.model.net_condition.out_s_channels = (
                    self.cfg.model.net.hidden_s_channels
                )
                self.cfg.model.net.condition_s_channels = (
                    self.cfg.model.net_condition.out_s_channels
                )
                if self.cfg.data.add_pid:
                    self.cfg.model.net.in_s_channels += 6
                    self.cfg.model.net_condition.in_s_channels += 6
                if self.cfg.cfm.add_jet:
                    self.cfg.model.net.in_s_channels += 1
                    self.cfg.model.net_condition.in_s_channels += 1
                if self.cfg.cfm.self_condition_prob > 0.0:
                    self.cfg.model.net.in_s_channels += 4

            # copy model-specific parameters
            self.cfg.model.odeint = self.cfg.odeint
            self.cfg.model.cfm = self.cfg.cfm

        self.define_process_specifics()

    def init_data(self):
        if self.cfg.evaluation.load_samples:
            # if we load samples, we do not need to initialize the data
            self.train_data = None
            self.val_data = None
            self.test_data = None
            return
        t0 = time.time()
        data_path = os.path.join(self.cfg.data.data_dir, f"{self.cfg.data.dataset}")
        LOGGER.info(f"Creating {self.cfg.data.dataset} from {data_path}")
        self._init_data(data_path)
        LOGGER.info(
            f"Created {self.cfg.data.dataset} with {len(self.train_data)} training events, {len(self.val_data)} validation events, and {len(self.test_data)} test events in {time.time() - t0:.2f} seconds"
        )

    def _init_data(self, data_path):
        t0 = time.time()
        data = self.load_fn(data_path, self.cfg.data, self.dtype)
        det_particles = data["det_particles"]
        det_mults = data["det_mults"]
        det_pids = data["det_pids"]
        det_jets = data["det_jets"]
        gen_particles = data["gen_particles"]
        gen_mults = data["gen_mults"]
        gen_pids = data["gen_pids"]
        gen_jets = data["gen_jets"]
        size = len(gen_particles)

        LOGGER.info(f"Loaded {size} events in {time.time() - t0:.2f} seconds")

        if self.cfg.data.max_constituents > 0:
            det_mults = torch.clamp(det_mults, max=self.cfg.data.max_constituents)
            gen_mults = torch.clamp(gen_mults, max=self.cfg.data.max_constituents)

        split = self.cfg.data.train_val_test
        train_idx, val_idx, test_idx = np.cumsum([int(s * size) for s in split])

        # initialize cfm (might require data)
        self.model.init_physics(
            pt_min=self.cfg.data.pt_min,
            mass=self.cfg.data.mass,
        )
        self.model.init_coordinates()

        # initialize geometry
        self.model.init_geometry()

        train_gen_mask = (
            torch.arange(gen_particles.shape[1])[None, :] < gen_mults[:train_idx, None]
        )
        self.model.coordinates.init_fit(
            gen_particles[:train_idx],
            mask=train_gen_mask,
            jet=torch.repeat_interleave(
                gen_jets[:train_idx], gen_mults[:train_idx], dim=0
            ),
        )

        train_det_mask = (
            torch.arange(det_particles.shape[1])[None, :] < det_mults[:train_idx, None]
        )
        self.model.condition_coordinates.init_fit(
            det_particles[:train_idx],
            mask=train_det_mask,
            jet=torch.repeat_interleave(
                det_jets[:train_idx], det_mults[:train_idx], dim=0
            ),
        )

        det_mask = torch.arange(det_particles.shape[1])[None, :] < det_mults[:, None]
        det_particles[det_mask] = self.model.condition_coordinates.fourmomenta_to_x(
            det_particles[det_mask],
            jet=torch.repeat_interleave(det_jets, det_mults, dim=0),
            ptr=torch.cumsum(
                torch.cat([torch.zeros(1, dtype=torch.int64), det_mults], dim=0), dim=0
            ),
        )

        gen_mask = torch.arange(gen_particles.shape[1])[None, :] < gen_mults[:, None]
        gen_particles[gen_mask] = self.model.coordinates.fourmomenta_to_x(
            gen_particles[gen_mask],
            jet=torch.repeat_interleave(gen_jets, gen_mults, dim=0),
            ptr=torch.cumsum(
                torch.cat([torch.zeros(1, dtype=torch.int64), gen_mults], dim=0), dim=0
            ),
        )

        self.train_data = Dataset(
            self.dtype, pos_encoding_dim=self.cfg.data.pos_encoding_dim
        )
        self.val_data = Dataset(
            self.dtype, pos_encoding_dim=self.cfg.data.pos_encoding_dim
        )
        self.test_data = Dataset(
            self.dtype, pos_encoding_dim=self.cfg.data.pos_encoding_dim
        )

        self.train_data.create_data_list(
            det_particles=det_particles[:train_idx],
            det_pids=det_pids[:train_idx],
            det_mults=det_mults[:train_idx],
            det_jets=det_jets[:train_idx],
            gen_particles=gen_particles[:train_idx],
            gen_pids=gen_pids[:train_idx],
            gen_mults=gen_mults[:train_idx],
            gen_jets=gen_jets[:train_idx],
        )
        self.val_data.create_data_list(
            det_particles=det_particles[train_idx:val_idx],
            det_pids=det_pids[train_idx:val_idx],
            det_mults=det_mults[train_idx:val_idx],
            det_jets=det_jets[train_idx:val_idx],
            gen_particles=gen_particles[train_idx:val_idx],
            gen_pids=gen_pids[train_idx:val_idx],
            gen_mults=gen_mults[train_idx:val_idx],
            gen_jets=gen_jets[train_idx:val_idx],
        )
        self.test_data.create_data_list(
            det_particles=det_particles[val_idx:test_idx],
            det_pids=det_pids[val_idx:test_idx],
            det_mults=det_mults[val_idx:test_idx],
            det_jets=det_jets[val_idx:test_idx],
            gen_particles=gen_particles[val_idx:test_idx],
            gen_pids=gen_pids[val_idx:test_idx],
            gen_mults=gen_mults[val_idx:test_idx],
            gen_jets=gen_jets[val_idx:test_idx],
        )

    def _init_dataloader(self):
        if self.cfg.evaluation.load_samples:
            self.train_loader = None
            self.val_loader = None
            self.test_loader = None
            return
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
        self.model.eval()
        if self.cfg.evaluation.sample:
            t0 = time.time()
            self._sample_events(loaders["test"])
            loaders["gen"] = self.sample_loader
            dt = time.time() - t0
            LOGGER.info(f"Finished sampling after {dt/60:.2f}min")
        elif self.cfg.evaluation.load_samples:
            self._load_samples()
            loaders["gen"] = self.sample_loader
        elif self.cfg.evaluation.overfit:
            t0 = time.time()
            self._sample_events(loaders["train"])
            loaders["gen"] = self.sample_loader
            dt = time.time() - t0
            LOGGER.info(f"Finished sampling after {dt/60:.2f}min")
        else:
            LOGGER.info("Skip sampling")

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

            sample_batch, base = self.model.sample(
                batch,
                self.device,
                self.dtype,
            )

            # Compute jets for sample_batch
            sample_gen_jets = torch.repeat_interleave(
                sample_batch.jet_gen, sample_batch.x_gen_ptr.diff(), dim=0
            )
            sample_det_jets = torch.repeat_interleave(
                sample_batch.jet_det, sample_batch.x_det_ptr.diff(), dim=0
            )

            # Compute jets for original batch
            batch_gen_jets = torch.repeat_interleave(
                batch.jet_gen, batch.x_gen_ptr.diff(), dim=0
            )
            batch_det_jets = torch.repeat_interleave(
                batch.jet_det, batch.x_det_ptr.diff(), dim=0
            )

            sample_batch.x_det = self.model.condition_coordinates.x_to_fourmomenta(
                sample_batch.x_det, jet=sample_det_jets, ptr=sample_batch.x_det_ptr
            )
            sample_batch.x_gen = self.model.coordinates.x_to_fourmomenta(
                sample_batch.x_gen, jet=sample_gen_jets, ptr=sample_batch.x_gen_ptr
            )

            batch.x_det = self.model.condition_coordinates.x_to_fourmomenta(
                batch.x_det, jet=batch_det_jets, ptr=batch.x_det_ptr
            )
            batch.x_gen = self.model.coordinates.x_to_fourmomenta(
                batch.x_gen, jet=batch_gen_jets, ptr=batch.x_gen_ptr
            )

            samples.extend(sample_batch.detach().to_data_list())
            targets.extend(batch.detach().to_data_list())

        self.data_raw["samples"] = Batch.from_data_list(
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

        if self.cfg.evaluation.save_samples:
            path = os.path.join(self.cfg.run_dir, f"samples_{self.cfg.run_idx}")
            os.makedirs(os.path.join(path), exist_ok=True)
            LOGGER.info(f"Saving samples in {path}")
            t0 = time.time()
            torch.save(self.data_raw["samples"], os.path.join(path, "samples.pt"))
            torch.save(self.data_raw["truth"], os.path.join(path, "truth.pt"))
            LOGGER.info(f"Saved samples in {time.time() - t0:.2f}s")

    def _load_samples(self):
        path = os.path.join(self.cfg.run_dir, f"samples_{self.cfg.warm_start_idx}")
        LOGGER.info(f"Loading samples from {path}")
        t0 = time.time()
        self.data_raw = {}
        self.data_raw["samples"] = torch.load(
            os.path.join(path, "samples.pt"),
            weights_only=False,
            map_location=self.device,
        )
        self.data_raw["truth"] = torch.load(
            os.path.join(path, "truth.pt"), weights_only=False, map_location=self.device
        )
        LOGGER.info(f"Loaded samples with {len(self.data_raw['samples'])} events")

        samples = self.data_raw["samples"].to_data_list()
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

        LOGGER.info(f"Loaded samples in {time.time() - t0:.2f}s")

    def plot(self):
        path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(os.path.join(path), exist_ok=True)
        LOGGER.info(f"Creating plots in {path}")
        t0 = time.time()

        if self.cfg.modelname == "ConditionalTransformer":
            model_label = "CondTr"
        elif self.cfg.modelname == "ConditionalLGATr":
            model_label = "CondLGATr"
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
            self.cfg.evaluation.sample
            or self.cfg.evaluation.load_samples
            or self.cfg.evaluation.overfit
        ):
            if self.cfg.plotting.fourmomenta:
                LOGGER.info("Plotting fourmomenta")
                filename = os.path.join(path, "fourmomenta.pdf")
                plotter.plot_fourmomenta(
                    filename=filename, **kwargs, weights=weights, mask_dict=mask_dict
                )

            if self.cfg.plotting.jetmomenta:
                LOGGER.info("Plotting jetmomenta")
                filename = os.path.join(path, "jetmomenta.pdf")
                plotter.plot_jetmomenta(
                    filename=filename, **kwargs, weights=weights, mask_dict=mask_dict
                )

            if self.cfg.plotting.preprocessed:
                filename = os.path.join(path, "preprocessed.pdf")
                plotter.plot_preprocessed(
                    filename=filename, **kwargs, weights=weights, mask_dict=mask_dict
                )
            if self.cfg.plotting.jetscaled:
                LOGGER.info("Plotting jetscaled")
                filename = os.path.join(path, "jetscaled.pdf")
                plotter.plot_jetscaled(
                    filename=filename, **kwargs, weights=weights, mask_dict=mask_dict
                )
            if len(self.obs.keys()) > 0:
                filename = os.path.join(path, "observables.pdf")
                plotter.plot_observables(
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
        assert torch.isfinite(loss).all()
        metrics = {"mse": mse}
        for k in range(4):
            metrics[f"mse_{k}"] = component_loss[k].cpu().item()
        return loss, metrics

    def _init_metrics(self):
        metrics = {"mse": []}
        for k in range(4):
            metrics[f"mse_{k}"] = []
        return metrics

    def define_process_specifics(self):
        if self.cfg.data.max_constituents >= self.cfg.data.max_num_particles:
            n_const = "All"
        else:
            n_const = str(self.cfg.data.max_constituents)
        # self.plot_title = n_const + " constituents"
        self.plot_title = None

        self.obs_coords = {}

        if "jet" in self.cfg.plotting.observables:

            self.obs_coords[r"\text{jet}"] = create_partial_jet(0.0, 1.0)

        if "slices" in self.cfg.plotting.observables:
            self.obs_coords[r"1-5"] = create_partial_jet(0, 5)
            self.obs_coords[r"6-10"] = create_partial_jet(5, 10)
            self.obs_coords[r"11-15"] = create_partial_jet(10, 15)
            self.obs_coords[r"16-20"] = create_partial_jet(15, 20)
            self.obs_coords[r"1-10"] = create_partial_jet(0, 10)
            self.obs_coords[r"11-20"] = create_partial_jet(10, 20)

        if self.cfg.plotting.n_pt > 0:
            if self.cfg.data.max_constituents == -1:
                n_pt = self.cfg.plotting.n_pt
            else:
                n_pt = min(self.cfg.data.max_constituents, self.cfg.plotting.n_pt)

            for i in range(n_pt):

                self.obs_coords[str(i + 1) + r"\text{ highest } p_T"] = (
                    create_partial_jet(start=i, end=i + 1)
                )

        self.obs = {}

        if "angle" in self.cfg.plotting.observables:
            self.obs[r"\Delta \phi_{4,5}"] = compute_angles(3, 4, 4, 5, "phi")
            self.obs[r"\Delta \eta_{4,5}"] = compute_angles(3, 4, 4, 5, "eta")
            self.obs[r"\Delta R_{4,5}"] = compute_angles(3, 4, 4, 5, "R")
            self.obs[r"\Delta \phi_{4,6}"] = compute_angles(3, 4, 5, 6, "phi")
            self.obs[r"\Delta \eta_{4,6}"] = compute_angles(3, 4, 5, 6, "eta")
            self.obs[r"\Delta R_{4,6}"] = compute_angles(3, 4, 5, 6, "R")
            self.obs[r"\Delta \phi_{5,6}"] = compute_angles(4, 5, 5, 6, "phi")
            self.obs[r"\Delta \eta_{5,6}"] = compute_angles(4, 5, 5, 6, "eta")
            self.obs[r"\Delta R_{5,6}"] = compute_angles(4, 5, 5, 6, "R")

        if "dimass" in self.cfg.plotting.observables:
            # dijet mass (only for CMS dataset with 3 jets)

            for i in range(3):
                for j in range(i + 1, 3):
                    self.obs[r"M_{" + str(i + 1) + str(j + 1) + "}"] = dimass(i, j)

        if "deltaR" in self.cfg.plotting.observables:

            for i in range(3):
                for j in range(i + 1, 3):
                    self.obs[r"\Delta R_{" + str(i + 1) + str(j + 1) + "}"] = deltaR(
                        i, j
                    )

        if self.cfg.data.dataset == "zplusjet":
            obs.R0 = 0.4
            obs.R0SoftDrop = 0.8
        elif self.cfg.data.dataset == "cms":
            obs.R0 = 1.2
            obs.R0SoftDrop = 1.2
        elif self.cfg.data.dataset == "ttbar":
            obs.R0 = 1.2
            obs.R0SoftDrop = 1.2

        if "tau1" in self.cfg.plotting.observables and FASTJET_AVAIL:
            self.obs[r"\tau_1"] = tau1
        if "tau2" in self.cfg.plotting.observables and FASTJET_AVAIL:
            self.obs[r"\tau_2"] = tau2
        if "tau21" in self.cfg.plotting.observables and FASTJET_AVAIL:
            self.obs[r"\tau_{21}"] = (
                lambda constituents, batch_idx, other_batch_idx: torch.where(
                    tau1(constituents, batch_idx, other_batch_idx) != 0,
                    tau2(constituents, batch_idx, other_batch_idx)
                    / tau1(constituents, batch_idx, other_batch_idx),
                    torch.tensor(0.0),
                )
            )
        if "sd_mass" in self.cfg.plotting.observables and FASTJET_AVAIL:

            self.obs[r"\log \rho"] = sd_mass

        if "momentum_fraction" in self.cfg.plotting.observables and FASTJET_AVAIL:

            self.obs[r"z_g"] = compute_zg

        if "jet_mass" in self.cfg.plotting.observables:

            self.obs[r"M_{jet}"] = jet_mass

        if "norm" in self.cfg.plotting.observables:
            self.obs[
                r"\sqrt{E_{\text{jet}}^2 + p_{x,\text{jet}}^2 + p_{y,\text{jet}}^2 + p_{z,\text{jet}}^2}"
            ] = create_jet_norm()
            self.obs[
                r"\sqrt{p_{x,\text{jet}}^2 + p_{y,\text{jet}}^2 + p_{z,\text{jet}}^2}"
            ] = create_jet_norm([1, 2, 3])
            self.obs[r"p_{T,\text{jet}}"] = create_jet_norm([1, 2])
            self.obs[r"M_{\text{jet}}"] = create_jet_norm([0], [1, 2, 3])
            self.obs[
                r"\sqrt{E_{\text{jet}}^2 - p_{x,\text{jet}}^2 - p_{y,\text{jet}}^2}"
            ] = create_jet_norm([0], [1, 2])
            self.obs[r"\sqrt{E_{\text{jet}}^2 - p_{z,\text{jet}}^2}"] = create_jet_norm(
                [0], [3]
            )
