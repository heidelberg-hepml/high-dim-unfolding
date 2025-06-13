import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from torch_geometric.data import Batch

import os, time
from omegaconf import open_dict

from fastjet_contribs import (
    compute_nsubjettiness,
    apply_soft_drop,
)

from gatr.interface import embed_spurions, extract_vector
from experiments.base_experiment import BaseExperiment
from experiments.dataset import (
    Dataset,
    load_cms,
    load_zplusjet,
    load_ttbar,
)
from experiments.utils import (
    get_ptr_from_batch,
    fourmomenta_to_jetmomenta,
    ensure_angle,
)
import experiments.unfolding.plotter as plotter
from experiments.unfolding.plots import plot_kinematics
from experiments.logger import LOGGER


class UnfoldingExperiment(BaseExperiment):
    def init_physics(self):

        with open_dict(self.cfg):
            self.cfg.modelname = self.cfg.model._target_.rsplit(".", 1)[-1][:-3]

            if self.cfg.data.dataset == "cms":
                self.cfg.data.max_num_particles = 3
                self.cfg.data.pt_min = 30.0
                self.cfg.data.units = 10.0
                self.cfg.cfm.masked_dims = []

            if self.cfg.data.dataset == "zplusjet":
                self.cfg.data.max_num_particles = 152
                self.cfg.data.pt_min = 0.0
                self.cfg.data.units = 10.0
                self.cfg.cfm.masked_dims = [3]

            if self.cfg.data.dataset == "ttbar":
                self.cfg.data.max_num_particles = 238
                self.cfg.data.pt_min = 0.0
                self.cfg.data.units = 10.0
                self.cfg.cfm.masked_dims = [3]

            if self.cfg.data.max_constituents == -1:
                self.cfg.data.max_constituents = self.cfg.data.max_num_particles

            if self.cfg.data.add_jet:
                self.cfg.data.max_constituents += 1
                self.cfg.cfm.mask_jets = True

            if self.cfg.modelname == "ConditionalGATr":
                self.cfg.data.transform = False
                self.cfg.data.embed_det_in_GA = True
                self.cfg.data.add_spurions = True

            if self.cfg.data.add_spurions:
                self.spurions = embed_spurions(
                    self.cfg.data.beam_reference,
                    self.cfg.data.add_time_reference,
                    self.cfg.data.two_beams,
                    self.cfg.data.add_xzplane,
                    self.cfg.data.add_yzplane,
                )
                self.cfg.data.num_spurions = self.spurions.size(-2)
            else:
                self.spurions = None
                self.cfg.data.num_spurions = 0

            if self.cfg.modelname == "ConditionalMLP":
                self.cfg.data.embed_det_in_GA = False
                self.cfg.model.net.in_shape = 4 + self.cfg.cfm.embed_t_dim
                self.cfg.model.net.out_shape = 4
                if self.cfg.data.add_pid:
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
            elif self.cfg.modelname == "ConditionalTransformer":
                self.cfg.data.embed_det_in_GA = False
                self.cfg.model.net.in_channels = 4 + self.cfg.cfm.embed_t_dim
                self.cfg.model.net_condition.in_channels = 4
                self.cfg.model.net_condition.out_channels = (
                    self.cfg.model.net.hidden_channels
                )
                if self.cfg.data.add_pid:
                    self.cfg.model.net.in_channels += 6
                    self.cfg.model.net_condition.in_channels += 6
                if self.cfg.model.net.pos_encoding_type == "absolute":
                    self.cfg.model.net.pos_encoding_base = (
                        self.cfg.data.max_constituents
                    )

                if self.cfg.model.net_condition.pos_encoding_type == "absolute":
                    self.cfg.model.net_condition.pos_encoding_base = (
                        self.cfg.data.max_constituents
                    )

            elif self.cfg.modelname == "ConditionalGATr":
                self.cfg.cfm.condition_coordinates = "Fourmomenta"
                self.cfg.model.net.in_s_channels = self.cfg.cfm.embed_t_dim
                self.cfg.model.net_condition.in_s_channels = self.cfg.cfm.embed_t_dim
                self.cfg.model.net_condition.out_mv_channels = (
                    self.cfg.model.net.hidden_mv_channels
                )
                self.cfg.model.net_condition.out_s_channels = (
                    self.cfg.model.net.hidden_s_channels
                )
                if self.cfg.data.add_pid:
                    self.cfg.model.net.in_s_channels += 6
                    self.cfg.model.net_condition.in_s_channels += 6
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

                if self.cfg.model.net.attention.pos_encoding_type == "absolute":
                    self.cfg.model.net.attention.pos_encoding_base = (
                        self.cfg.data.max_constituents + self.cfg.data.num_spurions
                    )

                if self.cfg.model.net.crossattention.pos_encoding_type == "absolute":
                    self.cfg.model.net.crossattention.pos_encoding_base = (
                        self.cfg.data.max_constituents + self.cfg.data.num_spurions
                    )

                if (
                    self.cfg.model.net_condition.attention.pos_encoding_type
                    == "absolute"
                ):
                    self.cfg.model.net_condition.attention.pos_encoding_base = (
                        self.cfg.data.max_constituents + self.cfg.data.num_spurions
                    )

            # copy model-specific parameters
            self.cfg.model.odeint = self.cfg.odeint
            self.cfg.model.cfm = self.cfg.cfm

        self.define_process_specifics()

    def init_data(self):
        if self.cfg.evaluation.load_samples:
            LOGGER.info("Not loading data, using saved samples")
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
        if self.cfg.data.dataset == "zplusjet":
            data = load_zplusjet(data_path, self.cfg.data, self.dtype)
        elif self.cfg.data.dataset == "cms":
            data = load_cms(data_path, self.cfg.data, self.dtype)
        elif self.cfg.data.dataset == "ttbar":
            data = load_ttbar(data_path, self.cfg.data, self.dtype)
        else:
            raise ValueError(f"Unknown dataset {self.cfg.data.dataset}")
        det_particles = data["det_particles"]
        det_mults = data["det_mults"]
        det_pids = data["det_pids"]
        gen_particles = data["gen_particles"]
        gen_mults = data["gen_mults"]
        gen_pids = data["gen_pids"]
        size = len(gen_particles)

        LOGGER.info(f"Loaded {size} events in {time.time() - t0:.2f} seconds")

        if self.cfg.data.add_jet:
            # add det jet as first particle to condition
            det_jets = det_particles.sum(dim=1, keepdim=True)
            det_particles = torch.cat([det_jets, det_particles], dim=1)
            det_pids = torch.cat([torch.zeros_like(det_pids[:, :1]), det_pids], dim=1)
            det_mults += 1

            # add gen jet as first particle to condition
            gen_jets = gen_particles.sum(dim=1, keepdim=True)
            gen_particles = torch.cat([gen_jets, gen_particles], dim=1)
            gen_pids = torch.cat([torch.zeros_like(gen_pids[:, :1]), gen_pids], dim=1)
            gen_mults += 1

        gen_particles /= self.cfg.data.units
        det_particles /= self.cfg.data.units

        if self.cfg.data.max_constituents > 0:
            det_mults = torch.clamp(det_mults, max=self.cfg.data.max_constituents)
            gen_mults = torch.clamp(gen_mults, max=self.cfg.data.max_constituents)

        split = self.cfg.data.train_val_test
        train_idx, val_idx, test_idx = np.cumsum([int(s * size) for s in split])

        # initialize cfm (might require data)
        self.model.init_physics(
            units=self.cfg.data.units,
            pt_min=self.cfg.data.pt_min,
            mass=self.cfg.data.mass,
        )
        self.model.init_distribution()
        self.model.init_coordinates()

        # initialize geometry
        self.model.init_geometry()

        train_gen_mask = (
            torch.arange(gen_particles.shape[1])[None, :] < gen_mults[:train_idx, None]
        )
        self.model.coordinates.init_fit(
            gen_particles[:train_idx][train_gen_mask],
            batch_ptr=torch.cumsum(
                torch.cat([torch.zeros(1), gen_mults[:train_idx]]), dim=0, dtype=int
            ),
        )

        train_det_mask = (
            torch.arange(det_particles.shape[1])[None, :] < det_mults[:train_idx, None]
        )
        self.model.condition_coordinates.init_fit(
            det_particles[:train_idx][train_det_mask],
            batch_ptr=torch.cumsum(
                torch.cat([torch.zeros(1), det_mults[:train_idx]]), dim=0, dtype=int
            ),
        )

        if self.cfg.data.transform:
            det_mask = (
                torch.arange(det_particles.shape[1])[None, :] < det_mults[:, None]
            )
            det_particles[det_mask] = self.model.condition_coordinates.fourmomenta_to_x(
                det_particles[det_mask],
                batch_ptr=torch.cumsum(
                    torch.cat([torch.zeros(1), det_mults]), dim=0, dtype=int
                ),
            )

            gen_mask = (
                torch.arange(gen_particles.shape[1])[None, :] < gen_mults[:, None]
            )
            gen_particles[gen_mask] = self.model.coordinates.fourmomenta_to_x(
                gen_particles[gen_mask],
                batch_ptr=torch.cumsum(
                    torch.cat([torch.zeros(1), gen_mults]), dim=0, dtype=int
                ),
            )
            if self.cfg.data.add_jet:
                det_mask[:, 0] = False
                gen_mask[:, 0] = False

            plot_kinematics(
                self.cfg.run_dir, det_particles[det_mask], gen_particles[gen_mask]
            )

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
            self.cfg.data.add_jet,
            self.cfg.data.embed_det_in_GA,
            self.spurions,
        )
        self.val_data = Dataset(
            self.dtype,
            self.cfg.data.add_jet,
            self.cfg.data.embed_det_in_GA,
            self.spurions,
        )
        self.test_data = Dataset(
            self.dtype,
            self.cfg.data.add_jet,
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
        if self.cfg.evaluation.sample:
            t0 = time.time()
            self._sample_events(loaders["test"])
            loaders["gen"] = self.sample_loader
            dt = time.time() - t0
            LOGGER.info(f"Finished sampling after {dt/60:.2f}min")
        elif self.cfg.evaluation.load_samples:
            self._load_samples()
            loaders["gen"] = self.sample_loader
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

            sample_batch, base = self.model.sample(
                batch,
                self.device,
                self.dtype,
            )

            if i == 0:
                if self.cfg.data.add_jet:
                    det_mask = ~torch.isin(
                        torch.arange(batch.x_det.size(0), device=self.device),
                        batch.x_det_ptr[:-1],
                    )
                    gen_mask = ~torch.isin(
                        torch.arange(batch.x_gen.size(0), device=self.device),
                        batch.x_gen_ptr[:-1],
                    )
                    plot_kinematics(
                        self.cfg.run_dir,
                        base[gen_mask].cpu(),
                        batch.x_gen[gen_mask].cpu(),
                        sample_batch.x_gen[gen_mask].cpu(),
                        filename="post_kinematics.pdf",
                    )
                else:
                    plot_kinematics(
                        self.cfg.run_dir,
                        base.cpu(),
                        batch.x_gen.cpu(),
                        sample_batch.x_gen.cpu(),
                        filename="post_kinematics.pdf",
                    )

            if self.cfg.data.transform:
                sample_batch.x_det = (
                    self.model.condition_coordinates.x_to_fourmomenta(
                        sample_batch.x_det, batch_ptr=sample_batch.x_det_ptr
                    )
                    * self.cfg.data.units
                )
                sample_batch.x_gen = (
                    self.model.coordinates.x_to_fourmomenta(
                        sample_batch.x_gen, batch_ptr=sample_batch.x_gen_ptr
                    )
                    * self.cfg.data.units
                )
                batch.x_det = (
                    self.model.condition_coordinates.x_to_fourmomenta(
                        batch.x_det, batch_ptr=batch.x_det_ptr
                    )
                    * self.cfg.data.units
                )
                batch.x_gen = (
                    self.model.coordinates.x_to_fourmomenta(
                        batch.x_gen, batch_ptr=batch.x_gen_ptr
                    )
                    * self.cfg.data.units
                )

            if i == 0:
                if self.cfg.data.add_jet:
                    det_mask = ~torch.isin(
                        torch.arange(batch.x_det.size(0), device=self.device),
                        batch.x_det_ptr[:-1],
                    )
                    gen_mask = ~torch.isin(
                        torch.arange(batch.x_gen.size(0), device=self.device),
                        batch.x_gen_ptr[:-1],
                    )
                    plot_kinematics(
                        self.cfg.run_dir,
                        fourmomenta_to_jetmomenta(batch.x_det[det_mask]).cpu(),
                        fourmomenta_to_jetmomenta(batch.x_gen[gen_mask]).cpu(),
                        fourmomenta_to_jetmomenta(sample_batch.x_gen[gen_mask]).cpu(),
                        filename="post_jetmomenta.pdf",
                    )
                else:
                    plot_kinematics(
                        self.cfg.run_dir,
                        fourmomenta_to_jetmomenta(batch.x_det).cpu(),
                        fourmomenta_to_jetmomenta(batch.x_gen).cpu(),
                        fourmomenta_to_jetmomenta(sample_batch.x_gen).cpu(),
                        filename="post_jetmomenta.pdf",
                    )

            samples.extend(sample_batch.detach().to_data_list())
            targets.extend(batch.detach().to_data_list())

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
        if self.cfg.data.add_jet:
            for data in samples:
                data.x_gen = data.x_gen[1:]
                data.x_det = data.x_det[1:]
            for data in targets:
                data.x_gen = data.x_gen[1:]
                data.x_det = data.x_det[1:]

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
            os.path.join(path, "samples.pt"), weights_only=False
        )
        self.data_raw["truth"] = torch.load(
            os.path.join(path, "truth.pt"), weights_only=False
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

        if self.cfg.evaluation.sample or self.cfg.evaluation.load_samples:
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
        if self.cfg.data.max_constituents == -1:
            n_const = "All"
        else:
            n_const = str(self.cfg.data.max_constituents)
        self.plot_title = n_const + " constituents"

        self.obs_coords = {}

        if "jet" in self.cfg.plotting.observables:

            def form_jet(constituents, batch_idx, other_batch_idx):
                jet = scatter(constituents, batch_idx, dim=0, reduce="sum")
                return jet

            self.obs_coords[r"\text{ jet }"] = form_jet

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
                        return selected_constituents

                    return ith_pt

                self.obs_coords[str(i + 1) + r"\text{ highest } p_T"] = select_pt(i)

        self.obs = {}

        if "dimass" in self.cfg.plotting.observables:
            # dijet mass (only for CMS dataset with 3 jets)
            def dimass(i, j):
                def dimass_ij(constituents, batch_idx, other_batch_idx):
                    batch_ptr = get_ptr_from_batch(batch_idx)
                    dimass = []
                    for n in range(len(batch_ptr) - 1):
                        if batch_ptr[n + 1] - batch_ptr[n] == 3:
                            dijet = (
                                constituents[batch_ptr[n] + i]
                                + constituents[batch_ptr[n] + j]
                            )
                            dimass.append(
                                torch.sqrt(dijet[0] ** 2 - (dijet[1:] ** 2).sum(dim=-1))
                            )
                    return torch.stack(dimass)

                return dimass_ij

            for i in range(3):
                for j in range(i + 1, 3):
                    self.obs[r"M_{" + str(i + 1) + str(j + 1) + "}"] = dimass(i, j)

        if "deltaR" in self.cfg.plotting.observables:

            def deltaR(i, j):
                def deltaR_ij(constituents, batch_idx, other_batch_idx):
                    batch_ptr = get_ptr_from_batch(batch_idx)
                    deltaR = []
                    for n in range(len(batch_ptr) - 1):
                        if batch_ptr[n + 1] - batch_ptr[n] == 3:
                            jet_i = fourmomenta_to_jetmomenta(
                                constituents[batch_ptr[n] + i]
                            )
                            jet_j = fourmomenta_to_jetmomenta(
                                constituents[batch_ptr[n] + j]
                            )
                            dR2 = (
                                ensure_angle(jet_i[..., 1] - jet_j[..., 1]) ** 2
                                + (jet_i[..., 2] - jet_j[..., 2]) ** 2
                            )
                            deltaR.append(torch.sqrt(dR2))
                    return torch.stack(deltaR)

                return deltaR_ij

            for i in range(3):
                for j in range(i + 1, 3):
                    self.obs[r"\Delta R_{" + str(i + 1) + str(j + 1) + "}"] = deltaR(
                        i, j
                    )

        if self.cfg.data.dataset == "zplusjet":
            R0 = 0.4
            R0SoftDrop = 0.8
        elif self.cfg.data.dataset == "cms":
            R0 = 1.2
            R0SoftDrop = 1.2
        elif self.cfg.data.dataset == "ttbar":
            R0 = 1.2
            R0SoftDrop = 1.2

        def tau1(constituents, batch_idx, other_batch_idx):
            constituents = np.array(constituents.detach().cpu())
            batch_ptr = get_ptr_from_batch(batch_idx)
            taus = []
            for i in range(len(batch_ptr) - 1):
                event = constituents[batch_ptr[i] : batch_ptr[i + 1]]
                tau = compute_nsubjettiness(
                    event[..., [1, 2, 3, 0]], N=1, beta=1.0, R0=R0, axis_mode=3
                )
                taus.append(tau)
            return torch.tensor(taus)

        def tau2(constituents, batch_idx, other_batch_idx):
            constituents = np.array(constituents.detach().cpu())
            batch_ptr = get_ptr_from_batch(batch_idx)
            taus = []
            for i in range(len(batch_ptr) - 1):
                event = constituents[batch_ptr[i] : batch_ptr[i + 1]]
                tau = compute_nsubjettiness(
                    event[..., [1, 2, 3, 0]], N=2, beta=1.0, R0=R0, axis_mode=3
                )
                taus.append(tau)
            return torch.tensor(taus)

        if "tau1" in self.cfg.plotting.observables:
            self.obs[r"\tau_1"] = tau1
        if "tau2" in self.cfg.plotting.observables:
            self.obs[r"\tau_2"] = tau2
        if "tau21" in self.cfg.plotting.observables:
            self.obs[r"\tau_{21}"] = (
                lambda constituents, batch_idx, other_batch_idx: torch.where(
                    tau1(constituents, batch_idx, other_batch_idx) != 0,
                    tau2(constituents, batch_idx, other_batch_idx)
                    / tau1(constituents, batch_idx, other_batch_idx),
                    torch.tensor(0.0),
                )
            )
        if "sd_mass" in self.cfg.plotting.observables:

            def sd_mass(constituents, batch_idx, other_batch_idx):
                constituents = np.array(constituents.detach().cpu())
                batch_ptr = get_ptr_from_batch(batch_idx)
                log_rhos = []
                for i in range(len(batch_ptr) - 1):
                    event = constituents[batch_ptr[i] : batch_ptr[i + 1]]
                    sd_fourm = np.array(
                        apply_soft_drop(
                            event[..., [1, 2, 3, 0]], R0=R0SoftDrop, beta=0.0, zcut=0.1
                        )
                    )
                    mass2 = sd_fourm[3] ** 2 - np.sum(sd_fourm[..., :3] ** 2)
                    pt2 = np.sum(np.sum(event[..., 1:3], axis=0) ** 2)
                    log_rho = np.log(mass2 / pt2)
                    log_rhos.append(log_rho)
                return torch.tensor(log_rhos)

            self.obs[r"\log \rho"] = sd_mass

        if "momentum_fraction" in self.cfg.plotting.observables:

            def compute_zg(constituents, batch_idx, other_batch_idx):
                constituents = np.array(constituents.detach().cpu())
                batch_ptr = get_ptr_from_batch(batch_idx)
                zgs = []
                for i in range(len(batch_ptr) - 1):
                    event = constituents[batch_ptr[i] : batch_ptr[i + 1]]
                    zg = apply_soft_drop(
                        event[..., [1, 2, 3, 0]], R0=R0SoftDrop, beta=0.0, zcut=0.1
                    )[-1]
                    zgs.append(zg)
                return torch.tensor(zgs)

            self.obs[r"z_g"] = compute_zg

        if "jet_mass" in self.cfg.plotting.observables:

            def jet_mass(constituents, batch_idx, other_batch_idx):
                batch_ptr = get_ptr_from_batch(batch_idx)
                jet_masses = []
                for n in range(len(batch_ptr) - 1):
                    jet = constituents[batch_ptr[n] : batch_ptr[n + 1]].sum(dim=0)
                    mass2 = jet[0] ** 2 - (jet[1:] ** 2).sum(dim=-1)
                    jet_masses.append(torch.sqrt(mass2))
                return torch.stack(jet_masses)

            self.obs[r"M_{jet}"] = jet_mass
