import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from torch_geometric.data import Batch

import os, time
from omegaconf import open_dict

from gatr.interface import embed_spurions, extract_vector
from experiments.base_experiment import BaseExperiment
from experiments.unfolding.dataset import (
    Dataset,
    load_cms,
    load_zplusjet,
    positional_encoding,
)
from experiments.unfolding.utils import (
    get_ptr_from_batch,
)
import experiments.unfolding.plotter as plotter
from experiments.unfolding.plots import plot_data
from experiments.logger import LOGGER


class UnfoldingExperiment(BaseExperiment):
    def init_physics(self):

        with open_dict(self.cfg):
            self.cfg.modelname = self.cfg.model._target_.rsplit(".", 1)[-1][:-3]

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
            elif self.cfg.modelname == "SimpleConditionalTransformer":
                self.cfg.data.embed_det_in_GA = False
                self.cfg.model.net.in_channels = 4 + self.cfg.cfm.embed_t_dim
                self.cfg.model.net_condition.in_channels = 4
                self.cfg.model.net_condition.out_channels = (
                    self.cfg.model.net.hidden_channels
                )
                if self.cfg.data.pid_encoding:
                    self.cfg.model.net.in_channels += 6
                    self.cfg.model.net_condition.in_channels += 6
                if self.cfg.model.net.pos_encoding_type == "absolute":
                    if self.cfg.data.max_constituents > 0:
                        self.cfg.model.net.pos_encoding_base = (
                            self.cfg.data.max_constituents
                        )
                    else:
                        self.cfg.model.net.pos_encoding_base = (
                            self.cfg.data.max_num_particles
                        )

                if self.cfg.model.net_condition.pos_encoding_type == "absolute":
                    if self.cfg.data.max_constituents > 0:
                        self.cfg.model.net_condition.pos_encoding_base = (
                            self.cfg.data.max_constituents
                        )
                    else:
                        self.cfg.model.net_condition.pos_encoding_base = (
                            self.cfg.data.max_num_particles
                        )

            elif self.cfg.modelname == "SimpleConditionalGATr":
                self.cfg.data.embed_det_in_GA = True
                self.cfg.model.net.in_s_channels = (
                    self.cfg.cfm.embed_t_dim + self.cfg.data.pos_encoding_dim
                )
                self.cfg.model.net_condition.in_s_channels = (
                    self.cfg.cfm.embed_t_dim + self.cfg.data.pos_encoding_dim
                )
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

                if self.cfg.model.net.attention.pos_encoding_type == "absolute":
                    if self.cfg.data.max_constituents > 0:
                        self.cfg.model.net.attention.pos_encoding_base = (
                            self.cfg.data.max_constituents + self.cfg.data.num_spurions
                        )
                    else:
                        self.cfg.model.net.attention.pos_encoding_base = (
                            self.cfg.data.max_num_particles + self.cfg.data.num_spurions
                        )

                if self.cfg.model.net.crossattention.pos_encoding_type == "absolute":
                    if self.cfg.data.max_constituents > 0:
                        self.cfg.model.net.crossattention.pos_encoding_base = (
                            self.cfg.data.max_constituents + self.cfg.data.num_spurions
                        )
                    else:
                        self.cfg.model.net.crossattention.pos_encoding_base = (
                            self.cfg.data.max_num_particles + self.cfg.data.num_spurions
                        )

                if (
                    self.cfg.model.net_condition.attention.pos_encoding_type
                    == "absolute"
                ):
                    if self.cfg.data.max_constituents > 0:
                        self.cfg.model.net_condition.attention.pos_encoding_base = (
                            self.cfg.data.max_constituents + self.cfg.data.num_spurions
                        )
                    else:
                        self.cfg.model.net_condition.attention.pos_encoding_base = (
                            self.cfg.data.max_num_particles + self.cfg.data.num_spurions
                        )

            if self.cfg.data.dataset == "cms":
                if self.cfg.data.max_constituents == -1:
                    self.cfg.data.max_constituents = 3
                self.cfg.data.max_num_particles = 3
                self.cfg.data.pt_min = 30.0
                self.cfg.data.units = 10.0
                self.cfg.cfm.masked_dims = []
                self.cfg.plotting.observables = []

            # copy model-specific parameters
            self.cfg.model.odeint = self.cfg.odeint
            self.cfg.model.cfm = self.cfg.cfm

        self.define_process_specifics()

    def init_data(self):
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
            data = load_zplusjet(data_path, self.cfg, self.dtype)
        elif self.cfg.data.dataset == "cms":
            data = load_cms(data_path, self.cfg, self.dtype)
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
            pt_min=self.cfg.data.pt_min,
            base_type=self.cfg.data.base_type,
            onshell_mass=self.cfg.data.mass,
        )
        self.model.init_distribution()
        self.model.init_coordinates()

        # initialize coordinates
        train_gen_mask = (
            torch.arange(gen_particles.shape[1])[None, :] < gen_mults[:train_idx, None]
        )
        train_gen_data = gen_particles[:train_idx][train_gen_mask]
        self.model.coordinates.init_fit(train_gen_data)
        self.model.distribution.coordinates.init_fit(train_gen_data)

        # initialize condition_coordinates (might require data)
        train_det_mask = (
            torch.arange(det_particles.shape[1])[None, :] < det_mults[:train_idx, None]
        )
        train_det_data = det_particles[:train_idx][train_det_mask]
        self.model.condition_coordinates.init_fit(train_det_data)

        # transform before training
        if self.cfg.modelname == "SimpleConditionalTransformer":
            gen_mask = (
                torch.arange(gen_particles.shape[1])[None, :] < gen_mults[:, None]
            )
            det_mask = (
                torch.arange(det_particles.shape[1])[None, :] < det_mults[:, None]
            )

            gen_data = gen_particles[gen_mask]
            gen_data = self.model.coordinates.fourmomenta_to_x(gen_data)

            det_data = det_particles[det_mask]
            det_data = self.model.condition_coordinates.fourmomenta_to_x(det_data)

            if self.cfg.data.dataset == "zplusjet":
                gen_data[..., 3] = 2 * torch.log(torch.tensor(self.cfg.data.mass))
                det_data[..., 3] = 2 * torch.log(torch.tensor(self.cfg.data.mass))

            gen_particles[gen_mask] = gen_data
            det_particles[det_mask] = det_data

            if self.cfg.data.standardize:
                train_gen_mask = train_gen_mask.unsqueeze(-1)
                train_det_mask = train_det_mask.unsqueeze(-1)

                self.gen_mean = (gen_particles[:train_idx] * train_gen_mask).sum(
                    dim=0, keepdim=True
                ) / train_gen_mask.sum(dim=0, keepdim=True)

                self.gen_std = torch.sqrt(
                    (
                        (
                            gen_particles[:train_idx] * train_gen_mask
                            - self.gen_mean * train_gen_mask
                        )
                        ** 2
                    ).sum(dim=0, keepdim=True)
                    / train_gen_mask.sum(dim=0, keepdim=True)
                )
                self.gen_std[self.gen_std == 0] = 1.0

                self.det_mean = (det_particles[:train_idx] * train_det_mask).sum(
                    dim=0, keepdim=True
                ) / train_det_mask.sum(dim=0, keepdim=True)

                self.det_std = torch.sqrt(
                    (
                        (
                            det_particles[:train_idx] * train_det_mask
                            - self.det_mean * train_det_mask
                        )
                        ** 2
                    ).sum(dim=0, keepdim=True)
                    / train_det_mask.sum(dim=0, keepdim=True)
                )
                self.det_std[self.det_std == 0] = 1.0

                if self.model.coordinates.contains_phi:
                    self.gen_std[..., 1] = 1.0
                if self.model.condition_coordinates.contains_phi:
                    self.det_std[..., 1] = 1.0
                self.gen_std[..., self.cfg.cfm.masked_dims] = 1.0
                self.det_std[..., self.cfg.cfm.masked_dims] = 1.0

                gen_particles = gen_particles - self.gen_mean
                det_particles = det_particles - self.det_mean
                gen_particles = gen_particles / self.gen_std
                det_particles = det_particles / self.det_std

            else:
                self.gen_mean = torch.zeros(1, *gen_particles.shape[1:])
                self.gen_std = torch.ones(1, *gen_particles.shape[1:])
                self.det_mean = torch.zeros(1, *det_particles.shape[1:])
                self.det_std = torch.ones(1, *det_particles.shape[1:])

        if self.cfg.modelname == "SimpleConditionalGATr":

            gen_mask = (
                torch.arange(gen_particles.shape[1])[None, :] < gen_mults[:, None]
            )

            gen_data = gen_particles[gen_mask]
            gen_data = self.model.coordinates.fourmomenta_to_x(gen_data)

            if self.cfg.data.dataset == "zplusjet":
                gen_data[..., 3] = 2 * torch.log(torch.tensor(self.cfg.data.mass))

            gen_particles[gen_mask] = gen_data

            if self.cfg.data.standardize:
                train_gen_mask = train_gen_mask.unsqueeze(-1)

                self.gen_mean = (gen_particles[:train_idx] * train_gen_mask).sum(
                    dim=0, keepdim=True
                ) / train_gen_mask.sum(dim=0, keepdim=True)

                self.gen_std = torch.sqrt(
                    (
                        (
                            gen_particles[:train_idx] * train_gen_mask
                            - self.gen_mean * train_gen_mask
                        )
                        ** 2
                    ).sum(dim=0, keepdim=True)
                    / train_gen_mask.sum(dim=0, keepdim=True)
                )
                self.gen_std[self.gen_std == 0] = 1.0

                if self.model.coordinates.contains_phi:
                    self.gen_std[..., 1] = 1.0

                self.gen_std[..., self.cfg.cfm.masked_dims] = 1.0

                gen_particles = gen_particles - self.gen_mean
                gen_particles = gen_particles / self.gen_std

            else:
                self.gen_mean = torch.zeros(1, *gen_particles.shape[1:])
                self.gen_std = torch.ones(1, *gen_particles.shape[1:])

            self.model.set_ms(self.gen_mean, self.gen_std)

        if self.cfg.data.pos_encoding_dim > 0:
            if self.cfg.data.max_constituents > 0:
                seq_length = self.cfg.data.max_constituents
            else:
                seq_length = self.cfg.data.max_num_particles
            pos_encoding = positional_encoding(
                seq_length, self.cfg.data.pos_encoding_dim
            )
        else:
            pos_encoding = None

        # initialize geometry
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

        if self.cfg.modelname == "SimpleConditionalTransformer":
            fourm = False
        else:
            fourm = True

        self.train_data = Dataset(
            self.dtype,
            self.cfg.data.embed_det_in_GA,
            self.spurions,
            fourm=fourm,
            pos_encoding=pos_encoding,
        )
        self.val_data = Dataset(
            self.dtype,
            self.cfg.data.embed_det_in_GA,
            self.spurions,
            fourm=fourm,
            pos_encoding=pos_encoding,
        )
        self.test_data = Dataset(
            self.dtype,
            self.cfg.data.embed_det_in_GA,
            self.spurions,
            fourm=fourm,
            pos_encoding=pos_encoding,
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

        if self.cfg.modelname == "SimpleConditionalTransformer":
            self.gen_mean = self.gen_mean.to(self.device)
            self.gen_std = self.gen_std.to(self.device)
            self.det_mean = self.det_mean.to(self.device)
            self.det_std = self.det_std.to(self.device)

        elif self.cfg.modelname == "SimpleConditionalGATr":
            self.gen_mean = self.gen_mean.to(self.device)
            self.gen_std = self.gen_std.to(self.device)

        for i in range(n_batches):
            batch = next(it).to(self.device)

            sample_batch = self.model.sample(
                batch,
                self.device,
                self.dtype,
            )

            if self.cfg.modelname == "SimpleConditionalTransformer":

                # undo gen standardization
                gen_indices = (
                    torch.arange(len(batch.x_gen), device=batch.x_gen.device)
                    - batch.x_gen_ptr[batch.x_gen_batch]
                )

                gen_std_broadcasted = self.gen_std.squeeze(0)[gen_indices]
                gen_mean_broadcasted = self.gen_mean.squeeze(0)[gen_indices]

                sample_batch.x_gen = (
                    sample_batch.x_gen * gen_std_broadcasted + gen_mean_broadcasted
                )
                batch.x_gen = batch.x_gen * gen_std_broadcasted + gen_mean_broadcasted

                # undo det standardization
                det_indices = (
                    torch.arange(len(batch.x_det), device=batch.x_det.device)
                    - batch.x_det_ptr[batch.x_det_batch]
                )

                det_std_broadcasted = self.det_std.squeeze(0)[det_indices]
                det_mean_broadcasted = self.det_mean.squeeze(0)[det_indices]
                sample_batch.x_det = (
                    sample_batch.x_det * det_std_broadcasted + det_mean_broadcasted
                )
                batch.x_det = batch.x_det * det_std_broadcasted + det_mean_broadcasted

                sample_batch.x_gen = self.model.coordinates.x_to_fourmomenta(
                    sample_batch.x_gen
                )
                sample_batch.x_det = self.model.condition_coordinates.x_to_fourmomenta(
                    sample_batch.x_det
                )
                batch.x_gen = self.model.coordinates.x_to_fourmomenta(batch.x_gen)
                batch.x_det = self.model.condition_coordinates.x_to_fourmomenta(
                    batch.x_det
                )

            elif self.cfg.modelname == "SimpleConditionalGATr":
                # undo gen standardization
                gen_indices = (
                    torch.arange(len(batch.x_gen), device=batch.x_gen.device)
                    - batch.x_gen_ptr[batch.x_gen_batch]
                )

                gen_std_broadcasted = self.gen_std.squeeze(0)[gen_indices]
                gen_mean_broadcasted = self.gen_mean.squeeze(0)[gen_indices]

                sample_batch.x_gen = (
                    sample_batch.x_gen * gen_std_broadcasted + gen_mean_broadcasted
                )
                batch.x_gen = batch.x_gen * gen_std_broadcasted + gen_mean_broadcasted

                sample_batch.x_gen = self.model.coordinates.x_to_fourmomenta(
                    sample_batch.x_gen
                )
                batch.x_gen = self.model.coordinates.x_to_fourmomenta(batch.x_gen)

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

    def plot(self):
        path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(os.path.join(path), exist_ok=True)
        LOGGER.info(f"Creating plots in {path}")
        t0 = time.time()

        if self.cfg.modelname == "ConditionalTransformer":
            model_label = "CondTr"
        elif self.cfg.modelname == "SimpleConditionalTransformer":
            model_label = "SCondTr"
        elif self.cfg.modelname == "ConditionalGATr":
            model_label = "CondGATr"
        elif self.cfg.modelname == "SimpleConditionalGATr":
            model_label = "SCondGATr"
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
