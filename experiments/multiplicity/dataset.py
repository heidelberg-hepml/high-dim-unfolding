import energyflow
import torch
import numpy as np
from torch_geometric.data import Data

from experiments.multiplicity.utils import (
    ensure_angle,
    jetmomenta_to_fourmomenta,
    pid_encoding,
)
from experiments.logger import LOGGER


EPS = 1e-5


class MultiplicityDataset:

    def __init__(self, data_path, cfg):
        self.cfg = cfg
        if self.cfg.training.dtype == "float32":
            self.dtype = torch.float32
        elif self.cfg.training.dtype == "float64":
            self.dtype = torch.float64
        else:
            self.dtype = torch.float32
            LOGGER.warning(
                f"dtype={self.cfg.training.dtype} not recognized, using float32"
            )
        self.data = energyflow.zjets_delphes.load(
            "Herwig",
            num_data=self.cfg.data.length,
            pad=True,
            cache_dir=data_path,
            include_keys=["particles", "mults", "jets"],
        )
        self.prepare_data()

    def prepare_data(self):
        split = self.cfg.data.split
        size = len(self.data["sim_particles"])
        train_idx = int(split[0] * size)
        val_idx = int(split[1] * size)

        det_particles = torch.tensor(self.data["sim_particles"], dtype=self.dtype)
        det_jets = torch.tensor(self.data["sim_jets"], dtype=self.dtype)
        det_mults = torch.tensor(self.data["sim_mults"], dtype=torch.int)
        gen_mults = torch.tensor(self.data["gen_mults"], dtype=torch.int)

        # undo the dataset scaling
        det_particles[..., 1:3] = det_particles[..., 1:3] + det_jets[:, np.newaxis, 1:3]
        del det_jets
        det_particles[..., 2] = ensure_angle(det_particles[..., 2])
        det_particles[..., 0] = det_particles[..., 0] * 100

        # swap eta and phi for consistency
        det_particles[..., [1, 2]] = det_particles[..., [2, 1]]

        # save pids before replacing with mass
        det_pids = det_particles[..., 3].clone().unsqueeze(-1)
        if self.cfg.data.pid_encoding:
            det_pids = pid_encoding(det_pids)
        det_particles[..., 3] = self.cfg.data.mass

        if self.cfg.modelname == "GATr":
            det_particles = jetmomenta_to_fourmomenta(det_particles)

        if self.cfg.data.standardize:
            self.prepare_standardize(det_particles[:train_idx], det_mults[:train_idx])
            det_particles = (det_particles - self.mean) / self.std

        self.train_data_list = self.create_data_list(
            det_particles[:train_idx],
            det_pids[:train_idx],
            det_mults[:train_idx],
            gen_mults[:train_idx],
        )
        self.val_data_list = self.create_data_list(
            det_particles[train_idx : train_idx + val_idx],
            det_pids[train_idx : train_idx + val_idx],
            det_mults[train_idx : train_idx + val_idx],
            gen_mults[train_idx : train_idx + val_idx],
        )
        self.test_data_list = self.create_data_list(
            det_particles[train_idx + val_idx :],
            det_pids[train_idx + val_idx :],
            det_mults[train_idx + val_idx :],
            gen_mults[train_idx + val_idx :],
        )

    def prepare_standardize(self, det_particles, det_mults):
        mask = torch.arange(det_particles.shape[1])[None, :] < det_mults[:, None]
        flattened_particles = det_particles[mask]

        if self.cfg.modelname == "GATr":
            self.mean = flattened_particles.mean().unsqueeze(0).expand(1, 4)
            self.std = flattened_particles.std().unsqueeze(0).expand(1, 4)
        elif self.cfg.modelname == "Transformer":
            self.mean = flattened_particles.mean(dim=0, keepdim=True)
            self.mean[..., -1] = 0
            self.std = flattened_particles.std(dim=0, keepdim=True)
            self.std[..., -1] = 1
        else:
            raise ValueError(f"Not implemented for model {self.cfg.modelname}")
        del mask, flattened_particles

    def create_data_list(self, det_particles, det_pids, det_mults, gen_mults):
        assert len(det_particles) == len(det_mults) == len(gen_mults)

        labels = gen_mults.to(dtype=torch.int)

        # create list of torch_geometric.data.Data objects
        data_list = []
        for i in range(det_particles.shape[0]):

            if self.cfg.data.pid_raw or self.cfg.data.pid_encoding:
                scalars = det_pids[i, : det_mults[i]]
            else:
                scalars = torch.zeros((det_mults[i], 0), dtype=self.dtype)

            # store standardized pt, phi, eta, mass
            fourvector = det_particles[i, : det_mults[i]]

            if self.cfg.data.standardize:
                fourvector = (fourvector - self.mean) / self.std

            label = labels[i]

            graph = Data(
                x=fourvector, scalars=scalars, label=label, det_mult=det_mults[i]
            )
            data_list.append(graph)
        return data_list
