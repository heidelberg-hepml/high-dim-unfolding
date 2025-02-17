import torch
import numpy as np
import energyflow
from torch_geometric.data import Data, Batch

from experiments.logger import LOGGER
from experiments.unfolding.utils import (
    jetmomenta_to_fourmomenta,
    ensure_angle,
    pid_encoding,
)
from experiments.unfolding.transforms import (
    Pt_to_LogPt,
    PtPhiEtaE_to_PtPhiEtaM2,
    EPPP_to_PtPhiEtaE,
    EPPP_to_PPPM2,
)


class ZplusJetDataset(torch.utils.data.Dataset):
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

        gen_particles = torch.tensor(self.data["gen_particles"], dtype=self.dtype)
        gen_jets = torch.tensor(self.data["gen_jets"], dtype=self.dtype)
        gen_mults = torch.tensor(self.data["gen_mults"], dtype=torch.int)

        # undo the dataset scaling
        det_particles[..., 1:3] = det_particles[..., 1:3] + det_jets[:, np.newaxis, 1:3]
        del det_jets
        det_particles[..., 2] = ensure_angle(det_particles[..., 2])
        det_particles[..., 0] = det_particles[..., 0] * 100

        gen_particles[..., 1:3] = gen_particles[..., 1:3] + gen_jets[:, np.newaxis, 1:3]
        del gen_jets
        gen_particles[..., 2] = ensure_angle(gen_particles[..., 2])
        gen_particles[..., 0] = gen_particles[..., 0] * 100

        # swap eta and phi for consistency
        det_particles[..., [1, 2]] = det_particles[..., [2, 1]]
        gen_particles[..., [1, 2]] = gen_particles[..., [2, 1]]

        # save pids before replacing with mass
        det_pids = det_particles[..., 3].clone().unsqueeze(-1)
        gen_pids = gen_particles[..., 3].clone().unsqueeze(-1)
        if self.cfg.data.pid_encoding:
            det_pids = pid_encoding(det_pids)
            gen_pids = pid_encoding(gen_pids)
        det_particles[..., 3] = self.cfg.data.mass
        gen_particles[..., 3] = self.cfg.data.mass

        det_particles = jetmomenta_to_fourmomenta(det_particles)
        gen_particles = jetmomenta_to_fourmomenta(gen_particles)

        if self.cfg.data.standardize:
            det_mean, det_std = self.prepare_standardize(
                det_particles[:train_idx], det_mults[:train_idx]
            )
            det_particles = (det_particles - det_mean) / det_std

        gen_mean, gen_std = self.prepare_standardize(
            gen_particles[:train_idx], gen_mults[:train_idx]
        )
        # gen_particles = (gen_particles - gen_mean) / gen_std
        self.train_gen_mean = gen_mean
        self.train_gen_std = gen_std

        self.train_data_list = zip(
            self.create_data_list(
                det_particles[:train_idx], det_pids[:train_idx], det_mults[:train_idx]
            ),
            self.create_data_list(
                gen_particles[:train_idx],
                gen_pids[:train_idx],
                gen_mults[:train_idx],
            ),
        )
        self.val_data_list = zip(
            self.create_data_list(
                det_particles[train_idx : train_idx + val_idx],
                det_pids[train_idx : train_idx + val_idx],
                det_mults[train_idx : train_idx + val_idx],
            ),
            self.create_data_list(
                gen_particles[train_idx : train_idx + val_idx],
                gen_pids[train_idx : train_idx + val_idx],
                gen_mults[train_idx : train_idx + val_idx],
            ),
        )
        self.test_data_list = zip(
            self.create_data_list(
                det_particles[train_idx + val_idx :],
                det_pids[train_idx + val_idx :],
                det_mults[train_idx + val_idx :],
            ),
            self.create_data_list(
                gen_particles[train_idx + val_idx :],
                gen_pids[train_idx + val_idx :],
                gen_mults[train_idx + val_idx :],
            ),
        )

    def prepare_standardize(self, particles, mults, coords=None):
        mask = torch.arange(particles.shape[1])[None, :] < mults[:, None]
        flattened_particles = particles[mask]

        if self.cfg.modelname == "ConditionalGATr":
            if coords == "StandardLogPtPhiEta":
                transform = Pt_to_LogPt(self.cfg.data.pt_min, self.cfg.data.units)
                flattened_particles = transform._forward(flattened_particles)
            mean = flattened_particles.mean().unsqueeze(0).expand(1, 4)
            std = flattened_particles.std().unsqueeze(0).expand(1, 4)
        elif self.cfg.modelname == "ConditionalTransformer":
            mean = flattened_particles.mean(dim=0, keepdim=True)
            mean[..., -1] = 0
            std = flattened_particles.std(dim=0, keepdim=True)
            std[..., -1] = 1
        else:
            raise ValueError(f"Not implemented for model {self.cfg.modelname}")

        return mean, std

    def create_data_list(self, particles, pids, mults):
        # create list of torch_geometric.data.Data objects
        data_list = []
        for i in range(particles.shape[0]):

            if self.cfg.data.pid_raw or self.cfg.data.pid_encoding:
                scalars = pids[i, : mults[i]]
            else:
                scalars = torch.zeros((mults[i], 0), dtype=self.dtype)

            # store standardized pt, phi, eta, mass
            event = particles[i, : mults[i]]

            graph = Data(x=event, scalars=scalars)
            data_list.append(graph)

        return data_list


def collate(data_list):
    gen_batch = Batch.from_data_list([data[0] for data in data_list])
    det_batch = Batch.from_data_list([data[1] for data in data_list])
    return gen_batch, det_batch
