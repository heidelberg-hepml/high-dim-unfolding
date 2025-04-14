import torch
from torch_geometric.data import Data
import energyflow
import numpy as np
import os

from gatr.interface import embed_vector
from experiments.unfolding.utils import (
    get_pt,
    ensure_angle,
    pid_encoding,
    jetmomenta_to_fourmomenta,
)
from experiments.unfolding.embedding import event_to_GA_with_spurions
from experiments.logger import LOGGER


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dtype, embed_into_GA=False, spurions=None):
        self.dtype = dtype
        self.embed_into_GA = embed_into_GA
        if embed_into_GA:
            self.spurions = spurions

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def create_data_list(
        self,
        det_particles,
        det_pids,
        det_mults,
        gen_particles,
        gen_pids,
        gen_mults,
        fourm=True,
    ):

        self.data_list = []
        for i in range(det_particles.shape[0]):
            if fourm:
                det_pt = get_pt(det_particles[i])
                gen_pt = get_pt(gen_particles[i])
            else:
                det_pt = det_particles[i, :, 0]
                gen_pt = gen_particles[i, :, 0]

            det_idx = torch.argsort(det_pt, descending=True)[: det_mults[i]]
            gen_idx = torch.argsort(gen_pt, descending=True)[: gen_mults[i]]

            det_event = det_particles[i, det_idx]
            det_scalars = det_pids[i, det_idx]
            gen_event = gen_particles[i, gen_idx]
            gen_scalars = gen_pids[i, gen_idx]

            if self.embed_into_GA:
                if self.spurions is not None:
                    det_event, det_scalars = event_to_GA_with_spurions(
                        det_event, det_scalars, self.spurions
                    )
                else:
                    det_event = embed_vector(det_event).unsqueeze(-2)

            graph = Data(
                x_det=det_event,
                scalars_det=det_scalars,
                x_gen=gen_event,
                scalars_gen=gen_scalars,
            )

            self.data_list.append(graph)


def load_zplusjet(data_path, cfg, dtype):
    data = energyflow.zjets_delphes.load(
        "Herwig",
        num_data=cfg.data.num_data,
        pad=True,
        cache_dir=data_path,
        include_keys=["particles", "mults", "jets"],
    )
    size = len(data["sim_particles"])

    det_particles = torch.tensor(data["sim_particles"], dtype=dtype)
    det_jets = torch.tensor(data["sim_jets"], dtype=dtype)
    det_mults = torch.tensor(data["sim_mults"], dtype=torch.int)

    gen_particles = torch.tensor(data["gen_particles"], dtype=dtype)
    gen_jets = torch.tensor(data["gen_jets"], dtype=dtype)
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
    if cfg.data.pid_encoding:
        det_pids = det_particles[..., 3].clone().unsqueeze(-1)
        det_pids = pid_encoding(det_pids)
        gen_pids = gen_particles[..., 3].clone().unsqueeze(-1)
        gen_pids = pid_encoding(gen_pids)
    else:
        det_pids = torch.empty(*det_particles.shape[:-1], 0, dtype=dtype)
        gen_pids = torch.empty(*gen_particles.shape[:-1], 0, dtype=dtype)

    det_particles[..., 3] = cfg.data.mass**2
    gen_particles[..., 3] = cfg.data.mass**2

    det_particles = jetmomenta_to_fourmomenta(det_particles)
    gen_particles = jetmomenta_to_fourmomenta(gen_particles)

    return {
        "det_particles": det_particles,
        "det_mults": det_mults,
        "det_pids": det_pids,
        "gen_particles": gen_particles,
        "gen_mults": gen_mults,
        "gen_pids": gen_pids,
    }


def load_cms(data_path, cfg, dtype):
    gen_particles = (
        torch.from_numpy(np.load(os.path.join(data_path, "gen_1725_delphes.npy")))
        .to(dtype)
        .reshape(-1, 3, 4)
    )[: cfg.data.num_data]
    det_particles = (
        torch.from_numpy(np.load(os.path.join(data_path, "rec_1725_delphes.npy")))
        .to(dtype)
        .reshape(-1, 3, 4)
    )[: cfg.data.num_data]
    size = len(gen_particles)
    gen_mults = (
        torch.zeros(gen_particles.shape[0], dtype=torch.int) + cfg.data.max_constituents
    )
    det_mults = (
        torch.zeros(det_particles.shape[0], dtype=torch.int) + cfg.data.max_constituents
    )
    gen_pids = torch.empty(*gen_particles.shape[:-1], 0, dtype=dtype)
    det_pids = torch.empty(*det_particles.shape[:-1], 0, dtype=dtype)
    return {
        "det_particles": det_particles,
        "det_mults": det_mults,
        "det_pids": det_pids,
        "gen_particles": gen_particles,
        "gen_mults": gen_mults,
        "gen_pids": gen_pids,
    }
