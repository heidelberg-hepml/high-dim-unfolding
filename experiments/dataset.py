import torch
from torch_geometric.data import Data
import energyflow
import numpy as np
import awkward as ak
import os

from experiments.utils import (
    ensure_angle,
    pid_encoding,
)
from experiments.coordinates import (
    jetmomenta_to_fourmomenta,
)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dtype, pos_encoding_dim=0):
        self.dtype = dtype
        if pos_encoding_dim > 0:
            self.pos_encoding = positional_encoding(pe_dim=pos_encoding_dim)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def create_data_list(
        self,
        det_particles,
        det_pids,
        det_mults,
        det_jets,
        gen_particles,
        gen_pids,
        gen_mults,
        gen_jets,
    ):

        self.data_list = []
        for i in range(det_particles.shape[0]):

            det_event = det_particles[i, : det_mults[i]]
            det_event_scalars = det_pids[i, : det_mults[i]]
            gen_event = gen_particles[i, : gen_mults[i]]
            gen_event_scalars = gen_pids[i, : gen_mults[i]]

            if hasattr(self, "pos_encoding"):
                det_event_scalars = torch.cat(
                    [det_event_scalars, self.pos_encoding[: det_mults[i]]], dim=-1
                )
                gen_event_scalars = torch.cat(
                    [gen_event_scalars, self.pos_encoding[: gen_mults[i]]], dim=-1
                )

            graph = Data(
                x_det=det_event,
                scalars_det=det_event_scalars,
                jet_det=det_jets[i : i + 1],
                x_gen=gen_event,
                scalars_gen=gen_event_scalars,
                jet_gen=gen_jets[i : i + 1],
            )

            self.data_list.append(graph)


def load_dataset(dataset_name):
    if dataset_name == "zplusjet":
        max_num_particles = 152
        diff = [-53, 78]
        pt_min = 0.0
        masked_dim = [3]
        load_fn = load_zplusjet
    elif dataset_name == "cms":
        max_num_particles = 3
        diff = [0, 0]
        pt_min = 30.0
        masked_dim = []
        load_fn = load_cms

    elif dataset_name == "ttbar":
        max_num_particles = 238
        diff = [-35, 101]
        pt_min = 0.0
        masked_dim = [3]
        load_fn = load_ttbar
    return max_num_particles, diff, pt_min, masked_dim, load_fn


def load_zplusjet(data_path, cfg, dtype):
    data = energyflow.zjets_delphes.load(
        "Herwig",
        num_data=cfg.length,
        pad=True,
        cache_dir=data_path,
        include_keys=["particles", "mults", "jets"],
    )

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

    det_idx = torch.argsort(det_particles[..., 0], descending=True, dim=1, stable=True)
    gen_idx = torch.argsort(gen_particles[..., 0], descending=True, dim=1, stable=True)
    det_particles = det_particles.take_along_dim(det_idx.unsqueeze(-1), dim=1)
    gen_particles = gen_particles.take_along_dim(gen_idx.unsqueeze(-1), dim=1)

    # save pids before replacing with mass
    if cfg.add_pid:
        det_pids = det_particles[..., 3].clone().unsqueeze(-1)
        det_pids = pid_encoding(det_pids)
        gen_pids = gen_particles[..., 3].clone().unsqueeze(-1)
        gen_pids = pid_encoding(gen_pids)
    else:
        det_pids = torch.empty(*det_particles.shape[:-1], 0, dtype=dtype)
        gen_pids = torch.empty(*gen_particles.shape[:-1], 0, dtype=dtype)

    det_particles[..., 3] = cfg.mass**2
    gen_particles[..., 3] = cfg.mass**2

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
    )[: cfg.length]
    det_particles = (
        torch.from_numpy(np.load(os.path.join(data_path, "rec_1725_delphes.npy")))
        .to(dtype)
        .reshape(-1, 3, 4)
    )[: cfg.length]
    gen_mults = torch.zeros(gen_particles.shape[0], dtype=torch.int) + 3
    det_mults = torch.zeros(det_particles.shape[0], dtype=torch.int) + 3
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


def load_ttbar(data_path, cfg, dtype):
    part1 = ak.from_parquet(os.path.join(data_path, "ttbar-t.parquet"))
    part2 = ak.from_parquet(os.path.join(data_path, "ttbar-tbar.parquet"))
    data = ak.concatenate([part1, part2], axis=0)[: cfg.length]

    size = cfg.length if cfg.length > 0 else len(data)
    shape = (size, 238, 4)

    det_mults = ak.to_torch(ak.num(data["rec_particles"], axis=1))
    det_jets = (ak.to_torch(data["rec_jets"])).to(dtype)
    det_particles = torch.zeros(shape, dtype=dtype)
    array = ak.to_torch(ak.flatten(data["rec_particles"], axis=1))
    start = 0
    for i, length in enumerate(det_mults):
        stop = start + length
        det_particles[i, :length] = array[start:stop]
        start = stop

    gen_mults = ak.to_torch(ak.num(data["gen_particles"], axis=1))
    gen_jets = (ak.to_torch(data["gen_jets"])).to(dtype)
    gen_particles = torch.zeros(shape, dtype=dtype)
    array = ak.to_torch(ak.flatten(data["gen_particles"], axis=1))
    start = 0
    for i, length in enumerate(gen_mults):
        stop = start + length
        gen_particles[i, :length] = array[start:stop]
        start = stop

    det_pids = torch.empty(*det_particles.shape[:-1], 0, dtype=dtype)
    gen_pids = torch.empty(*gen_particles.shape[:-1], 0, dtype=dtype)

    det_particles[..., 3] = cfg.mass**2
    gen_particles[..., 3] = cfg.mass**2

    det_idx = torch.argsort(det_particles[..., 0], descending=True, dim=1, stable=True)
    gen_idx = torch.argsort(gen_particles[..., 0], descending=True, dim=1, stable=True)
    det_particles = det_particles.take_along_dim(det_idx.unsqueeze(-1), dim=1)
    gen_particles = gen_particles.take_along_dim(gen_idx.unsqueeze(-1), dim=1)

    det_particles = jetmomenta_to_fourmomenta(det_particles)
    gen_particles = jetmomenta_to_fourmomenta(gen_particles)

    det_jets = jetmomenta_to_fourmomenta(det_jets)
    gen_jets = jetmomenta_to_fourmomenta(gen_jets)

    return {
        "det_particles": det_particles,
        "det_jets": det_jets,
        "det_mults": det_mults,
        "det_pids": det_pids,
        "gen_particles": gen_particles,
        "gen_jets": gen_jets,
        "gen_mults": gen_mults,
        "gen_pids": gen_pids,
    }


def positional_encoding(seq_length=256, pe_dim=16):
    """
    Create sinusoidal positional encoding.
    :param seq_length: Length of the sequence.
    :param pe_dim: Dimension of the encoding.
    :return: Positional encoding tensor of shape (seq_length, pe_dim).
    """
    position = torch.arange(seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, pe_dim, 2).float() * -(np.log(10000.0) / pe_dim)
    )
    pe = torch.zeros(seq_length, pe_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
