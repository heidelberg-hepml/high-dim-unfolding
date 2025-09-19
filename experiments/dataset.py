import torch
from torch import nn
from torch_geometric.data import Data
import energyflow
import numpy as np
import awkward as ak
import os
import glob

from experiments.logger import LOGGER
from experiments.utils import (
    ensure_angle,
    fix_mass,
    get_mass,
    pid_encoding,
    GaussianFourierProjection,
)
from experiments.coordinates import jetmomenta_to_fourmomenta, fourmomenta_to_jetmomenta


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dtype, pos_encoding=None, mult_encoding=None):
        self.dtype = dtype
        self.pos_encoding = pos_encoding
        self.mult_encoding = mult_encoding
        self.data_list = []

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def append(
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

        for i in range(det_particles.shape[0]):

            det_event = det_particles[i, : det_mults[i]]
            det_event_scalars = det_pids[i, : det_mults[i]]
            gen_event = gen_particles[i, : gen_mults[i]]
            gen_event_scalars = gen_pids[i, : gen_mults[i]]

            if self.pos_encoding is not None:
                det_event_scalars = torch.cat(
                    [det_event_scalars, self.pos_encoding[: det_mults[i]]], dim=-1
                )
                gen_event_scalars = torch.cat(
                    [gen_event_scalars, self.pos_encoding[: gen_mults[i]]], dim=-1
                )
            if self.mult_encoding is not None:
                jet_scalars_det = self.mult_encoding(
                    torch.tensor([[det_mults[i]]], dtype=self.dtype)
                ).detach()
                jet_scalars_gen = self.mult_encoding(
                    torch.tensor([[gen_mults[i]]], dtype=self.dtype)
                ).detach()
            else:
                jet_scalars_det = torch.empty(1, 0, dtype=self.dtype)
                jet_scalars_gen = torch.empty(1, 0, dtype=self.dtype)

            graph = Data(
                x_det=det_event,
                scalars_det=det_event_scalars,
                jet_det=det_jets[i : i + 1],
                jet_scalars_det=jet_scalars_det,
                x_gen=gen_event,
                scalars_gen=gen_event_scalars,
                jet_gen=gen_jets[i : i + 1],
                jet_scalars_gen=jet_scalars_gen,
            )

            self.data_list.append(graph)


def load_dataset(dataset_name):
    if dataset_name == "zplusjet":
        max_num_particles = 152
        diff = [-53, 78]
        pt_min = 0.01
        jet_pt_min = 10.0
        masked_dim = [3]
        load_fn = load_zplusjet

    elif dataset_name == "cms":
        max_num_particles = 3
        diff = [0, 0]
        pt_min = 30.0
        jet_pt_min = 350.0
        masked_dim = []
        load_fn = load_cms

    elif dataset_name == "ttbar":
        max_num_particles = 238
        diff = [-35, 101]
        pt_min = 0.01
        jet_pt_min = 400.0
        masked_dim = [3]
        load_fn = load_ttbar
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return max_num_particles, diff, pt_min, jet_pt_min, masked_dim, load_fn


def load_zplusjet(data_path, cfg, dtype):
    data = energyflow.zjets_delphes.load(
        "Pythia26",
        # "Herwig",
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

    mult_mask = (det_mults >= cfg.min_mult) * (gen_mults >= cfg.min_mult)
    det_particles = det_particles[mult_mask]
    det_jets = det_jets[mult_mask]
    det_mults = det_mults[mult_mask]
    gen_particles = gen_particles[mult_mask]
    gen_jets = gen_jets[mult_mask]
    gen_mults = gen_mults[mult_mask]

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

    det_mask = (
        torch.arange(det_particles.shape[1])[None, :] < det_mults[:, None]
    ).unsqueeze(-1)
    gen_mask = (
        torch.arange(gen_particles.shape[1])[None, :] < gen_mults[:, None]
    ).unsqueeze(-1)

    det_particles = fix_mass(jetmomenta_to_fourmomenta(det_particles), cfg.mass)
    gen_particles = fix_mass(jetmomenta_to_fourmomenta(gen_particles), cfg.mass)

    det_jets = (det_particles * det_mask).sum(dim=1)
    gen_jets = (gen_particles * gen_mask).sum(dim=1)

    if cfg.pt_cut > 0:
        det_mask = det_jets[..., 0] > cfg.pt_cut
        gen_mask = gen_jets[..., 0] > cfg.pt_cut
        mask = det_mask & gen_mask

        det_jets = det_jets[mask]
        det_particles = det_particles[mask]
        det_mults = det_mults[mask]
        det_pids = det_pids[mask]
        gen_jets = gen_jets[mask]
        gen_particles = gen_particles[mask]
        gen_mults = gen_mults[mask]
        gen_pids = gen_pids[mask]

    LOGGER.info(f"First mult: {gen_mults[0]}")

    if getattr(cfg, "use_sampled_jets", False):
        train = torch.load(
            "/remote/gpu04/petitjean/high-dim-unfolding/runs/long_jets_6/z_Tr_Lion2-4_1024_noconst_2481358/samples_4/samples_train.pt",
            weights_only=False,
            map_location=gen_jets.device,
        )
        val = torch.load(
            "/remote/gpu04/petitjean/high-dim-unfolding/runs/long_jets_6/z_Tr_Lion2-4_1024_noconst_2481358/samples_4/samples_val.pt",
            weights_only=False,
            map_location=gen_jets.device,
        )
        test = torch.load(
            "/remote/gpu04/petitjean/high-dim-unfolding/runs/long_jets_6/z_Tr_Lion2-4_1024_noconst_2481358/samples_4/samples_test.pt",
            weights_only=False,
            map_location=gen_jets.device,
        )
        LOGGER.info(
            f"Using sampled jets from {train.jet_gen.shape[0]} train, {val.jet_gen.shape[0]} val, {test.jet_gen.shape[0]} test events"
        )
        mult = torch.cat(
            [train.x_gen_ptr.diff(), val.x_gen_ptr.diff(), test.x_gen_ptr.diff()], dim=0
        )
        det_jets = det_jets[:-1]
        det_particles = det_particles[:-1]
        det_mults = det_mults[:-1]
        det_pids = det_pids[:-1]
        gen_jets = gen_jets[:-1]
        gen_particles = gen_particles[:-1]
        gen_mults = gen_mults[:-1]
        gen_pids = gen_pids[:-1]
        for i in range(mult.shape[0]):
            if mult[i] != gen_mults[i + 1]:
                LOGGER.warning(
                    f"Sampled jet mult {mult[i]} != gen mult {gen_mults[i]} at index {i}"
                )
                break

        sampled_jets = torch.cat([train.jet_gen, val.jet_gen, test.jet_gen], dim=0)
        gen_jets = sampled_jets

    if cfg.part_to_jet:
        det_particles = jetmomenta_to_fourmomenta(det_jets.unsqueeze(1))
        det_mults = torch.ones(det_jets.shape[0], dtype=torch.int)
        gen_particles = jetmomenta_to_fourmomenta(gen_jets.unsqueeze(1))
        gen_mults = torch.ones(gen_jets.shape[0], dtype=torch.int)

    return {
        "det_jets": det_jets,
        "det_particles": det_particles,
        "det_mults": det_mults,
        "det_pids": det_pids,
        "gen_jets": gen_jets,
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
    parquet_files = sorted(glob.glob(os.path.join(data_path, "ttbar*.parquet")))
    data_parts = [ak.from_parquet(file) for file in parquet_files]
    data = ak.concatenate(data_parts, axis=0)[: cfg.length]

    size = cfg.length if cfg.length > 0 else len(data)
    shape = (size, 240, 4)

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

    mult_mask = (det_mults >= cfg.min_mult) * (gen_mults >= cfg.min_mult)
    det_particles = det_particles[mult_mask]
    det_jets = det_jets[mult_mask]
    det_mults = det_mults[mult_mask]
    gen_particles = gen_particles[mult_mask]
    gen_jets = gen_jets[mult_mask]
    gen_mults = gen_mults[mult_mask]

    det_pids = torch.empty(*det_particles.shape[:-1], 0, dtype=dtype)
    gen_pids = torch.empty(*gen_particles.shape[:-1], 0, dtype=dtype)

    det_idx = torch.argsort(det_particles[..., 0], descending=True, dim=1, stable=True)
    gen_idx = torch.argsort(gen_particles[..., 0], descending=True, dim=1, stable=True)
    det_particles = det_particles.take_along_dim(det_idx.unsqueeze(-1), dim=1)
    gen_particles = gen_particles.take_along_dim(gen_idx.unsqueeze(-1), dim=1)

    det_mask = (
        torch.arange(det_particles.shape[1])[None, :] < det_mults[:, None]
    ).unsqueeze(-1)
    gen_mask = (
        torch.arange(gen_particles.shape[1])[None, :] < gen_mults[:, None]
    ).unsqueeze(-1)

    det_particles = fix_mass(jetmomenta_to_fourmomenta(det_particles), cfg.mass)
    gen_particles = fix_mass(jetmomenta_to_fourmomenta(gen_particles), cfg.mass)

    det_jets = (det_particles * det_mask).sum(dim=1)
    gen_jets = (gen_particles * gen_mask).sum(dim=1)

    if cfg.part_to_jet:
        det_particles = jetmomenta_to_fourmomenta(det_jets.unsqueeze(1))
        det_mults = torch.ones(det_jets.shape[0], dtype=torch.int)
        gen_particles = jetmomenta_to_fourmomenta(gen_jets.unsqueeze(1))
        gen_mults = torch.ones(gen_jets.shape[0], dtype=torch.int)

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


def load_ttbar_file(file, cfg, dtype, length):
    data = ak.from_parquet(file)
    mult_mask = (ak.num(data["rec_particles"], axis=1) >= cfg.min_mult) * (
        ak.num(data["gen_particles"], axis=1) >= cfg.min_mult
    )
    data = data[mult_mask]

    size = (
        min(cfg.length, len(data["rec_particles"]))
        if cfg.length > 0
        else len(data["rec_particles"])
    )
    data = data[:size]
    det_shape = (size, 182, 4)
    gen_shape = (size, 240, 4)

    det_mults = ak.to_torch(ak.num(data["rec_particles"], axis=1))
    det_jets = (ak.to_torch(data["rec_jets"])).to(dtype)
    det_particles = torch.empty(det_shape, dtype=dtype)
    array = ak.to_torch(ak.flatten(data["rec_particles"], axis=1))
    start = 0
    for i, length in enumerate(det_mults):
        stop = start + length
        det_particles[i, :length] = array[start:stop]
        start = stop

    gen_mults = ak.to_torch(ak.num(data["gen_particles"], axis=1))
    gen_jets = (ak.to_torch(data["gen_jets"])).to(dtype)
    gen_particles = torch.empty(gen_shape, dtype=dtype)
    array = ak.to_torch(ak.flatten(data["gen_particles"], axis=1))
    start = 0
    for i, length in enumerate(gen_mults):
        stop = start + length
        gen_particles[i, :length] = array[start:stop]
        start = stop

    # mult_mask = (det_mults >= cfg.min_mult) * (gen_mults >= cfg.min_mult)
    # det_particles = det_particles[mult_mask]
    # det_jets = det_jets[mult_mask]
    # det_mults = det_mults[mult_mask]
    # gen_particles = gen_particles[mult_mask]
    # gen_jets = gen_jets[mult_mask]
    # gen_mults = gen_mults[mult_mask]

    det_pids = torch.empty(*det_particles.shape[:-1], 0, dtype=dtype)
    gen_pids = torch.empty(*gen_particles.shape[:-1], 0, dtype=dtype)

    det_idx = torch.argsort(det_particles[..., 0], descending=True, dim=1, stable=True)
    gen_idx = torch.argsort(gen_particles[..., 0], descending=True, dim=1, stable=True)
    det_particles = det_particles.take_along_dim(det_idx.unsqueeze(-1), dim=1)
    gen_particles = gen_particles.take_along_dim(gen_idx.unsqueeze(-1), dim=1)

    det_mask = (
        torch.arange(det_particles.shape[1])[None, :] < det_mults[:, None]
    ).unsqueeze(-1)
    gen_mask = (
        torch.arange(gen_particles.shape[1])[None, :] < gen_mults[:, None]
    ).unsqueeze(-1)

    det_particles = fix_mass(jetmomenta_to_fourmomenta(det_particles), cfg.mass)
    gen_particles = fix_mass(jetmomenta_to_fourmomenta(gen_particles), cfg.mass)

    det_jets = (det_particles * det_mask).sum(dim=1)
    gen_jets = (gen_particles * gen_mask).sum(dim=1)

    if cfg.part_to_jet:
        det_particles = jetmomenta_to_fourmomenta(det_jets.unsqueeze(1))
        det_mults = torch.ones(det_jets.shape[0], dtype=torch.int)
        gen_particles = jetmomenta_to_fourmomenta(gen_jets.unsqueeze(1))
        gen_mults = torch.ones(gen_jets.shape[0], dtype=torch.int)

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
