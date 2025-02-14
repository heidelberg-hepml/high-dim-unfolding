import torch
import numpy as np
import energyflow
from torch_geometric.data import Data, Batch

from experiments.logger import LOGGER
from experiments.unfolding.utils import jetmomenta_to_fourmomenta, ensure_angle


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
            gen_particles = (gen_particles - gen_mean) / gen_std

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

    def prepare_standardize(self, particles, mults):
        mask = torch.arange(particles.shape[1])[None, :] < mults[:, None]
        flattened_particles = particles[mask]

        if self.cfg.modelname == "GATr":
            mean = flattened_particles.mean().unsqueeze(0).expand(1, 4)
            std = flattened_particles.std().unsqueeze(0).expand(1, 4)
        elif self.cfg.modelname == "Transformer":
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


float_to_pid = {
    0.0: 22,  # photon
    0.1: 211,  # pi+
    0.2: -211,  # pi-
    0.3: 130,  # K0_L
    0.4: 11,  # e-
    0.5: -11,  # e+
    0.6: 13,  # mu-
    0.7: -13,  # mu+
    0.8: 321,  # K+
    0.9: -321,  # K-
    1.0: 2212,  # proton
    1.1: -2212,  # anti-proton
    1.2: 2112,  # neutron
    1.3: -2112,  # anti-neutron
}


def single_pid_encoding(pid):
    if pid in [211, -11, -13, 321, 2212]:
        charge = 1
    elif pid in [-211, 11, 13, -321, -2212]:
        charge = -1
    else:
        charge = 0
    abs_pid = abs(pid)
    vector = [
        charge,  # Charge
        1 if abs_pid == 11 else 0,  # Electron
        1 if abs_pid == 13 else 0,  # Muon
        1 if abs_pid == 22 else 0,  # Photon
        1 if abs_pid in [211, 321, 2212] else 0,  # Charged Hadron
        1 if abs_pid in [130, 2112, 0] else 0,  # Neutral Hadron
    ]
    return vector


def pid_encoding(float_pids, cfg) -> torch.Tensor:
    vectors = []
    for float_pid in float_pids:
        if float_pid.item() not in float_to_pid:
            raise ValueError(f"Unknown PID: {float_pid}")
        pid = float_to_pid[float_pid.item()]
        vectors.append(single_pid_encoding(pid))
    vectors = torch.tensor(vectors)
    if cfg.save_pid:
        vectors = torch.cat([vectors, float_pids.unsqueeze(-1)], dim=-1)
    return vectors
