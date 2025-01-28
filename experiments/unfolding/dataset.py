import torch
import numpy as np
import energyflow
from torch_geometric.data import Data, Batch
from experiments.logger import LOGGER
from experiments.eventgen.helpers import jetmomenta_to_fourmomenta

EPS = 1e-5


class ZplusJetDataset(torch.utils.data.Dataset):
    def __init__(self, data_cfg):
        super().__init__()
        self.cfg = data_cfg

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def load_data(self, data, bounds):
        # create list of torch_geometric.data.Data objects
        self.data_list = []
        for det_event, gen_event, det_mult, gen_mult, index in zip(
            data["sim_particles"][bounds[0] : bounds[1]],
            data["gen_particles"][bounds[0] : bounds[1]],
            data["sim_mults"][bounds[0] : bounds[1]],
            data["gen_mults"][bounds[0] : bounds[1]],
            range(bounds[1] - bounds[0]),
        ):
            det_event = det_event[:det_mult]
            gen_event = gen_event[:gen_mult]
            if self.cfg.save_pid_encoding:
                det_scalars = pid_encoding(det_event[:, -1], cfg=self.cfg)
                gen_scalars = pid_encoding(gen_event[:, -1], cfg=self.cfg)
            elif self.cfg.save_pid:
                det_scalars = torch.tensor(det_event[:, -1], dtype=self.cfg.dtype)
                gen_scalars = torch.tensor(gen_event[:, -1], dtype=self.cfg.dtype)
            else:
                det_scalars = torch.zeros((det_event.shape[0], 0), dtype=self.cfg.dtype)
                gen_scalars = torch.zeros((gen_event.shape[0], 0), dtype=self.cfg.dtype)

            # replace pid with constant mass for all particles
            det_event[..., -1] = self.cfg.onshell_mass
            gen_event[..., -1] = self.cfg.onshell_mass

            # convert to fourmomenta
            det_fourmomenta = jetmomenta_to_fourmomenta(det_event)
            gen_fourmomenta = jetmomenta_to_fourmomenta(gen_event)

            det_data = Data(
                x=det_fourmomenta,
                scalars=det_scalars,
            )
            gen_data = Data(
                x=gen_fourmomenta,
                scalars=gen_scalars,
            )
            self.data_list.append((det_data, gen_data))


def collate(data_list):
    det_batch = Batch.from_data_list([data[0] for data in data_list])
    gen_batch = Batch.from_data_list([data[1] for data in data_list])
    return det_batch, gen_batch


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
    vectors = torch.tensor(vectors, dtype=cfg.dtype)
    if cfg.save_pid:
        float_pids = torch.tensor(float_pids, dtype=cfg.dtype)
        vectors = torch.cat(vectors, float_pids, dim=-1)
    return vectors
