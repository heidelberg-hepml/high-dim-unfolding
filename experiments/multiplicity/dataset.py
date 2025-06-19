import torch
from torch_geometric.data import Data
import energyflow
import awkward as ak
import os

from experiments.logger import LOGGER
from experiments.multiplicity.utils import ensure_angle, pid_encoding


EPS = 1e-5


class MultiplicityDataset:

    def __init__(self, dtype):
        self.dtype = dtype

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def create_data_list(self, det_particles, det_pids, det_mults, gen_mults):

        # create list of torch_geometric.data.Data objects
        self.data_list = []
        for i in range(det_particles.shape[0]):
            # remove padding
            scalars = det_pids[i, : det_mults[i]]
            det_event = det_particles[i, : det_mults[i]]

            graph = Data(
                x=det_event, scalars=scalars, label=gen_mults[i], det_mult=det_mults[i]
            )
            self.data_list.append(graph)


def load_zplusjet(cfg, dtype, data_path):
    data = energyflow.zjets_delphes.load(
        "Herwig",
        num_data=cfg.length,
        pad=True,
        cache_dir=data_path,
        include_keys=["particles", "mults", "jets"],
    )
    LOGGER.info(f"Loaded {len(data['sim_particles'])} events from zplusjet dataset")

    det_particles = torch.tensor(data["sim_particles"], dtype=dtype)
    det_jets = torch.tensor(data["sim_jets"], dtype=dtype)
    det_mults = torch.tensor(data["sim_mults"], dtype=torch.int)
    gen_mults = torch.tensor(data["gen_mults"], dtype=torch.int)

    # undo the dataset scaling
    det_particles[..., 1:3] = det_particles[..., 1:3] + det_jets[:, None, 1:3]
    det_particles[..., 2] = ensure_angle(det_particles[..., 2])
    det_particles[..., 0] = det_particles[..., 0] * 100

    # swap eta and phi for consistency
    det_particles[..., [1, 2]] = det_particles[..., [2, 1]]

    # save pids before replacing with mass
    if cfg.pid_encoding:
        det_pids = det_particles[..., 3].clone().unsqueeze(-1)
        det_pids = pid_encoding(det_pids)
    else:
        det_pids = torch.empty(*det_particles.shape[:-1], 0, dtype=dtype)
    det_particles[..., 3] = cfg.mass

    return {
        "det_particles": det_particles,
        "det_pids": det_pids,
        "det_mults": det_mults,
        "gen_mults": gen_mults,
    }


def load_ttbar(cfg, dtype, data_path):
    part1 = ak.from_parquet(os.path.join(data_path, "ttbar-t.parquet"))
    part2 = ak.from_parquet(os.path.join(data_path, "ttbar-tbar.parquet"))
    data = ak.concatenate([part1, part2], axis=0)

    size = len(data["rec_particles"])
    shape = (size, cfg.max_num_particles, 4)

    det_mults = ak.to_torch(ak.num(data["rec_particles"], axis=1))
    det_particles = torch.zeros(shape, dtype=dtype)
    array = ak.to_torch(ak.flatten(data["rec_particles"], axis=1))
    start = 0
    for i, length in enumerate(det_mults):
        stop = start + length
        det_particles[i, :length] = array[start:stop]
        start = stop

    gen_mults = ak.to_torch(ak.num(data["gen_particles"], axis=1))

    det_pids = torch.empty(*det_particles.shape[:-1], 0, dtype=dtype)

    det_particles[..., 3] = cfg.mass

    return {
        "det_particles": det_particles,
        "det_pids": det_pids,
        "det_mults": det_mults,
        "gen_mults": gen_mults,
    }
