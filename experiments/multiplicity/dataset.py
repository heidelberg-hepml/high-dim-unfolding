import torch
from torch_geometric.data import Data

from experiments.logger import LOGGER


EPS = 1e-5


class MultiplicityDataset:

    def __init__(self, pid_encoding, dtype):
        self.pid_encoding = pid_encoding
        self.dtype = dtype

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def create_data_list(self, det_particles, det_pids, det_mults, gen_mults):

        # create list of torch_geometric.data.Data objects
        self.data_list = []
        for i in range(det_particles.shape[0]):

            if self.cfg.data.pid_encoding:
                scalars = det_pids[i, : det_mults[i]]
            else:
                scalars = torch.zeros((det_mults[i], 0), dtype=self.dtype)

            # store standardized pt, phi, eta, mass
            det_event = det_particles[i, : det_mults[i]]

            label = labels[i]

            graph = Data(
                x=det_event, scalars=scalars, label=gen_mults, det_mult=det_mults[i]
            )
            self.data_list.append(graph)
