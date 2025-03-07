import torch
from torch_geometric.data import Data


class ZplusJetDataset(torch.utils.data.Dataset):
    def __init__(self, dtype):
        self.dtype = dtype

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def create_data_list(
        self, det_particles, det_pids, det_mults, gen_particles, gen_pids, gen_mults
    ):

        self.data_list = []
        for i in range(det_particles.shape[0]):
            det_event = det_particles[i, : det_mults[i]]
            det_scalars = det_pids[i, : det_mults[i]]

            gen_event = gen_particles[i, : gen_mults[i]]
            gen_scalars = gen_pids[i, : gen_mults[i]]

            graph = Data(
                x_det=det_event,
                scalars_det=det_scalars,
                mult_det=det_mults[i],
                x_gen=gen_event,
                scalars_gen=gen_scalars,
                mult_gen=gen_mults[i],
            )

            self.data_list.append(graph)
