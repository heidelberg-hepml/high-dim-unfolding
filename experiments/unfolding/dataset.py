import torch
from torch_geometric.data import Data, Batch


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
            det_graph = Data(x=det_event, scalars=det_scalars, mult=det_mults[i])

            gen_event = gen_particles[i, : gen_mults[i]]
            gen_scalars = gen_pids[i, : gen_mults[i]]
            gen_graph = Data(x=gen_event, scalars=gen_scalars, mult=gen_mults[i])

            self.data_list.append((gen_graph, det_graph))


def collate(data_list):
    gen_batch = Batch.from_data_list([data[0] for data in data_list])
    det_batch = Batch.from_data_list([data[1] for data in data_list])
    return gen_batch, det_batch
