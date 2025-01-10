import torch
import numpy as np
import h5py
from torch_geometric.data import Data

EPS = 1e-5


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, rescale_data):
        super().__init__()
        self.rescale_data = rescale_data

    def load_data(self, filename, mode):
        raise NotImplementedError

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class MultiplicityDataset(BaseDataset):
    def load_data(
        self,
        filename,
        mode,
        mass=0.1,
        dtype=torch.float32,
    ):
        """
        Parameters
        ----------
        filename : str
            Path to file in npz format where the dataset in stored
        mode : {"train", "test", "val"}
            Purpose of the dataset
            Train, test and validation datasets are already separated in the specified file
        """
        data = h5py.File(filename, "r")
        particles = np.array(data[mode]["sim_particles"])
        sim_mults = np.array(data[mode]["sim_mults"], dtype=int)
        gen_mults = np.array(data[mode]["gen_mults"], dtype=int)

        kinematics = torch.tensor(particles, dtype=dtype)
        labels = torch.tensor(gen_mults, dtype=dtype)

        # create list of torch_geometric.data.Data objects
        self.data_list = []
        for i in range(kinematics.shape[0]):
            scalars = kinematics[
                i, : sim_mults[i], -1
            ].clone()  # store PID as scalar info
            fourmomenta = kinematics[i, : sim_mults[i]]
            fourmomenta[..., -1] = mass  # set constant mass for all fourmomenta
            label = labels[i, ...]

            data = Data(x=fourmomenta, scalars=scalars, label=label)
            self.data_list.append(data)
