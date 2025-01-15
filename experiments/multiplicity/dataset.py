import torch
import numpy as np
import energyflow
from torch_geometric.data import Data
from experiments.logger import LOGGER

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
        data,
        mode,
        split=[0.8, 0.1, 0.1],
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
        split : list of float
            Fraction of data to use for training, testing and validation
        mass : float
            Mass of the particle to reconstruct
        dtype : torch.dtype
            Data type of the tensors
        """

        size = len(data["sim_particles"])

        if mode == "train":
            particles = np.array(data["sim_particles"])[: int(split[0] * size)]
            sim_mults = np.array(data["sim_mults"], dtype=int)[: int(split[0] * size)]
            gen_mults = np.array(data["gen_mults"], dtype=int)[: int(split[0] * size)]
        elif mode == "val":
            particles = np.array(data["sim_particles"])[
                int(split[0] * size) : int(split[0] * size) + int(split[1] * size)
            ]
            sim_mults = np.array(data["sim_mults"], dtype=int)[
                int(split[0] * size) : int(split[0] * size) + int(split[1] * size)
            ]
            gen_mults = np.array(data["gen_mults"], dtype=int)[
                int(split[0] * size) : int((split[0] + split[1]) * size)
            ]
        else:
            particles = np.array(data["sim_particles"])[
                int((split[0] + split[1]) * size) :
            ]
            sim_mults = np.array(data["sim_mults"], dtype=int)[
                int((split[0] + split[1]) * size) :
            ]
            gen_mults = np.array(data["gen_mults"], dtype=int)[
                int((split[0] + split[1]) * size) :
            ]

        kinematics = torch.tensor(particles, dtype=dtype)
        labels = torch.tensor(gen_mults, dtype=torch.int)

        # create list of torch_geometric.data.Data objects
        self.data_list = []
        for i in range(kinematics.shape[0]):
            scalars = kinematics[
                i, : sim_mults[i], -1
            ].clone()  # store PID as scalar info
            fourmomenta = kinematics[i, : sim_mults[i]]
            fourmomenta[..., -1] = mass  # set constant mass for all fourmomenta
            label = labels[i, ...]

            data = Data(x=fourmomenta, scalars=scalars, label=label, sim_mult=sim_mults[i])
            self.data_list.append(data)
