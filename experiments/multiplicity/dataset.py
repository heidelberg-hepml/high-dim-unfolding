import torch
import numpy as np
import energyflow
from torch_geometric.data import Data
from experiments.logger import LOGGER

EPS = 1e-5


class MultiplicityDataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

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
            det_particles = np.array(data["sim_particles"])[
                : int(split[0] * size)
            ]  # detector-level particles
            det_mults = np.array(data["sim_mults"], dtype=int)[
                : int(split[0] * size)
            ]  # detector-level multiplicity
            gen_mults = np.array(data["gen_mults"], dtype=int)[
                : int(split[0] * size)
            ]  # genlevel multiplicity
        elif mode == "val":
            det_particles = np.array(data["sim_particles"])[
                int(split[0] * size) : int((split[0] + split[1]) * size)
            ]
            det_mults = np.array(data["sim_mults"], dtype=int)[
                int(split[0] * size) : int((split[0] + split[1]) * size)
            ]
            gen_mults = np.array(data["gen_mults"], dtype=int)[
                int(split[0] * size) : int((split[0] + split[1]) * size)
            ]
        elif mode == "test":
            det_particles = np.array(data["sim_particles"])[
                int((split[0] + split[1]) * size) :
            ]
            det_mults = np.array(data["sim_mults"], dtype=int)[
                int((split[0] + split[1]) * size) :
            ]
            gen_mults = np.array(data["gen_mults"], dtype=int)[
                int((split[0] + split[1]) * size) :
            ]
        else:
            raise ValueError("Mode must be one of {'train', 'test', 'val'}")

        kinematics = torch.tensor(
            det_particles, dtype=dtype
        )  # contains p_T, eta, phi, PID
        labels = torch.tensor(gen_mults, dtype=torch.int)

        # create list of torch_geometric.data.Data objects
        self.data_list = []
        for i in range(kinematics.shape[0]):
            scalars = (
                kinematics[i, : det_mults[i], -1].clone().unsqueeze(-1)
            )  # store PID as scalar info
            pepm = kinematics[i, : det_mults[i]]  # save pt, eta, phi, mass
            pepm[..., -1] = mass  # replace PID by constant mass for all particles
            label = labels[i, ...]

            data = Data(x=pepm, scalars=scalars, label=label, det_mult=det_mults[i])
            self.data_list.append(data)
