import torch
import numpy as np
import energyflow
from torch_geometric.data import Data, Batch
from experiments.logger import LOGGER
from experiments.eventgen.helpers import jetmomenta_to_fourmomenta

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


class UnfoldingDataset(BaseDataset):
    def load_data(
        self,
        data,
        mode,
        split=[0.8, 0.1, 0.1],
        mass=0.1,
        save_pid=False,
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
            sim_particles = np.array(data["sim_particles"])[: int(split[0] * size)]
            gen_particles = np.array(data["gen_particles"])[: int(split[0] * size)]
            sim_mults = np.array(data["sim_mults"], dtype=int)[: int(split[0] * size)]
            gen_mults = np.array(data["gen_mults"], dtype=int)[: int(split[0] * size)]
        elif mode == "val":
            sim_particles = np.array(data["sim_particles"])[
                int(split[0] * size) : int(split[0] * size) + int(split[1] * size)
            ]
            gen_particles = np.array(data["gen_particles"])[
                int(split[0] * size) : int(split[0] * size) + int(split[1] * size)
            ]
            sim_mults = np.array(data["sim_mults"], dtype=int)[
                int(split[0] * size) : int(split[0] * size) + int(split[1] * size)
            ]
            gen_mults = np.array(data["gen_mults"], dtype=int)[
                int(split[0] * size) : int(split[0] * size) + int(split[1] * size)
            ]
        else:
            sim_particles = np.array(data["sim_particles"])[
                int((split[0] + split[1]) * size) :
            ]
            gen_particles = np.array(data["gen_particles"])[
                int((split[0] + split[1]) * size) :
            ]
            sim_mults = np.array(data["sim_mults"], dtype=int)[
                int((split[0] + split[1]) * size) :
            ]
            gen_mults = np.array(data["gen_mults"], dtype=int)[
                int((split[0] + split[1]) * size) :
            ]

        sim_kinematics = torch.tensor(sim_particles, dtype=dtype)
        gen_kinematics = torch.tensor(gen_particles, dtype=dtype)

        # create list of torch_geometric.data.Data objects
        self.data_list = []
        for i in range(sim_kinematics.shape[0]):
            if save_pid:
                sim_scalars = (
                    sim_kinematics[i, : sim_mults[i], -1].clone().unsqueeze(-1)
                )  # store PID as scalar info
                gen_scalars = (
                    gen_kinematics[i, : gen_mults[i], -1].clone().unsqueeze(-1)
                )  # store PID as scalar info
            else:
                sim_scalars = torch.zeros((sim_mults[i], 0))
                gen_scalars = torch.zeros((gen_mults[i], 0))

            sim_fourmomenta = sim_kinematics[i, : sim_mults[i]]
            sim_fourmomenta[..., -1] = mass  # set constant mass for all fourmomenta
            sim_fourmomenta = jetmomenta_to_fourmomenta(sim_fourmomenta)

            gen_fourmomenta = gen_kinematics[i, : gen_mults[i]]
            gen_fourmomenta[..., -1] = mass  # set constant mass for all fourmomenta
            gen_fourmomenta = jetmomenta_to_fourmomenta(gen_fourmomenta)

            data_sim = Data(
                x=sim_fourmomenta,
                scalars=sim_scalars,
            )
            data_gen = Data(
                x=gen_fourmomenta,
                scalars=gen_scalars,
            )
            self.data_list.append((data_sim, data_gen))


def collate(data_list):
    batch_sim = Batch.from_data_list([data[0] for data in data_list])
    batch_gen = Batch.from_data_list([data[1] for data in data_list])
    return batch_sim, batch_gen
