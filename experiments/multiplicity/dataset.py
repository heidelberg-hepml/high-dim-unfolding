import torch
import numpy as np
from torch_geometric.data import Data
from experiments.tagging import TaggingDataset

EPS = 1e-5


class MultiplicityDataset(TaggingDataset):
    def load_data(
        self,
        filename,
        mode,
        dtype=torch.float32,
    ):
        """
        Parameters
        ----------
        filename : str
            Path to file in npz format where the dataset in stored
        mode : {"train", "test", "val"}
            Purpose of the dataset
            Train, test and validation datasets are already seperated in the specified file
        """
        data = np.load(filename)
        kinematics = data[f"kinematics_{mode}"]
        labels = data[f"labels_{mode}"]

        kinematics = torch.tensor(kinematics, dtype=dtype)
        labels = torch.tensor(labels, dtype=torch.int)

        # create list of torch_geometric.data.Data objects
        self.data_list = []
        for i in range(kinematics.shape[0]):
            # drop zero-padded components
            mask = (kinematics[i, ...].abs() > EPS).all(dim=-1)
            fourmomenta = kinematics[i, ...][mask]
            label = labels[i, ...]
            scalars = torch.zeros(
                fourmomenta.shape[0],
                0,
                dtype=dtype,
            )  # no scalar information
            data = Data(x=fourmomenta, scalars=scalars, label=label)
            self.data_list.append(data)


class QGMultiplicityDataset(MultiplicityDataset):
    def load_data(
        self,
        filename,
        mode,
        dtype=torch.float32,
    ):
        """
        Parameters
        ----------
        filename : str
            Path to file in npz format where the dataset in stored
        mode : {"train", "test", "val"}
            Purpose of the dataset
            Train, test and validation datasets are already seperated in the specified file
        """
        data = np.load(filename)
        kinematics = data[f"kinematics_{mode}"]
        pids = data[f"pid_{mode}"]
        labels = data[f"labels_{mode}"]

        kinematics = torch.tensor(kinematics, dtype=dtype)
        pids = torch.tensor(pids, dtype=dtype)
        labels = torch.tensor(labels, dtype=torch.bool)

        # create list of torch_geometric.data.Data objects
        self.data_list = []
        for i in range(kinematics.shape[0]):
            # drop zero-padded components
            mask = (kinematics[i, ...].abs() > EPS).all(dim=-1)
            fourmomenta = kinematics[i, ...][mask]
            scalars = pids[i, ...][mask]  # PID scalar information
            label = labels[i, ...]
            data = Data(x=fourmomenta, scalars=scalars, label=label)
            self.data_list.append(data)
