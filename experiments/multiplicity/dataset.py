import energyflow
import torch
import numpy as np
from torch_geometric.data import Data

from experiments.multiplicity.utils import ensure_angle, jetmomenta_to_fourmomenta


EPS = 1e-5


class MultiplicityDataset:

    def __init__(self, data_path, cfg):
        self.cfg = cfg
        self.data = energyflow.zjets_delphes.load(
            "Herwig",
            num_data=self.cfg.data.length,
            pad=True,
            cache_dir=data_path,
            include_keys=["particles", "mults", "jets"],
        )
        self.standardize = {
            "mean": torch.zeros(1, 4),
            "std": torch.ones(1, 4),
        }

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def load_data(
        self,
        data,
        mode,
        model,
        split=[0.8, 0.1, 0.1],
        mass=0.1,
        standardize=True,
        dtype=torch.float32,
    ):
        """
        Parameters
        ----------
        filename : str
            Path to file in npz format where the dataset in stored
        mode : {"train", "test", "val"}
            Purpose of the dataset
        model : {"Transformer", "GATr"}
            Model used
        split : list of float
            Fraction of data to use for training, testing and validation
        mass : float
            Mass of the particle to reconstruct
        standardize : bool
            Whether to standardize the data
            if model == GATr, standardization on fourmomenta with the same factors for all coordinates
            if model == Transformer, standardization on (Pt,Phi,Eta)
        dtype : torch.dtype
            Data type of the tensors
        """
        size = len(data["sim_particles"])

        if mode == "train":
            det_particles = torch.tensor(data["sim_particles"])[
                : int(split[0] * size)
            ]  # detector-level particles
            det_jets = torch.tensor(data["sim_jets"])[
                : int(split[0] * size)
            ]  # detector-level jets
            det_mults = torch.tensor(data["sim_mults"], dtype=int)[
                : int(split[0] * size)
            ]  # detector-level multiplicity
            gen_mults = torch.tensor(data["gen_mults"], dtype=int)[
                : int(split[0] * size)
            ]  # genlevel multiplicity

            mask = torch.arange(det_particles.shape[1])[None, :] < det_mults[:, None]
            flattened_particles = det_particles[mask]
            flattened_particles[..., [1, 2]] = flattened_particles[..., [2, 1]]

            if model == "GATr":
                flattened_particles[..., 3] = mass
                flattened_particles = jetmomenta_to_fourmomenta(flattened_particles)
                self.standardize["mean"] = flattened_particles.mean().view(1, 4)
                self.standardize["std"] = flattened_particles.std().view(1, 4)
            else:
                self.standardize["mean"][..., :3] = flattened_particles.mean(
                    dim=0, keepdim=True
                )[..., :3]
                self.standardize["std"][..., :3] = flattened_particles.std(
                    dim=0, keepdim=True
                )[..., :3]

        elif mode == "val":
            det_particles = torch.tensor(data["sim_particles"])[
                int(split[0] * size) : int((split[0] + split[1]) * size)
            ]
            det_jets = torch.tensor(data["sim_jets"])[
                int(split[0] * size) : int((split[0] + split[1]) * size)
            ]
            det_mults = torch.tensor(data["sim_mults"], dtype=int)[
                int(split[0] * size) : int((split[0] + split[1]) * size)
            ]
            gen_mults = torch.tensor(data["gen_mults"], dtype=int)[
                int(split[0] * size) : int((split[0] + split[1]) * size)
            ]
        elif mode == "test":
            det_particles = torch.tensor(data["sim_particles"])[
                int((split[0] + split[1]) * size) :
            ]
            det_jets = torch.tensor(data["sim_jets"])[
                int((split[0] + split[1]) * size) :
            ]
            det_mults = torch.tensor(data["sim_mults"], dtype=int)[
                int((split[0] + split[1]) * size) :
            ]
            gen_mults = torch.tensor(data["gen_mults"], dtype=int)[
                int((split[0] + split[1]) * size) :
            ]
        else:
            raise ValueError("Mode must be one of {'train', 'test', 'val'}")

        # undo the dataset scaling
        det_particles[..., 1:3] = det_particles[..., 1:3] + det_jets[:, np.newaxis, 1:3]
        det_particles[..., 2] = ensure_angle(det_particles[..., 2])
        det_particles[..., [1, 2]] = det_particles[..., [2, 1]]  # swap eta and phi

        kinematics = torch.tensor(
            det_particles, dtype=dtype
        )  # contains p_T, phi, eta, PID
        labels = torch.tensor(gen_mults, dtype=torch.int)

        # create list of torch_geometric.data.Data objects
        self.data_list = []
        for i in range(kinematics.shape[0]):
            # extract PID
            scalars = kinematics[i, : det_mults[i], -1].clone().unsqueeze(-1)

            # store standardized pt, phi, eta, mass
            fourvector = kinematics[i, : det_mults[i]]
            fourvector[..., -1] = mass  # replace PID by constant mass for all particles

            if model == "GATr":
                fourvector = jetmomenta_to_fourmomenta(fourvector)

            if standardize:
                fourvector = (fourvector - self.standardize["mean"]) / self.standardize[
                    "std"
                ]

            label = labels[i]

            data = Data(
                x=fourvector, scalars=scalars, label=label, det_mult=det_mults[i]
            )
            self.data_list.append(data)
