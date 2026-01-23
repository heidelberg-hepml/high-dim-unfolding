import numpy as np
import os
import torch
from torch_geometric.data import Data

from experiments.logger import LOGGER

EPS = 1e-5


class TaggingDataset(torch.utils.data.Dataset):
    """
    We use torch_geometric to handle point cloud of jet constituents more efficiently
    The torch_geometric dataloader concatenates jets along their constituent direction,
    effectively combining the constituent index with the batch index in a single dimension.
    An extra object batch.batch for each batch specifies to which jet the constituent belongs.
    We extend the constituent list by a global token that is used to embed extra global
    information and extract the classifier score.

    Structure of the elements in self.data_list
    x : torch.tensor of shape (num_elements, 4)
        List of 4-momenta of jet constituents
    scalars : empty placeholder
    label : torch.tensor of shape (1), dtype torch.int
        label of the jet (0=QCD, 1=top)
    is_global : torch.tensor of shape (num_elements), dtype torch.bool
        True for the global token (first element in constituent list), False otherwise
        We set is_global=None if no global token is used
    """

    def __init__(self):
        super().__init__()

    def load_data(self, filename, mode):
        raise NotImplementedError

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class TopTaggingDataset(TaggingDataset):
    def load_data(
        self,
        filename,
        mode,
        network_float64=False,
        momentum_float64=True,
    ):
        """
        Parameters
        ----------
        filename : str
            Path to file in npz format where the dataset in stored
        mode : {"train", "test", "val"}
            Purpose of the dataset
            Train, test and validation datasets are already seperated in the specified file
        network_float64 : bool
        momentum_float64 : bool
        """
        network_dtype = torch.float64 if network_float64 else torch.float32
        momentum_dtype = torch.float64 if momentum_float64 else torch.float32

        data = np.load(filename)
        kinematics = data[f"kinematics_{mode}"]
        labels = data[f"labels_{mode}"]

        kinematics = torch.tensor(kinematics, dtype=momentum_dtype)
        labels = torch.tensor(labels, dtype=torch.bool)

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
                dtype=network_dtype,
            )  # no scalar information
            data = Data(x=fourmomenta, scalars=scalars, label=label)
            self.data_list.append(data)


class ClassificationDataset(TaggingDataset):
    _cached_splits = None
    _cache_key = None

    def load_data(
        self,
        filename,
        mode,
        network_float64=False,
        momentum_float64=True,
        train_test=(0.8, 0.2),
        split_seed=0,
    ):
        """
        Parameters
        ----------
        filename : str
            Base path where torch .pt files live. Files are expected at
            <filename>/truth.pt and <filename>/samples.pt.
        mode : {"train", "test"}
            Purpose of the dataset. Splits are created once and cached.
        train_test : tuple[float, float]
            Fractions for train/test split (must sum to <= 1).
        split_seed : int
            RNG seed for deterministic shuffling before splitting.
        """

        label_paths = self._parse_filenames(filename)

        # cache splits so train/val/test get consistent subsets
        if (
            ClassificationDataset._cached_splits is None
            or ClassificationDataset._cache_key != label_paths
        ):
            ClassificationDataset._cached_splits = self._build_splits(
                label_paths=label_paths,
                split=train_test,
                split_seed=split_seed,
                network_float64=network_float64,
                momentum_float64=momentum_float64,
            )
            ClassificationDataset._cache_key = label_paths

        if mode not in ClassificationDataset._cached_splits:
            raise ValueError(f"Unknown mode {mode}, expected one of {list(ClassificationDataset._cached_splits)}")

        # copy list wrapper so consumers don't accidentally modify the cache
        self.data_list = list(ClassificationDataset._cached_splits[mode])

    @staticmethod
    def _parse_filenames(filename):
        labelled = {
            0: os.path.join(filename, "truth.pt"),
            1: os.path.join(filename, "samples.pt"),
        }

        paths = tuple((label, os.fspath(path)) for label, path in sorted(labelled.items()))
        for _, path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"ClassificationDataset could not find file {path}")
        return paths

    @classmethod
    def _build_splits(cls, label_paths, split, split_seed, network_float64, momentum_float64):
        graphs = []
        for label, path in label_paths:
            batch = torch.load(path, map_location="cpu", weights_only=False)
            data_list = batch.to_data_list()
            label_tensor = torch.tensor([label], dtype=torch.bool)
            for old_graph in data_list:
                x_gen = old_graph.x_gen.to(dtype=
                    torch.float64 if momentum_float64 else torch.float32
                )
                scalars_gen = torch.zeros(x_gen.shape[0], 0, dtype=torch.float64 if network_float64 else torch.float32)
                x_det = old_graph.x_det.to(dtype=
                    torch.float64 if momentum_float64 else torch.float32
                )
                scalars_det = torch.zeros(x_det.shape[0], 0, dtype=torch.float64 if network_float64 else torch.float32)
                new_graph = Data(
                    x_gen=x_gen,
                    scalars_gen=scalars_gen,
                    x_det=x_det,
                    scalars_det=scalars_det,
                    label=label_tensor,
                )
                graphs.append(new_graph)

        if len(graphs) == 0:
            raise ValueError("No graphs loaded from the provided .pt files")

        rng = np.random.default_rng(split_seed)
        rng.shuffle(graphs)

        train_end = int(split[0] * len(graphs))
        splits = {
            "train": graphs[:train_end],
            "test": graphs[train_end:],
        }
        return splits
