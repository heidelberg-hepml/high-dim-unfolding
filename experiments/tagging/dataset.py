import os

import numpy as np
import torch
from torch_geometric.data import Data

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


class ClassificationDataset(TaggingDataset):
    _cached_splits = None
    _cache_key = None

    def load_data(
        self,
        filename,
        mode,
        dtype = torch.float32,
        train_val_test=(0.8, 0.1, 0.1),
        split_seed=0,
    ):
        """
        Parameters
        ----------
        filename : tuple[str, str] | list[str] | dict
            Two torch .pt files containing batches of graphs, one per class.
            Dict keys are interpreted as labels (0/1), otherwise order is [label 0, label 1].
        mode : {"train", "test", "val"}
            Purpose of the dataset. Splits are created once and cached.
        train_val_test : tuple[float, float, float]
            Fractions for train/val/test split (must sum to <= 1).
        split_seed : int
            RNG seed for deterministic shuffling before splitting.
        """

        label_paths = self._parse_filenames(filename)
        split = self._sanitize_split(train_val_test)

        # cache splits so train/val/test get consistent subsets
        if (
            ClassificationDataset._cached_splits is None
            or ClassificationDataset._cache_key != label_paths
        ):
            ClassificationDataset._cached_splits = self._build_splits(
                label_paths=label_paths,
                split=split,
                split_seed=split_seed,
                dtype=dtype,
            )
            ClassificationDataset._cache_key = label_paths

        if mode not in ClassificationDataset._cached_splits:
            raise ValueError(f"Unknown mode {mode}, expected one of {list(ClassificationDataset._cached_splits)}")

        # copy list wrapper so consumers don't accidentally modify the cache
        self.data_list = list(ClassificationDataset._cached_splits[mode])

    @staticmethod
    def _parse_filenames(filename):
        if isinstance(filename, (list, tuple)):
            if len(filename) != 2:
                raise ValueError("ClassificationDataset expects exactly two .pt files when filename is a list/tuple")
            labelled = {0: filename[0], 1: filename[1]}
        elif isinstance(filename, dict):
            labelled = {}
            for key, path in filename.items():
                try:
                    label = int(key)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Dictionary keys for filename must be 0 or 1, got {key}") from exc
                labelled[label] = path
        else:
            raise TypeError(
                "filename must be a tuple/list of two .pt files or a dict mapping labels {0,1} to paths"
            )

        if set(labelled.keys()) != {0, 1}:
            raise ValueError(f"Expected labels {{0,1}}, got {sorted(labelled.keys())}")

        paths = tuple((label, os.fspath(path)) for label, path in sorted(labelled.items()))
        for _, path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"ClassificationDataset could not find file {path}")
        return paths

    @staticmethod
    def _sanitize_split(split):
        if len(split) != 3:
            raise ValueError(f"train_val_test must have three entries, got {split}")
        if any(s < 0 for s in split):
            raise ValueError(f"train_val_test entries must be non-negative, got {split}")
        if sum(split) > 1.0 + 1e-6:
            raise ValueError(f"train_val_test must sum to <= 1.0, got {split} (sum={sum(split):.3f})")
        return tuple(float(s) for s in split)

    @staticmethod
    def _ensure_scalars(data, network_dtype):
        scalars = getattr(data, "scalars", None)
        if scalars is None:
            scalars = torch.zeros(data.x.shape[0], 0, dtype=network_dtype)
        else:
            scalars = scalars.to(network_dtype)
        return scalars

    @classmethod
    def _build_splits(cls, label_paths, split, split_seed, dtype):
        graphs = []
        for label, path in label_paths:
            batch = torch.load(path, map_location="cpu", weights_only=False)
            data_list = batch.to_data_list()
            label_tensor = torch.tensor([label], dtype=torch.bool)
            for old_graph in data_list:
                x = old_graph.x_gen.to(dtype)
                scalars = torch.zeros(x.shape[0], 0, dtype=dtype)
                new_graph = Data(
                    x=x,
                    scalars=scalars,
                    label=label_tensor,
                )
                graphs.append(new_graph)

        if len(graphs) == 0:
            raise ValueError("No graphs loaded from the provided .pt files")

        rng = np.random.default_rng(split_seed)
        rng.shuffle(graphs)

        train_end = int(split[0] * len(graphs))
        val_end = train_end + int(split[1] * len(graphs))
        splits = {
            "train": graphs[:train_end],
            "val": graphs[train_end:val_end],
            "test": graphs[val_end:],
        }
        return splits
