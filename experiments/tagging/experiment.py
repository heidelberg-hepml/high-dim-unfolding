import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from experiments.base_experiment import BaseExperiment
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow
from experiments.tagging.dataset import ClassificationDataset, TopTaggingDataset
from experiments.tagging.embedding import embed_tagging_data, get_num_tagging_features
from experiments.tagging.plots import plot_mixer


class TaggingExperiment(BaseExperiment):
    """
    Base class for jet tagging experiments, focusing on binary classification
    """

    def init_physics(self):
        modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]
        if modelname == "ConditionalTransformer":
            modelname = "Transformer"
        self.momentum_dtype = torch.float64 if self.cfg.data.momentum_float64 else torch.float32

        self.cfg.model.out_channels = self.num_outputs
        if modelname in [
            "LGATr",
            "LGATrSlim",
            "LorentzNet",
            "PELICAN",
            "PELICANOfficial",
            "CGENN",
        ]:
            # Lorentz-equivariance by internal representations
            in_s_channels = self.extra_scalars
            in_s_channels += get_num_tagging_features(
                tagging_features=self.cfg.data.tagging_features
            )
            if modelname in ["LGATr", "LGATrSlim"]:
                self.cfg.model.net.in_s_channels = 0 if self.cfg.model.mean_aggregation else 1
                self.cfg.model.net.in_s_channels += in_s_channels
            elif modelname == "LorentzNet":
                self.cfg.model.net.in_s_channels = in_s_channels
            elif modelname == "PELICAN":
                self.cfg.model.net.in_channels_rank1 = in_s_channels
            elif modelname == "PELICANOfficial":
                self.cfg.model.net.num_scalars = in_s_channels
            elif modelname == "CGENN":
                # CGENN cant handle zero scalar inputs -> give 1 input with zeros
                self.cfg.model.net.in_features_h = 1 + in_s_channels

            # doesn't affect results and never needed
            self.cfg.data.boost_jet = False
        elif modelname in [
            "Transformer",
            "ParticleTransformer",
            "GraphNet",
            "ParticleNet",
            "MIParticleTransformer",
        ]:
            # Non-equivariant or canonicalization
            self.cfg.model.in_channels = 4 + self.extra_scalars
            if self.cfg.model.add_fourmomenta_backbone:
                self.cfg.model.in_channels += 4

            if modelname == "Transformer":
                self.cfg.model.in_channels += 0 if self.cfg.model.mean_aggregation else 1
            elif modelname == "GraphNet":
                self.cfg.model.net.num_edge_attr = 1 if self.cfg.model.include_edges else 0
            elif modelname == "ParticleNet":
                self.cfg.model.net.hidden_reps_list[0] = f"{self.cfg.model.in_channels}x0n"

            # decide which entries to use for the framesnet
            if "equivectors" in self.cfg.model.framesnet:
                num_tagging_features = get_num_tagging_features(
                    tagging_features=self.cfg.data.tagging_features
                )
                self.cfg.model.framesnet.equivectors.num_scalars = self.extra_scalars
                self.cfg.model.framesnet.equivectors.num_scalars += num_tagging_features
            else:
                # not allowed, because the network is not Lorentz-equivariant
                self.cfg.data.boost_jet = False
        else:
            raise NotImplementedError(f"Model {modelname} not implemented")

    def init_data(self):
        raise NotImplementedError

    def _init_data(self, Dataset, data_path):
        LOGGER.info(f"Creating {Dataset.__name__} from {data_path}")
        t0 = time.time()
        self.data_train = Dataset()
        self.data_test = Dataset()
        self.data_val = Dataset()
        kwargs = dict(
            network_float64=self.cfg.use_float64,
            momentum_float64=self.cfg.data.momentum_float64,
        )
        if hasattr(self.cfg.data, "train_val_test"):
            kwargs["train_val_test"] = tuple(self.cfg.data.train_val_test)
        if hasattr(self.cfg.data, "split_seed"):
            kwargs["split_seed"] = self.cfg.data.split_seed
        self.data_train.load_data(data_path, "train", **kwargs)
        self.data_test.load_data(data_path, "test", **kwargs)
        self.data_val.load_data(data_path, "val", **kwargs)
        dt = time.time() - t0
        LOGGER.info(f"Finished creating datasets after {dt:.2f} s = {dt / 60:.2f} min")

    def _init_dataloader(self):
        trn_sampler = torch.utils.data.DistributedSampler(
            self.data_train,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )
        tst_sampler = torch.utils.data.DistributedSampler(
            self.data_test,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )
        val_sampler = torch.utils.data.DistributedSampler(
            self.data_val,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )

        self.train_loader = DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.training.batchsize // self.world_size,
            sampler=trn_sampler,
        )
        self.test_loader = DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            sampler=tst_sampler,
        )
        self.val_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            sampler=val_sampler,
        )

        LOGGER.info(
            f"Constructed dataloaders with "
            f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )

        self.init_standardization()

    def init_standardization(self):
        if hasattr(self.model, "init_standardization"):
            batch = next(iter(self.train_loader))
            fourmomenta, scalars, ptr, _ = self._extract_batch(batch)
            embedding = embed_tagging_data(
                fourmomenta,
                scalars,
                ptr,
                self.cfg.data,
            )
            self.model.init_standardization(embedding["fourmomenta"], embedding["ptr"])

    def evaluate(self):
        self.results = {}
        loader_dict = {
            "train": self.train_loader,
            "test": self.test_loader,
            "val": self.val_loader,
        }
        for set_label in self.cfg.evaluation.eval_set:
            if self.ema is not None:
                with self.ema.average_parameters():
                    self.results[set_label] = self._evaluate_single(
                        loader_dict[set_label], f"{set_label}_ema", mode="eval"
                    )

                self._evaluate_single(loader_dict[set_label], set_label, mode="eval")

            else:
                self.results[set_label] = self._evaluate_single(
                    loader_dict[set_label], set_label, mode="eval"
                )

    @torch.no_grad()
    def _evaluate_single(self, loader, title, mode, step=None):
        assert mode in ["val", "eval"]

        if mode == "eval":
            LOGGER.info(
                f"### Starting to evaluate model on {title} dataset with "
                f"{len(loader.dataset)} elements, batchsize {loader.batch_size} ###"
            )
        metrics = {}

        # predictions
        labels_true, labels_predict = [], []
        self.model.eval()
        for batch in loader:
            y_pred, label, _, _ = self._get_ypred_and_label(batch)
            labels_true.append(label.cpu().float())
            labels_predict.append(y_pred.cpu().float())
        labels_true, labels_predict = torch.cat(labels_true), torch.cat(labels_predict)

        # bce loss
        metrics["loss"] = torch.nn.functional.binary_cross_entropy_with_logits(
            labels_predict, labels_true
        ).item()
        labels_predict = torch.nn.functional.sigmoid(labels_predict)
        labels_true, labels_predict = labels_true.numpy(), labels_predict.numpy()

        if mode == "eval":
            metrics["labels_true"], metrics["labels_predict"] = (
                labels_true,
                labels_predict,
            )

        # accuracy
        metrics["accuracy"] = accuracy_score(labels_true, np.round(labels_predict))
        if mode == "eval":
            LOGGER.info(f"Accuracy on {title} dataset: {metrics['accuracy']:.4f}")

        # roc (fpr = epsB, tpr = epsS)
        fpr, tpr, th = roc_curve(labels_true, labels_predict)
        if mode == "eval":
            metrics["fpr"], metrics["tpr"] = fpr, tpr
        metrics["auc"] = roc_auc_score(labels_true, labels_predict)
        if mode == "eval":
            LOGGER.info(f"AUC score on {title} dataset: {metrics['auc']:.4f}")

        # 1/epsB at fixed epsS
        def get_rej(epsS):
            idx = np.argmin(np.abs(tpr - epsS))
            return 1 / fpr[idx]

        metrics["rej03"] = get_rej(0.3)
        metrics["rej05"] = get_rej(0.5)
        metrics["rej08"] = get_rej(0.8)
        if mode == "eval":
            LOGGER.info(
                f"Rejection rate {title} dataset: {metrics['rej03']:.0f} (epsS=0.3), "
                f"{metrics['rej05']:.0f} (epsS=0.5), {metrics['rej08']:.0f} (epsS=0.8)"
            )

        if self.cfg.use_mlflow:
            for key, value in metrics.items():
                if key in ["labels_true", "labels_predict", "fpr", "tpr"]:
                    # do not log matrices
                    continue
                name = f"{mode}.{title}" if mode == "eval" else "val"
                log_mlflow(f"{name}.{key}", value, step=step)

        if mode == "eval":
            framesString = type(self.model.framesnet).__name__
            num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            LOGGER.info(
                f"table {title}: {framesString} ({self.cfg.training.iterations} iterations)"
                f" & {num_parameters} & {metrics['accuracy']:.4f} & {metrics['auc']:.4f}"
                f" & {metrics['rej03']:.0f} & {metrics['rej05']:.0f} & {metrics['rej08']:.0f} \\\\"
            )
        return metrics

    def plot(self):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path, exist_ok=True)
        title = type(self.model.net).__name__
        LOGGER.info(f"Creating plots in {plot_path}")

        if (
            self.cfg.evaluation.save_roc
            and self.cfg.evaluate
            and ("test" in self.cfg.evaluation.eval_set)
        ):
            file = f"{plot_path}/roc.txt"
            roc = np.stack((self.results["test"]["fpr"], self.results["test"]["tpr"]), axis=-1)
            np.savetxt(file, roc)

        plot_dict = {}
        if self.cfg.evaluate:
            if "test" in self.cfg.evaluation.eval_set:
                plot_dict = {"results_test": self.results["test"]}
            elif "val" in self.cfg.evaluation.eval_set:
                plot_dict = {"results_test": self.results["val"]}
        if self.cfg.train:
            plot_dict["train_loss"] = self.train_loss
            plot_dict["val_loss"] = self.val_loss
            plot_dict["train_lr"] = self.train_lr
            plot_dict["grad_norm"] = torch.tensor(self.train_grad_norm).cpu()
            for key, value in self.train_metrics.items():
                plot_dict[key] = value
        plot_mixer(self.cfg, plot_path, title, plot_dict)

    def _init_loss(self):
        self.loss = torch.nn.BCEWithLogitsLoss()

    # overwrite _validate method to compute metrics over the full validation set
    def _validate(self, step):
        if self.ema is not None:
            with self.ema.average_parameters():
                metrics = self._evaluate_single(self.val_loader, "val", mode="val", step=step)
        else:
            metrics = self._evaluate_single(self.val_loader, "val", mode="val", step=step)
        self.val_loss.append(metrics["loss"])
        return metrics["loss"]

    def _batch_loss(self, batch):
        y_pred, label, tracker, _ = self._get_ypred_and_label(batch)
        loss = self.loss(y_pred, label)

        metrics = tracker
        return loss, metrics

    def _extract_batch(self, batch):
        batch = batch.to(self.device)
        fourmomenta = batch.x.to(self.momentum_dtype)
        scalars = batch.scalars.to(self.dtype)
        ptr = batch.ptr
        label = batch.label.to(self.dtype)
        return fourmomenta, scalars, ptr, label

    def _get_ypred_and_label(self, batch):
        fourmomenta, scalars, ptr, label = self._extract_batch(batch)
        embedding = embed_tagging_data(
            fourmomenta,
            scalars,
            ptr,
            self.cfg.data,
        )
        embedding["num_graphs"] = label.shape[0]
        y_pred, tracker, frames = self.model(embedding)
        if isinstance(self.loss, torch.nn.BCEWithLogitsLoss):
            y_pred = y_pred[:, 0]

        return y_pred, label, tracker, frames

    def _init_metrics(self):
        return {
            "reg_collinear": [],
            "reg_coplanar": [],
            "reg_lightlike": [],
            "reg_gammamax": [],
            "gamma_mean": [],
            "gamma_max": [],
        }


class TopTaggingExperiment(TaggingExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_outputs = 1
        self.extra_scalars = 0

    def init_data(self):
        data_path = os.path.join(self.cfg.data.data_dir, f"toptagging_{self.cfg.data.dataset}.npz")
        self._init_data(TopTaggingDataset, data_path)


class ClassificationExperiment(TaggingExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_outputs = 1
        self.extra_scalars = get_num_tagging_features(
            tagging_features=self.cfg.data.tagging_features
        )

    def init_data(self):
        data_path = self.cfg.data.data_path
        train_path = getattr(self.cfg.data, "train_data_path", None) or data_path
        val_path = getattr(self.cfg.data, "val_data_path", None) or data_path
        test_path = getattr(self.cfg.data, "test_data_path", None) or data_path

        LOGGER.info(
            f"Creating {ClassificationDataset.__name__}: "
            f"train={train_path}, val={val_path}, test={test_path}"
        )
        t0 = time.time()

        base_kwargs = dict(
            network_float64=self.cfg.use_float64,
            momentum_float64=self.cfg.data.momentum_float64,
            train_val_test=tuple(self.cfg.data.train_val_test),
            split_seed=getattr(self.cfg.data, "split_seed", 0),
        )

        # Group modes by path so each unique file is loaded only once.
        # A path used exclusively for one split gets all its data assigned to
        # that split; a path shared by multiple splits uses the configured ratio.
        _all_data = {
            "train": (1.0, 0.0, 0.0),
            "val": (0.0, 1.0, 0.0),
            "test": (0.0, 0.0, 1.0),
        }
        path_to_modes: dict = {}
        for mode, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
            path_to_modes.setdefault(path, []).append(mode)

        split_results = {}
        for path, modes in path_to_modes.items():
            ratio = base_kwargs["train_val_test"] if len(modes) > 1 else _all_data[modes[0]]
            label_paths = ClassificationDataset._parse_filenames(path)
            splits = ClassificationDataset._build_splits(
                label_paths=label_paths,
                split=ratio,
                split_seed=base_kwargs["split_seed"],
                network_float64=base_kwargs["network_float64"],
                momentum_float64=base_kwargs["momentum_float64"],
            )
            for mode in modes:
                split_results[mode] = splits[mode]

        self.data_train = ClassificationDataset()
        self.data_train.load_from_list(split_results["train"])
        self.data_val = ClassificationDataset()
        self.data_val.load_from_list(split_results["val"])
        self.data_test = ClassificationDataset()
        self.data_test.load_from_list(split_results["test"])

        dt = time.time() - t0
        LOGGER.info(f"Finished creating datasets after {dt:.2f} s = {dt / 60:.2f} min")

    def _init_dataloader(self):
        trn_sampler = torch.utils.data.DistributedSampler(
            self.data_train, num_replicas=self.world_size, rank=self.rank, shuffle=True
        )
        tst_sampler = torch.utils.data.DistributedSampler(
            self.data_test, num_replicas=self.world_size, rank=self.rank, shuffle=False
        )
        val_sampler = torch.utils.data.DistributedSampler(
            self.data_val, num_replicas=self.world_size, rank=self.rank, shuffle=False
        )
        fb = ["x_gen", "x_det"]
        self.train_loader = DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.training.batchsize // self.world_size,
            sampler=trn_sampler,
            follow_batch=fb,
        )
        self.test_loader = DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            sampler=tst_sampler,
            follow_batch=fb,
        )
        self.val_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            sampler=val_sampler,
            follow_batch=fb,
        )
        LOGGER.info(
            f"Constructed dataloaders with "
            f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, "
            f"val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), "
            f"{self.cfg.evaluation.batchsize} (evaluation)"
        )
        self.init_standardization()

    def _extract_batch(self, batch):
        batch = batch.to(self.device)
        fourmomenta = batch.x_gen.to(self.momentum_dtype)
        scalars = batch.scalars_gen.to(self.dtype)
        ptr = batch.x_gen_ptr
        label = batch.label.to(self.dtype)
        det_fourmomenta = batch.x_det.to(self.momentum_dtype)
        det_scalars = batch.scalars_det.to(self.dtype)
        det_ptr = batch.x_det_ptr
        return (fourmomenta, scalars, ptr), (det_fourmomenta, det_scalars, det_ptr), label

    def _get_ypred_and_label(self, batch):
        gen_args, det_args, label = self._extract_batch(batch)
        gen_embedding = embed_tagging_data(
            *gen_args,
            self.cfg.data,
        )
        gen_embedding["num_graphs"] = label.shape[0]
        det_embedding = embed_tagging_data(
            *det_args,
            self.cfg.data,
        )
        det_embedding["num_graphs"] = label.shape[0]
        embedding = {
            "gen": gen_embedding,
            "det": det_embedding,
        }
        y_pred, tracker, frames = self.model(embedding)
        if isinstance(self.loss, torch.nn.BCEWithLogitsLoss):
            y_pred = y_pred[:, 0]

        return y_pred, label, tracker, frames

    @torch.no_grad()
    def _evaluate_single(self, loader, title, mode, step=None):
        assert mode in ["val", "eval"]

        if mode == "eval":
            LOGGER.info(
                f"### Starting to evaluate model on {title} dataset with "
                f"{len(loader.dataset)} elements, batchsize {loader.batch_size} ###"
            )
            if self.cfg.evaluation.save_predictions:
                data_list = []
        metrics = {}

        # predictions
        labels_true, labels_predict = [], []
        self.model.eval()
        for batch in loader:
            y_pred, label, _, _ = self._get_ypred_and_label(batch)
            labels_true.append(label.cpu().float())
            labels_predict.append(y_pred.cpu().float())
            if mode == "eval" and self.cfg.evaluation.save_predictions:
                batch.weight = torch.nn.functional.sigmoid(y_pred).cpu().float()
                data_list.extend(batch.to_data_list())

        if mode == "eval" and self.cfg.evaluation.save_predictions:
            full_batch = Batch.from_data_list(data_list, follow_batch=["x_gen", "x_det"])
            path = os.path.join(self.cfg.run_dir, f"predictions_{self.cfg.run_idx}")
            os.makedirs(path, exist_ok=True)
            LOGGER.info(f"Saving samples in {path}")
            torch.save(full_batch, os.path.join(path, "weighted_events.pt"))

        labels_true, labels_predict = torch.cat(labels_true), torch.cat(labels_predict)

        # bce loss
        metrics["loss"] = torch.nn.functional.binary_cross_entropy_with_logits(
            labels_predict, labels_true
        ).item()
        labels_predict = torch.nn.functional.sigmoid(labels_predict)
        labels_true, labels_predict = labels_true.numpy(), labels_predict.numpy()

        if mode == "eval":
            metrics["labels_true"], metrics["labels_predict"] = (
                labels_true,
                labels_predict,
            )

        # accuracy
        metrics["accuracy"] = accuracy_score(labels_true, np.round(labels_predict))
        if mode == "eval":
            LOGGER.info(f"Accuracy on {title} dataset: {metrics['accuracy']:.4f}")

        # roc (fpr = epsB, tpr = epsS)
        fpr, tpr, th = roc_curve(labels_true, labels_predict)
        if mode == "eval":
            metrics["fpr"], metrics["tpr"] = fpr, tpr
        metrics["auc"] = roc_auc_score(labels_true, labels_predict)
        if mode == "eval":
            LOGGER.info(f"AUC score on {title} dataset: {metrics['auc']:.4f}")

        # 1/epsB at fixed epsS
        def get_rej(epsS):
            idx = np.argmin(np.abs(tpr - epsS))
            return 1 / fpr[idx]

        metrics["rej03"] = get_rej(0.3)
        metrics["rej05"] = get_rej(0.5)
        metrics["rej08"] = get_rej(0.8)
        if mode == "eval":
            LOGGER.info(
                f"Rejection rate {title} dataset: {metrics['rej03']:.0f} (epsS=0.3), "
                f"{metrics['rej05']:.0f} (epsS=0.5), {metrics['rej08']:.0f} (epsS=0.8)"
            )

        if self.cfg.use_mlflow:
            for key, value in metrics.items():
                if key in ["labels_true", "labels_predict", "fpr", "tpr"]:
                    # do not log matrices
                    continue
                name = f"{mode}.{title}" if mode == "eval" else "val"
                log_mlflow(f"{name}.{key}", value, step=step)

        if mode == "eval":
            framesString = type(self.model.framesnet).__name__
            num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            LOGGER.info(
                f"table {title}: {framesString} ({self.cfg.training.iterations} iterations)"
                f" & {num_parameters} & {metrics['accuracy']:.4f} & {metrics['auc']:.4f}"
                f" & {metrics['rej03']:.0f} & {metrics['rej05']:.0f} & {metrics['rej08']:.0f} \\\\"
            )
        return metrics
