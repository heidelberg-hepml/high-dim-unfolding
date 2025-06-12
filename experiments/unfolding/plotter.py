import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages


from experiments.base_plots import plot_loss, plot_metric
from experiments.unfolding.plots import (
    plot_histogram,
    plot_calibration,
    simple_histogram,
    plot_roc,
    plot_correlations,
)
from experiments.utils import get_range, fourmomenta_to_jetmomenta
from experiments.logger import LOGGER


def plot_losses(exp, filename, model_label):
    with PdfPages(filename) as file:
        plot_loss(
            file,
            [exp.train_loss, exp.val_loss],
            exp.train_lr,
            labels=["train loss", "val loss"],
            logy=True,
        )
        plot_metric(
            file,
            [exp.train_grad_norm],
            "Gradient norm",
            logy=True,
        )
        for k in range(4):
            plot_loss(
                file,
                [
                    exp.train_metrics[f"mse_{k}"],
                    exp.val_metrics[f"mse_{k}"],
                ],
                lr=exp.train_lr,
                labels=[f"train mse_{k}", f"val mse_{k}"],
                logy=True,
            )


def plot_fourmomenta(exp, filename, model_label, weights=None, mask_dict=None):

    with PdfPages(filename) as file:
        for name in exp.obs_coords.keys():
            extract = exp.obs_coords[name]
            det_lvl = (
                extract(
                    exp.data_raw["samples"].x_det,
                    exp.data_raw["samples"].x_det_batch,
                    exp.data_raw["samples"].x_gen_batch,
                )
                .cpu()
                .detach()
            )
            part_lvl = (
                extract(
                    exp.data_raw["truth"].x_gen,
                    exp.data_raw["truth"].x_gen_batch,
                    exp.data_raw["truth"].x_det_batch,
                )[: len(det_lvl)]
                .cpu()
                .detach()
            )
            model = (
                extract(
                    exp.data_raw["samples"].x_gen,
                    exp.data_raw["samples"].x_gen_batch,
                    exp.data_raw["samples"].x_det_batch,
                )
                .cpu()
                .detach()
            )
            obs_names = [
                "E_{" + name + "}",
                "p_{x," + name + "}",
                "p_{y," + name + "}",
                "p_{z," + name + "}",
            ]

            for channel in range(4):
                xlabel = obs_names[channel]
                xrange = np.array(
                    get_range(
                        [
                            part_lvl[..., channel],
                            # det_lvl[..., channel],
                            model[..., channel],
                        ]
                    )
                )
                logy = True
                plot_histogram(
                    file=file,
                    train=part_lvl[..., channel],
                    test=det_lvl[..., channel],
                    model=model[..., channel],
                    title=exp.plot_title,
                    xlabel=xlabel,
                    xrange=xrange,
                    logy=logy,
                    model_label=model_label,
                    weights=weights,
                    mask_dict=mask_dict,
                )
                if exp.cfg.plotting.correlations:
                    plot_correlations(
                        file=file,
                        det=det_lvl[..., channel],
                        part=part_lvl[..., channel],
                        gen=model[..., channel],
                        title=exp.plot_title,
                        label=xlabel,
                        range=xrange,
                        model_label=model_label,
                    )


def plot_jetmomenta(exp, filename, model_label, weights=None, mask_dict=None):

    with PdfPages(filename) as file:
        for name in exp.obs_coords.keys():
            extract = exp.obs_coords[name]
            det_lvl = extract(
                exp.data_raw["samples"].x_det,
                exp.data_raw["samples"].x_det_batch,
                exp.data_raw["samples"].x_gen_batch,
            )
            part_lvl = extract(
                exp.data_raw["truth"].x_gen,
                exp.data_raw["truth"].x_gen_batch,
                exp.data_raw["truth"].x_det_batch,
            )[: len(det_lvl)]
            model = extract(
                exp.data_raw["samples"].x_gen,
                exp.data_raw["samples"].x_gen_batch,
                exp.data_raw["samples"].x_det_batch,
            )
            part_lvl = fourmomenta_to_jetmomenta(part_lvl).cpu().detach()
            det_lvl = fourmomenta_to_jetmomenta(det_lvl).cpu().detach()
            model = fourmomenta_to_jetmomenta(model).cpu().detach()
            obs_names = [
                r"p_{T," + name + "}",
                "\phi_{" + name + "}",
                "\eta_{" + name + "}",
                "m_{" + name + "}",
            ]
            for channel in range(4):
                xlabel = obs_names[channel]
                xrange = np.array(
                    get_range(
                        [
                            part_lvl[..., channel],
                            # det_lvl[..., channel],
                            model[..., channel],
                        ]
                    )
                )
                if channel == 0:
                    logy = True
                else:
                    logy = False
                plot_histogram(
                    file=file,
                    train=part_lvl[..., channel],
                    test=det_lvl[..., channel],
                    model=model[..., channel],
                    title=exp.plot_title,
                    xlabel=xlabel,
                    xrange=xrange,
                    logy=logy,
                    model_label=model_label,
                    weights=weights,
                    mask_dict=mask_dict,
                )
                if exp.cfg.plotting.correlations:
                    plot_correlations(
                        file=file,
                        det=det_lvl[..., channel],
                        part=part_lvl[..., channel],
                        gen=model[..., channel],
                        title=exp.plot_title,
                        label=xlabel,
                        range=xrange,
                        model_label=model_label,
                    )


def plot_preprocessed(exp, filename, model_label, weights=None, mask_dict=None):

    coords = exp.model.coordinates
    det_lvl_coords = exp.model.condition_coordinates

    with PdfPages(filename) as file:
        for name in exp.obs_coords.keys():
            extract = exp.obs_coords[name]
            det_lvl = extract(
                exp.data_raw["samples"].x_det,
                exp.data_raw["samples"].x_det_batch,
                exp.data_raw["samples"].x_gen_batch,
            )
            part_lvl = extract(
                exp.data_raw["truth"].x_gen,
                exp.data_raw["truth"].x_gen_batch,
                exp.data_raw["truth"].x_det_batch,
            )[: len(det_lvl)]
            model = extract(
                exp.data_raw["samples"].x_gen,
                exp.data_raw["samples"].x_gen_batch,
                exp.data_raw["samples"].x_det_batch,
            )
            part_lvl = coords.fourmomenta_to_x(part_lvl).cpu().detach()
            det_lvl = det_lvl_coords.fourmomenta_to_x(det_lvl).cpu().detach()
            model = coords.fourmomenta_to_x(model).cpu().detach()

            obs_names = [
                r"\log p_{T," + name + "}",
                "\phi_{" + name + "}",
                "\eta_{" + name + "}",
                "\log m^2_{" + name + "}",
            ]

            for channel in range(4):
                xlabel = obs_names[channel]
                xrange = np.array(
                    get_range(
                        [
                            part_lvl[..., channel],
                            # det_lvl[..., channel],
                            model[..., channel],
                        ]
                    )
                )
                if channel == 0:
                    logy = True
                else:
                    logy = False
                plot_histogram(
                    file=file,
                    train=part_lvl[..., channel],
                    test=det_lvl[..., channel],
                    model=model[..., channel],
                    title=exp.plot_title,
                    xlabel=xlabel,
                    xrange=xrange,
                    logy=logy,
                    model_label=model_label,
                    weights=weights,
                    mask_dict=mask_dict,
                )
                if exp.cfg.plotting.correlations:
                    plot_correlations(
                        file=file,
                        det=det_lvl[..., channel],
                        part=part_lvl[..., channel],
                        gen=model[..., channel],
                        title=exp.plot_title,
                        label=xlabel,
                        range=xrange,
                        model_label=model_label,
                    )


def plot_observables(
    exp,
    filename,
    model_label,
    weights=None,
    mask_dict=None,
):
    with PdfPages(filename) as file:
        for name in exp.obs.keys():
            extract = exp.obs[name]
            det_lvl = (
                extract(
                    exp.data_raw["samples"].x_det,
                    exp.data_raw["samples"].x_det_batch,
                    exp.data_raw["samples"].x_gen_batch,
                )
                .cpu()
                .detach()
            )
            part_lvl = (
                extract(
                    exp.data_raw["truth"].x_gen,
                    exp.data_raw["truth"].x_gen_batch,
                    exp.data_raw["truth"].x_det_batch,
                )[: len(det_lvl)]
                .cpu()
                .detach()
            )
            model = (
                extract(
                    exp.data_raw["samples"].x_gen,
                    exp.data_raw["samples"].x_gen_batch,
                    exp.data_raw["samples"].x_det_batch,
                )
                .cpu()
                .detach()
            )

            nan_mask = torch.isnan(part_lvl) | torch.isnan(det_lvl) | torch.isnan(model)

            LOGGER.info(
                f"Masking {nan_mask.sum()} NaNs in {name} observable, keeping {(~nan_mask).sum()} valid entries"
            )
            part_lvl = part_lvl[~nan_mask]
            det_lvl = det_lvl[~nan_mask]
            model = model[~nan_mask]

            xrange = np.array(
                get_range(
                    [
                        part_lvl,
                        # det_lvl[..., channel],
                        model,
                    ]
                )
            )

            if name == "z_g":
                xrange[0] = 0.0
            xlabel = name
            logy = False

            plot_histogram(
                file=file,
                train=part_lvl,
                test=det_lvl,
                model=model,
                title=exp.plot_title,
                xlabel=xlabel,
                xrange=xrange,
                logy=logy,
                model_label=model_label,
                weights=weights,
                mask_dict=mask_dict,
            )
            if exp.cfg.plotting.correlations:
                plot_correlations(
                    file=file,
                    det=det_lvl,
                    part=part_lvl,
                    gen=model,
                    title=exp.plot_title,
                    label=xlabel,
                    range=xrange,
                    model_label=model_label,
                )
