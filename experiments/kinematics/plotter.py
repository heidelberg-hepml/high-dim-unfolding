import torch
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from experiments.base_plots import plot_loss, plot_metric
from experiments.kinematics.plots import (
    plot_histogram,
    plot_calibration,
    simple_histogram,
    plot_roc,
    plot_2d_histogram,
)
from experiments.utils import get_range
from experiments.coordinates import fourmomenta_to_jetmomenta, JetScaledLogPtPhiEtaLogM2
from experiments.logger import LOGGER

N_SAMPLES = 100000


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


def plot_log_prob(exp, filename, model_label):
    # ugly out of the box plot
    import matplotlib.pyplot as plt

    with PdfPages(filename) as file:
        plt.hist(exp.NLLs, bins=100, alpha=0.5)
        plt.xlabel(r"$-\log p(x)$")
        plt.savefig(file, bbox_inches="tight", format="pdf")
        plt.close()


def plot_classifier(exp, filename, model_label):
    with PdfPages(filename) as file:
        # classifier train and validation loss
        plot_loss(
            file,
            [exp.classifier.tracker[key] for key in ["loss", "val_loss"]],
            lr=exp.classifier.tracker["lr"],
            labels=[f"train mse", f"val mse"],
            logy=True,
        )

        # probabilities
        data = [
            exp.classifier.results["logits"]["true"],
            exp.classifier.results["logits"]["fake"],
        ]
        simple_histogram(
            file,
            data,
            labels=["Test", "Generator"],
            xrange=[0, 1],
            xlabel="Classifier score",
            logx=False,
            logy=False,
        )
        simple_histogram(
            file,
            data,
            labels=["Test", "Generator"],
            xrange=[0, 1],
            xlabel="Classifier score",
            logx=False,
            logy=True,
        )

        # weights
        data = [
            exp.classifier.results["weights"]["true"],
            exp.classifier.results["weights"]["fake"],
        ]
        simple_histogram(
            file,
            data,
            labels=["Test", "Generator"],
            xrange=[0, 5],
            xlabel="Classifier weights",
            logx=False,
            logy=False,
        )
        simple_histogram(
            file,
            data,
            labels=["Test", "Generator"],
            xrange=[1e-3, 1e2],
            xlabel="Classifier weights",
            logx=True,
            logy=True,
        )

        # roc curve
        plot_roc(
            file,
            exp.classifier.results["tpr"],
            exp.classifier.results["fpr"],
            exp.classifier.results["auc"],
        )
        # calibration curve
        plot_calibration(
            file,
            exp.classifier.results["prob_true"],
            exp.classifier.results["prob_pred"],
        )


def plot_fourmomenta(
    exp, filename, model_label, jet=False, weights=None, mask_dict=None
):

    max_n = min(N_SAMPLES, exp.data_raw["truth"].x_gen_ptr.shape[0] - 1)

    part_max_n_ptr = exp.data_raw["truth"].x_gen_ptr[max_n]
    det_max_n_ptr = exp.data_raw["truth"].x_det_ptr[max_n]
    model_max_n_ptr = exp.data_raw["samples"].x_gen_ptr[max_n]

    part_batch_idx = exp.data_raw["truth"].x_gen_batch[:part_max_n_ptr]
    det_batch_idx = exp.data_raw["truth"].x_det_batch[:det_max_n_ptr]
    model_batch_idx = exp.data_raw["samples"].x_gen_batch[:model_max_n_ptr]

    part_x = exp.data_raw["truth"].x_gen[:part_max_n_ptr]
    det_x = exp.data_raw["truth"].x_det[:det_max_n_ptr]
    model_x = exp.data_raw["samples"].x_gen[:model_max_n_ptr]

    with PdfPages(filename) as file:
        for name in exp.obs_coords.keys():
            extract = exp.obs_coords[name]

            if not jet:
                det_lvl = (
                    extract(
                        det_x,
                        det_batch_idx,
                        part_batch_idx,
                        exp.data_raw["truth"].jet_det[:max_n],
                    )[0]
                    .cpu()
                    .detach()
                )
                part_lvl = (
                    extract(
                        part_x,
                        part_batch_idx,
                        det_batch_idx,
                        exp.data_raw["truth"].jet_gen[:max_n],
                    )[0][: len(det_lvl)]
                    .cpu()
                    .detach()
                )
                model = (
                    extract(
                        model_x,
                        model_batch_idx,
                        det_batch_idx,
                        exp.data_raw["samples"].jet_gen[:max_n],
                    )[0][: len(det_lvl)]
                    .cpu()
                    .detach()
                )
            else:
                det_lvl = exp.data_raw["truth"].jet_det[:max_n]
                part_lvl = exp.data_raw["truth"].jet_gen[:max_n]
                model = exp.data_raw["samples"].jet_gen[:max_n]

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
                    plot_2d_histogram(
                        file=file,
                        x1=det_lvl[..., channel],
                        y1=part_lvl[..., channel],
                        x2=det_lvl[..., channel],
                        y2=model[..., channel],
                        xlabel=xlabel + " (det)",
                        ylabel=xlabel + " (gen)",
                        range=xrange,
                        model_label=model_label,
                    )


def plot_jetmomenta(
    exp, filename, model_label, jet=False, weights=None, mask_dict=None
):

    max_n = min(N_SAMPLES, exp.data_raw["truth"].x_gen_ptr.shape[0] - 1)

    part_max_n_ptr = exp.data_raw["truth"].x_gen_ptr[max_n]
    det_max_n_ptr = exp.data_raw["truth"].x_det_ptr[max_n]
    model_max_n_ptr = exp.data_raw["samples"].x_gen_ptr[max_n]

    part_batch_idx = exp.data_raw["truth"].x_gen_batch[:part_max_n_ptr]
    det_batch_idx = exp.data_raw["truth"].x_det_batch[:det_max_n_ptr]
    model_batch_idx = exp.data_raw["samples"].x_gen_batch[:model_max_n_ptr]

    part_x = exp.data_raw["truth"].x_gen[:part_max_n_ptr]
    det_x = exp.data_raw["truth"].x_det[:det_max_n_ptr]
    model_x = exp.data_raw["samples"].x_gen[:model_max_n_ptr]

    with PdfPages(filename) as file:
        for name in exp.obs_coords.keys():
            extract = exp.obs_coords[name]
            if not jet:
                det_lvl = extract(
                    det_x,
                    det_batch_idx,
                    part_batch_idx,
                    exp.data_raw["truth"].jet_det[:max_n],
                )[0]
                part_lvl = extract(
                    part_x,
                    part_batch_idx,
                    det_batch_idx,
                    exp.data_raw["truth"].jet_gen[:max_n],
                )[0][: len(det_lvl)]
                model = extract(
                    model_x,
                    model_batch_idx,
                    det_batch_idx,
                    exp.data_raw["samples"].jet_gen[:max_n],
                )[0][: len(det_lvl)]
            else:
                det_lvl = exp.data_raw["truth"].jet_det[:max_n]
                part_lvl = exp.data_raw["truth"].jet_gen[:max_n]
                model = exp.data_raw["samples"].jet_gen[:max_n]

            part_lvl = fourmomenta_to_jetmomenta(part_lvl).cpu().detach()
            det_lvl = fourmomenta_to_jetmomenta(det_lvl).cpu().detach()
            model = fourmomenta_to_jetmomenta(model).cpu().detach()
            part_lvl[..., 3] = torch.sqrt(part_lvl[..., 3])
            det_lvl[..., 3] = torch.sqrt(det_lvl[..., 3])
            model[..., 3] = torch.sqrt(model[..., 3])

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
                if channel == 3 and exp.cfg.data.dataset == "ttbar":
                    xrange = np.array([120.0, 210.0])
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
                    plot_2d_histogram(
                        file=file,
                        x1=det_lvl[..., channel],
                        y1=part_lvl[..., channel],
                        x2=det_lvl[..., channel],
                        y2=model[..., channel],
                        xlabel=xlabel + " (det)",
                        ylabel=xlabel + " (gen)",
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
                exp.data_raw["truth"].x_det,
                exp.data_raw["truth"].x_det_batch,
                exp.data_raw["truth"].x_gen_batch,
            )[0]
            part_lvl = extract(
                exp.data_raw["truth"].x_gen,
                exp.data_raw["truth"].x_gen_batch,
                exp.data_raw["truth"].x_det_batch,
            )[0][: len(det_lvl)]
            model = extract(
                exp.data_raw["samples"].x_gen,
                exp.data_raw["samples"].x_gen_batch,
                exp.data_raw["samples"].x_det_batch,
            )[0][: len(det_lvl)]

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
                    plot_2d_histogram(
                        file=file,
                        x1=det_lvl[..., channel],
                        y1=part_lvl[..., channel],
                        x2=det_lvl[..., channel],
                        y2=model[..., channel],
                        xlabel=xlabel + " (det)",
                        ylabel=xlabel + " (gen)",
                        range=xrange,
                        model_label=model_label,
                    )


def plot_jetscaled(exp, filename, model_label, weights=None, mask_dict=None):

    coords = JetScaledLogPtPhiEtaLogM2(exp.cfg.data.pt_min)
    condition_coords = JetScaledLogPtPhiEtaLogM2(exp.cfg.data.pt_min)

    max_n = min(N_SAMPLES, exp.data_raw["truth"].x_gen_ptr.shape[0] - 1)

    part_max_n_ptr = exp.data_raw["truth"].x_gen_ptr[max_n]
    det_max_n_ptr = exp.data_raw["truth"].x_det_ptr[max_n]
    model_max_n_ptr = exp.data_raw["samples"].x_gen_ptr[max_n]

    part_batch_idx = exp.data_raw["truth"].x_gen_batch[:part_max_n_ptr]
    det_batch_idx = exp.data_raw["truth"].x_det_batch[:det_max_n_ptr]
    model_batch_idx = exp.data_raw["samples"].x_gen_batch[:model_max_n_ptr]

    part_x = exp.data_raw["truth"].x_gen[:part_max_n_ptr]
    det_x = exp.data_raw["truth"].x_det[:det_max_n_ptr]
    model_x = exp.data_raw["samples"].x_gen[:model_max_n_ptr]

    with PdfPages(filename) as file:
        for name in exp.obs_coords.keys():
            extract = exp.obs_coords[name]

            det_lvl, det_lvl_jet, det_lvl_ptr, det_lvl_pos = extract(
                det_x,
                det_batch_idx,
                part_batch_idx,
                true_jet=exp.data_raw["truth"].jet_det[:max_n],
            )
            part_lvl, part_lvl_jet, part_lvl_ptr, part_lvl_pos = extract(
                part_x,
                part_batch_idx,
                det_batch_idx,
                true_jet=exp.data_raw["truth"].jet_gen[:max_n],
            )
            model, model_jet, model_ptr, model_pos = extract(
                model_x,
                model_batch_idx,
                det_batch_idx,
                true_jet=exp.data_raw["samples"].jet_gen[:max_n],
            )

            part_lvl = (
                coords.fourmomenta_to_x(
                    part_lvl,
                    jet=part_lvl_jet,
                    ptr=part_lvl_ptr,
                    pos=part_lvl_pos,
                )
                .cpu()
                .detach()
            )
            det_lvl = (
                condition_coords.fourmomenta_to_x(
                    det_lvl,
                    jet=det_lvl_jet,
                    ptr=det_lvl_ptr,
                    pos=det_lvl_pos,
                )
                .cpu()
                .detach()
            )
            model = (
                coords.fourmomenta_to_x(
                    model,
                    jet=model_jet,
                    ptr=model_ptr,
                    pos=model_pos,
                )
                .cpu()
                .detach()
            )

            obs_names = [
                r"\log{p_{T," + name + r"}} - \log{p_{T,\text{ jet}}}",
                r"\phi_{" + name + r"} - \phi_{\text{jet}}",
                r"\eta_{" + name + r"} - \eta_{\text{jet}}",
                r"\log m_{" + name + r"}^2 - \log m_{\text{jet}}^2",
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
                    plot_2d_histogram(
                        file=file,
                        x1=det_lvl[..., channel],
                        y1=part_lvl[..., channel],
                        x2=det_lvl[..., channel],
                        y2=model[..., channel],
                        xlabel=xlabel + " (det)",
                        ylabel=xlabel + " (gen)",
                        range=xrange,
                        model_label=model_label,
                    )


def plot_correlations(exp, filename, model_label, weights=None, mask_dict=None):

    coords = exp.model.coordinates
    det_lvl_coords = exp.model.condition_coordinates

    with PdfPages(filename) as file:
        for name in exp.corr.keys():
            extract_x = exp.corr[name][0]
            max_n = min(N_SAMPLES, exp.data_raw["truth"].x_gen_ptr.shape[0] - 1)
            max_n_ptr = exp.data_raw["truth"].x_gen_ptr[max_n]
            gen_lvl_x = (
                extract_x(
                    exp.data_raw["truth"].x_gen[:max_n_ptr],
                    exp.data_raw["truth"].x_gen_batch[:max_n_ptr],
                    exp.data_raw["truth"].x_det_batch[:max_n_ptr],
                )
                .cpu()
                .detach()
            )
            model_x = (
                extract_x(
                    exp.data_raw["samples"].x_gen[:max_n_ptr],
                    exp.data_raw["samples"].x_gen_batch[:max_n_ptr],
                    exp.data_raw["samples"].x_det_batch[:max_n_ptr],
                )
                .cpu()
                .detach()
            )
            extract_y = exp.corr[name][1]
            gen_lvl_y = (
                extract_y(
                    exp.data_raw["truth"].x_gen[:max_n_ptr],
                    exp.data_raw["truth"].x_gen_batch[:max_n_ptr],
                    exp.data_raw["truth"].x_det_batch[:max_n_ptr],
                )
                .cpu()
                .detach()
            )
            model_y = (
                extract_y(
                    exp.data_raw["samples"].x_gen[:max_n_ptr],
                    exp.data_raw["samples"].x_gen_batch[:max_n_ptr],
                    exp.data_raw["samples"].x_det_batch[:max_n_ptr],
                )
                .cpu()
                .detach()
            )
            # mask = (
            #     torch.isnan(gen_lvl_x)
            #     | torch.isnan(gen_lvl_y)
            #     | torch.isnan(model_x)
            #     | torch.isnan(model_y)
            # )
            # gen_lvl_x = gen_lvl_x[~mask]
            # gen_lvl_y = gen_lvl_y[~mask]
            # model_x = model_x[~mask]
            # model_y = model_y[~mask]

            xlabel = name.split(" vs ")[0]
            ylabel = name.split(" vs ")[1]
            xrange = np.array(get_range([model_x, gen_lvl_x]))
            yrange = np.array(get_range([model_y, gen_lvl_y]))

            plot_2d_histogram(
                file=file,
                x1=gen_lvl_x,
                y1=gen_lvl_y,
                x2=model_x,
                y2=model_y,
                xlabel=xlabel,
                ylabel=ylabel,
                range=(xrange, yrange),
                model_label=model_label,
            )


def plot_observables(
    exp,
    filename,
    model_label,
    weights=None,
    mask_dict=None,
):
    max_n = min(N_SAMPLES, exp.data_raw["truth"].x_gen_ptr.shape[0] - 1)

    part_max_n_ptr = exp.data_raw["truth"].x_gen_ptr[max_n]
    det_max_n_ptr = exp.data_raw["truth"].x_det_ptr[max_n]
    model_max_n_ptr = exp.data_raw["samples"].x_gen_ptr[max_n]

    part_batch_idx = exp.data_raw["truth"].x_gen_batch[:part_max_n_ptr]
    det_batch_idx = exp.data_raw["truth"].x_det_batch[:det_max_n_ptr]
    model_batch_idx = exp.data_raw["samples"].x_gen_batch[:model_max_n_ptr]

    part_consts = exp.data_raw["truth"].x_gen[:part_max_n_ptr]
    det_consts = exp.data_raw["truth"].x_det[:det_max_n_ptr]
    model_consts = exp.data_raw["samples"].x_gen[:model_max_n_ptr]

    with PdfPages(filename) as file:
        for name in exp.obs.keys():
            LOGGER.info(f"Plotting observable {name}")
            extract = exp.obs[name]
            det_lvl = (
                extract(
                    det_consts,
                    det_batch_idx,
                    part_batch_idx,
                )
                .cpu()
                .detach()
            )
            part_lvl = (
                extract(
                    part_consts,
                    part_batch_idx,
                    det_batch_idx,
                )
                .cpu()
                .detach()
            )
            model = (
                extract(
                    model_consts,
                    model_batch_idx,
                    det_batch_idx,
                )
                .cpu()
                .detach()
            )

            min_length = min(det_lvl.shape[0], part_lvl.shape[0], model.shape[0])
            nan_filter = (
                torch.isnan(det_lvl[:min_length])
                | torch.isnan(part_lvl[:min_length])
                | torch.isnan(model[:min_length])
            ).squeeze()

            det_lvl = det_lvl[:min_length][~nan_filter]
            part_lvl = part_lvl[:min_length][~nan_filter]
            model = model[:min_length][~nan_filter]

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
            if "phi" in name or "eta" in name:
                xrange[0] = -0.06
                xrange[1] = 0.06
            if "Delta R" in name:
                xrange[0] = 0.0
                xrange[1] = 0.1
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
                plot_2d_histogram(
                    file=file,
                    x1=det_lvl,
                    y1=part_lvl,
                    x2=det_lvl,
                    y2=model,
                    xlabel=xlabel + " (det)",
                    ylabel=xlabel + " (gen)",
                    range=xrange,
                    model_label=model_label,
                )
