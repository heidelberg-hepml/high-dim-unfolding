from operator import truth
import torch
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from torch_geometric.utils import scatter

from experiments.base_plots import (
    plot_loss,
    plot_metric,
    plot_histogram,
    plot_ratio_histogram,
)
from experiments.kinematics.observables import (
    get_constituent,
    get_deta,
    get_dphi,
    get_dr,
    calculate_eec,
    tau,
    sd_mass,
    compute_zg,
    NSUB_AVAIL,
    SOFTDROP_AVAIL,
)

# from experiments.kinematics.plots import (
#     plot_histogram,
#     plot_calibration,
#     simple_histogram,
#     plot_roc,
#     plot_2d_histogram,
# )
from experiments.utils import get_range, get_pt, get_phi, get_eta, get_mass
from experiments.coordinates import fourmomenta_to_jetmomenta, JetScaledLogPtPhiEtaLogM2
from experiments.logger import LOGGER

N_SAMPLES = -1


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
        for key in exp.train_metrics.keys():
            plot_loss(
                file,
                [
                    exp.train_metrics[key],
                    exp.val_metrics[key],
                ],
                lr=exp.train_lr,
                labels=[f"train {key}", f"val {key}"],
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

    max_n = (
        min(N_SAMPLES, exp.data_raw["truth"].x_gen_ptr.shape[0] - 1)
        if N_SAMPLES > 0
        else exp.data_raw["truth"].x_gen_ptr.shape[0] - 1
    )

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

    max_n = (
        min(N_SAMPLES, exp.data_raw["truth"].x_gen_ptr.shape[0] - 1)
        if N_SAMPLES > 0
        else exp.data_raw["truth"].x_gen_ptr.shape[0] - 1
    )

    part_max_n_ptr = exp.data_raw["truth"].x_gen_ptr  # [max_n]
    det_max_n_ptr = exp.data_raw["truth"].x_det_ptr  # [max_n]
    model_max_n_ptr = exp.data_raw["samples"].x_gen_ptr  # [max_n]

    part_batch_idx = exp.data_raw["truth"].x_gen_batch  # [:part_max_n_ptr]
    det_batch_idx = exp.data_raw["truth"].x_det_batch  # [:det_max_n_ptr]
    model_batch_idx = exp.data_raw["samples"].x_gen_batch  # [:model_max_n_ptr]

    part_x = exp.data_raw["truth"].x_gen  # [:part_max_n_ptr]
    det_x = exp.data_raw["truth"].x_det  # [:det_max_n_ptr]
    model_x = exp.data_raw["samples"].x_gen  # [:model_max_n_ptr]

    with PdfPages(filename) as file:
        for name in exp.obs_coords.keys():
            extract = exp.obs_coords[name]
            if not jet:
                det_lvl = extract(
                    det_x,
                    det_batch_idx,
                    part_batch_idx,
                    exp.data_raw["truth"].jet_det,  # [:max_n],
                )[0]
                part_lvl = extract(
                    part_x,
                    part_batch_idx,
                    det_batch_idx,
                    exp.data_raw["truth"].jet_gen,  # [:max_n],
                )[0][: len(det_lvl)]
                model = extract(
                    model_x,
                    model_batch_idx,
                    det_batch_idx,
                    exp.data_raw["samples"].jet_gen,  # [:max_n],
                )[0][: len(det_lvl)]
            else:
                det_lvl = exp.data_raw["truth"].jet_det.clone()  # [:max_n]
                part_lvl = exp.data_raw["truth"].jet_gen.clone()  # [:max_n]
                model = exp.data_raw["samples"].jet_gen.clone()  # [:max_n]

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

    coords = exp.model.const_coordinates
    det_lvl_coords = exp.model.condition_const_coordinates

    max_n = (
        min(N_SAMPLES, exp.data_raw["truth"].x_gen_ptr.shape[0] - 1)
        if N_SAMPLES > 0
        else exp.data_raw["truth"].x_gen_ptr.shape[0] - 1
    )

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
            det_lvl = extract(
                det_x,
                det_batch_idx,
                part_batch_idx,
            )[0]
            part_lvl = extract(
                part_x,
                part_batch_idx,
                det_batch_idx,
            )[
                0
            ][: len(det_lvl)]
            model = extract(
                model_x,
                model_batch_idx,
                det_batch_idx,
            )[
                0
            ][: len(det_lvl)]

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

    max_n = (
        min(N_SAMPLES, exp.data_raw["truth"].x_gen_ptr.shape[0] - 1)
        if N_SAMPLES > 0
        else exp.data_raw["truth"].x_gen_ptr.shape[0] - 1
    )

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

    coords = exp.model.const_coordinates
    det_lvl_coords = exp.model.condition_const_coordinates

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
    max_n = (
        min(N_SAMPLES, exp.data_raw["truth"].x_gen_ptr.shape[0] - 1)
        if N_SAMPLES > 0
        else exp.data_raw["truth"].x_gen_ptr.shape[0] - 1
    )

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


def plot_z(exp, filename, model_label):
    samples = exp.data_raw["samples"].detach().cpu()
    truth = exp.data_raw["truth"].detach().cpu()
    with PdfPages(filename) as file:
        if "multiplicity" in exp.cfg.plotting.observables:
            gen_mult = truth.x_gen_ptr.diff()
            det_mult = truth.x_det_ptr.diff()
            sample_mult = samples.x_gen_ptr.diff()
            plot_ratio_histogram(
                data={
                    "part": gen_mult,
                    "reco": det_mult,
                    model_label: sample_mult,
                },
                reference_key="part",
                xlabel=r"N",
                bins_range=torch.tensor([2, 58], dtype=torch.int32),
                n_int_per_bin=2,
                no_ratio_keys=["reco"],
                file=file,
                title={"title": "Z+jets", "x": 0.18, "y": 0.95},
                yrange=[0, 0.075],
            )
        gen_jets = truth.jet_gen
        det_jets = truth.jet_det
        sample_jets = scatter(samples.x_gen, samples.x_gen_batch)

        if "jet" in exp.cfg.plotting.observables:
            plot_ratio_histogram(
                data={
                    "part": get_pt(gen_jets),
                    "reco": get_pt(det_jets),
                    model_label: get_pt(samples.jet_gen),
                },
                reference_key="part",
                xlabel=r"p_{T,J} \text{ [GeV]}",
                bins_range=torch.tensor([50, 800], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=True,
                legend_loc={"loc": "lower left", "bbox_to_anchor": (0.05, 0.0)},
                file=file,
                title={"title": "Z+jets", "x": 0.85, "y": 0.95},
            )

            plot_ratio_histogram(
                data={
                    "part": get_phi(gen_jets),
                    "reco": get_phi(det_jets),
                    model_label: get_phi(samples.jet_gen),
                },
                reference_key="part",
                xlabel=r"\phi_J",
                bins_range=torch.tensor([-torch.pi, torch.pi], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                xrange=(-3.3, 3.3),
                legend_loc="lower right",
                title={"title": "Z+jets", "x": 0.85, "y": 0.95},
                yrange=[0, 0.2],
            )

            plot_ratio_histogram(
                data={
                    "part": get_eta(gen_jets),
                    "reco": get_eta(det_jets),
                    model_label: get_eta(samples.jet_gen),
                },
                reference_key="part",
                xlabel=r"\eta_J",
                bins_range=torch.tensor([-4, 4], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                legend_loc="lower center",
                title={"title": "Z+jets", "x": 0.85, "y": 0.95},
            )

            plot_ratio_histogram(
                data={
                    "part": get_mass(gen_jets),
                    "reco": get_mass(det_jets),
                    model_label: get_mass(samples.jet_gen),
                },
                reference_key="part",
                xlabel=r"m_J \text{ [GeV]}",
                bins_range=torch.tensor([3, 50], dtype=torch.float32),
                no_ratio_keys=["reco"],
                file=file,
                title={"title": "Z+jets", "x": 0.18, "y": 0.95},
                yrange=[0, 0.09],
            )

        if exp.__class__.__name__ == "JetKinematicsExperiment":
            return

        if "constjet" in exp.cfg.plotting.observables:
            plot_ratio_histogram(
                data={
                    "part": get_pt(gen_jets),
                    "reco": get_pt(det_jets),
                    model_label: get_pt(sample_jets),
                },
                reference_key="part",
                xlabel=r"p_{T,\text{jet}} \text{ [GeV]}",
                bins_range=torch.tensor([50, 800], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=True,
                legend_loc={"loc": "lower left", "bbox_to_anchor": (0.05, 0.0)},
                file=file,
                title={"title": "Z+jets", "x": 0.85, "y": 0.95},
            )

            plot_ratio_histogram(
                data={
                    "part": get_phi(gen_jets),
                    "reco": get_phi(det_jets),
                    model_label: get_phi(sample_jets),
                },
                reference_key="part",
                xlabel=r"\phi_\text{jet}",
                bins_range=torch.tensor([-torch.pi, torch.pi], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                xrange=(-3.3, 3.3),
                legend_loc="lower right",
                title={"title": "Z+jets", "x": 0.85, "y": 0.95},
                yrange=[0, 0.2],
            )

            plot_ratio_histogram(
                data={
                    "part": get_eta(gen_jets),
                    "reco": get_eta(det_jets),
                    model_label: get_eta(sample_jets),
                },
                reference_key="part",
                xlabel=r"\eta_\text{jet}",
                bins_range=torch.tensor([-4, 4], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                legend_loc="lower center",
                title={"title": "Z+jets", "x": 0.85, "y": 0.95},
            )

            plot_ratio_histogram(
                data={
                    "part": get_mass(gen_jets),
                    "reco": get_mass(det_jets),
                    model_label: get_mass(sample_jets),
                },
                reference_key="part",
                xlabel=r"m_\text{jet} \text{ [GeV]}",
                bins_range=torch.tensor([3, 50], dtype=torch.float32),
                no_ratio_keys=["reco"],
                file=file,
                title={"title": "Z+jets", "x": 0.18, "y": 0.95},
                yrange=[0, 0.09],
            )

        for i in range(1, exp.cfg.plotting.n_pt + 1):
            gen_const_i = get_constituent(truth.x_gen, truth.x_gen_ptr, i)
            det_const_i = get_constituent(truth.x_det, truth.x_det_ptr, i)
            sample_const_i = get_constituent(samples.x_gen, samples.x_gen_ptr, i)
            plot_ratio_histogram(
                data={
                    "part": get_pt(gen_const_i),
                    "reco": get_pt(det_const_i),
                    model_label: get_pt(sample_const_i),
                },
                reference_key="part",
                xlabel=rf"p_{{T,{i}}}",
                bins_range=get_range(
                    [
                        get_pt(gen_const_i),
                        get_pt(det_const_i),
                        get_pt(sample_const_i),
                    ]
                ),
                no_ratio_keys=["reco"],
                logy=True,
                legend_loc={"loc": "lower left", "bbox_to_anchor": (0.05, 0.05)},
                file=file,
                title={"title": "Z+jets", "x": 0.85, "y": 0.95},
            )
            plot_ratio_histogram(
                data={
                    "part": get_phi(gen_const_i),
                    "reco": get_phi(det_const_i),
                    model_label: get_phi(sample_const_i),
                },
                reference_key="part",
                xlabel=rf"\phi_{i}",
                bins_range=torch.tensor([-torch.pi, torch.pi], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                legend_loc="lower right",
                xrange=(-3.3, 3.3),
                file=file,
                title={"title": "Z+jets", "x": 0.85, "y": 0.95},
                yrange=[0, 0.2],
            )
            plot_ratio_histogram(
                data={
                    "part": get_eta(gen_const_i),
                    "reco": get_eta(det_const_i),
                    model_label: get_eta(sample_const_i),
                },
                reference_key="part",
                xlabel=rf"\eta_{i}",
                bins_range=torch.tensor([-4, 4], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                legend_loc="lower center",
                file=file,
                title={"title": "Z+jets", "x": 0.18, "y": 0.95},
            )

        if "nsubjettiness" in exp.cfg.plotting.observables and NSUB_AVAIL:
            LOGGER.info("Calculating n-subjettiness")
            gen_tau1 = tau(truth.x_gen, truth.x_gen_batch, N=1, R0=0.4)
            det_tau1 = tau(truth.x_det, truth.x_det_batch, N=1, R0=0.4)
            sample_tau1 = tau(samples.x_gen, samples.x_gen_batch, N=1, R0=0.4)

            gen_tau2 = tau(truth.x_gen, truth.x_gen_batch, N=2, R0=0.4)
            det_tau2 = tau(truth.x_det, truth.x_det_batch, N=2, R0=0.4)
            sample_tau2 = tau(samples.x_gen, samples.x_gen_batch, N=2, R0=0.4)

            gen_tau21 = torch.where(
                gen_tau1 > 0, gen_tau2 / gen_tau1, torch.tensor(0.0)
            )
            det_tau21 = torch.where(
                det_tau1 > 0, det_tau2 / det_tau1, torch.tensor(0.0)
            )
            sample_tau21 = torch.where(
                sample_tau1 > 0, sample_tau2 / sample_tau1, torch.tensor(0.0)
            )

            plot_ratio_histogram(
                data={
                    "part": gen_tau1,
                    "reco": det_tau1,
                    model_label: sample_tau1,
                },
                reference_key="part",
                xlabel=r"w",
                bins_range=torch.tensor([0, 0.4], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                title={"title": "Z+jets", "x": 0.18, "y": 0.95},
                yrange=[0, 12],
            )

            plot_ratio_histogram(
                data={
                    "part": gen_tau2,
                    "reco": det_tau2,
                    model_label: sample_tau2,
                },
                reference_key="part",
                xlabel=r"\tau_2^{\beta=1}",
                bins_range=torch.tensor([0, 0.25], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                title={"title": "Z+jets", "x": 0.18, "y": 0.95},
                yrange=[0, 16],
            )

            plot_ratio_histogram(
                data={
                    "part": gen_tau21,
                    "reco": det_tau21,
                    model_label: sample_tau21,
                },
                reference_key="part",
                xlabel=r"\tau_{21}^{\beta=1}",
                bins_range=torch.tensor([0, 1.4], dtype=torch.float32),
                yrange=(0, 2.1),
                no_ratio_keys=["reco"],
                logy=False,
                legend_loc=None,
                file=file,
                title={"title": "Z+jets", "x": 0.18, "y": 0.95},
            )

        if "softdropmass" in exp.cfg.plotting.observables and SOFTDROP_AVAIL:
            LOGGER.info("Calculating softdrop mass")
            gen_rho = sd_mass(truth.x_gen, truth.x_gen_batch, R0=0.8)
            det_rho = sd_mass(truth.x_det, truth.x_det_batch, R0=0.8)
            sample_rho = sd_mass(samples.x_gen, samples.x_gen_batch, R0=0.8)

            plot_ratio_histogram(
                data={
                    "part": gen_rho,
                    "reco": det_rho,
                    model_label: sample_rho,
                },
                reference_key="part",
                xlabel=r"\log \rho ",
                bins_range=torch.tensor([-14, -2.5], dtype=torch.float32),
                yrange=(0.0, 0.21),
                no_ratio_keys=["reco"],
                logy=False,
                legend_loc=None,
                file=file,
                title={"title": "Z+jets", "x": 0.85, "y": 0.95},
            )

        if "momentumfraction" in exp.cfg.plotting.observables and SOFTDROP_AVAIL:
            LOGGER.info("Calculating groomed jet momentum fraction")
            gen_zg = compute_zg(truth.x_gen, truth.x_gen_batch, R0=0.8)
            det_zg = compute_zg(truth.x_det, truth.x_det_batch, R0=0.8)
            sample_zg = compute_zg(samples.x_gen, samples.x_gen_batch, R0=0.8)

            plot_ratio_histogram(
                data={
                    "part": gen_zg,
                    "reco": det_zg,
                    model_label: sample_zg,
                },
                reference_key="part",
                xlabel=r"z_g",
                bins_range=torch.tensor([0.1, 0.5], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                xrange=(0.09, 0.51),
                title={"title": "Z+jets", "x": 0.3, "y": 0.95},
            )

        if "eec" in exp.cfg.plotting.observables:
            LOGGER.info("Calculating EEC")
            gen_eecs = calculate_eec(truth.x_gen, truth.x_gen_ptr)
            det_eecs = calculate_eec(truth.x_det, truth.x_det_ptr)
            sample_eecs = calculate_eec(samples.x_gen, samples.x_gen_ptr)

            plot_ratio_histogram(
                data={
                    "part": gen_eecs[:, 0],
                    "reco": det_eecs[:, 0],
                    model_label: sample_eecs[:, 0],
                },
                reference_key="part",
                no_ratio_keys=["reco"],
                bins_range=torch.tensor([1e-6, 1e-2], dtype=torch.float32),
                xlabel=r"z",
                ylabel=r"\text{Normalized EEC}",
                yrange=(0, 3e2),
                logx=True,
                logy=True,
                file=file,
                title={"title": "Z+jets", "x": 0.18, "y": 0.95},
                legend_loc={"loc": "lower left", "bbox_to_anchor": (0.4, 0.0)},
                weights={
                    "part": gen_eecs[:, 1],
                    "reco": det_eecs[:, 1],
                    model_label: sample_eecs[:, 1],
                },
            )

        for i, j in exp.cfg.plotting.angles:
            gen_dphi = get_dphi(truth.x_gen, truth.x_gen_ptr, i, j)
            det_dphi = get_dphi(truth.x_det, truth.x_det_ptr, i, j)
            sample_dphi = get_dphi(samples.x_gen, samples.x_gen_ptr, i, j)
            plot_ratio_histogram(
                data={
                    "part": gen_dphi,
                    "reco": det_dphi,
                    model_label: sample_dphi,
                },
                reference_key="part",
                xlabel=rf"\Delta \phi_{{{i},{j}}}",
                bins_range=torch.tensor([-0.1, 0.1], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                title={"title": "Z+jets", "x": 0.18, "y": 0.95},
            )
            gen_deta = get_deta(truth.x_gen, truth.x_gen_ptr, i, j)
            det_deta = get_deta(truth.x_det, truth.x_det_ptr, i, j)
            sample_deta = get_deta(samples.x_gen, samples.x_gen_ptr, i, j)
            plot_ratio_histogram(
                data={
                    "part": gen_deta,
                    "reco": det_deta,
                    model_label: sample_deta,
                },
                reference_key="part",
                xlabel=rf"\Delta \eta_{{{i},{j}}}",
                bins_range=torch.tensor([-0.1, 0.1], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                title={"title": "Z+jets", "x": 0.18, "y": 0.95},
            )
            gen_dr = get_dr(truth.x_gen, truth.x_gen_ptr, i, j)
            det_dr = get_dr(truth.x_det, truth.x_det_ptr, i, j)
            sample_dr = get_dr(samples.x_gen, samples.x_gen_ptr, i, j)
            plot_ratio_histogram(
                data={
                    "part": gen_dr,
                    "reco": det_dr,
                    model_label: sample_dr,
                },
                reference_key="part",
                xlabel=rf"\Delta R_{{{i},{j}}}",
                bins_range=torch.tensor([0, 0.07], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                legend_loc=(
                    "upper right"
                    if i < 3
                    else {"loc": "upper left", "bbox_to_anchor": (0.15, 0.7)}
                ),
                title={
                    "title": "Z+jets",
                    "x": 0.2 if i < 3 else 0.85,
                    "y": 0.18 if i < 3 else 0.95,
                },
            )


def plot_t(exp, filename, model_label):
    samples = exp.data_raw["samples"].detach().cpu()
    truth = exp.data_raw["truth"].detach().cpu()
    with PdfPages(filename) as file:
        if "multiplicity" in exp.cfg.plotting.observables:
            gen_mult = truth.x_gen_ptr.diff()
            det_mult = truth.x_det_ptr.diff()
            sample_mult = samples.x_gen_ptr.diff()
            plot_ratio_histogram(
                data={
                    "part": gen_mult,
                    "reco": det_mult,
                    model_label: sample_mult,
                },
                reference_key="part",
                xlabel=r"N",
                bins_range=torch.tensor([15, 130], dtype=torch.int32),
                n_int_per_bin=3,
                no_ratio_keys=["reco"],
                file=file,
                title={"title": r"$t\bar{t}$", "x": 0.09, "y": 0.95},
                yrange=[0, 0.075],
            )
        gen_jets = truth.jet_gen
        det_jets = truth.jet_det
        sample_jets = scatter(samples.x_gen, samples.x_gen_batch)

        if "jet" in exp.cfg.plotting.observables:
            plot_ratio_histogram(
                data={
                    "part": get_pt(gen_jets),
                    "reco": get_pt(det_jets),
                    model_label: get_pt(samples.jet_gen),
                },
                reference_key="part",
                xlabel=r"p_{T,J} \text{ [GeV]}",
                bins_range=torch.tensor([400, 800], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=True,
                file=file,
                xrange=(390, 800),
                legend_loc="lower left",
                title={"title": r"$t\bar{t}$", "x": 0.93, "y": 0.95},
            )

            plot_ratio_histogram(
                data={
                    "part": get_phi(gen_jets),
                    "reco": get_phi(det_jets),
                    model_label: get_phi(samples.jet_gen),
                },
                reference_key="part",
                xlabel=r"\phi_J",
                bins_range=torch.tensor([-torch.pi, torch.pi], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                xrange=(-3.3, 3.3),
                legend_loc="lower right",
                title=r"$t\bar{t}$",
                yrange=[0, 0.21],
            )

            plot_ratio_histogram(
                data={
                    "part": get_eta(gen_jets),
                    "reco": get_eta(det_jets),
                    model_label: get_eta(samples.jet_gen),
                },
                reference_key="part",
                xlabel=r"\eta_J",
                bins_range=torch.tensor([-2.4, 2.4], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                legend_loc="lower center",
                title=r"$t\bar{t}$",
            )

            plot_ratio_histogram(
                data={
                    "part": get_mass(gen_jets),
                    "reco": get_mass(det_jets),
                    model_label: get_mass(samples.jet_gen),
                },
                reference_key="part",
                xlabel=r"m_J \text{ [GeV]}",
                bins_range=torch.tensor([130, 200], dtype=torch.float32),
                no_ratio_keys=["reco"],
                file=file,
                legend_loc="upper left",
                title={"title": r"$t\bar{t}$", "x": 0.93, "y": 0.95},
            )

        if exp.__class__.__name__ == "JetKinematicsExperiment":
            return

        if "constjet" in exp.cfg.plotting.observables:

            plot_ratio_histogram(
                data={
                    "part": get_pt(gen_jets),
                    "reco": get_pt(det_jets),
                    model_label: get_pt(sample_jets),
                },
                reference_key="part",
                xlabel=r"p_{T,\text{jet}} \text{ [GeV]}",
                bins_range=torch.tensor([400, 800], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=True,
                file=file,
                xrange=(390, 800),
                legend_loc="lower left",
                title={"title": r"$t\bar{t}$", "x": 0.93, "y": 0.95},
            )

            plot_ratio_histogram(
                data={
                    "part": get_phi(gen_jets),
                    "reco": get_phi(det_jets),
                    model_label: get_phi(sample_jets),
                },
                reference_key="part",
                xlabel=r"\phi_{\text{jet}}",
                bins_range=torch.tensor([-torch.pi, torch.pi], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                legend_loc="lower right",
                xrange=(-3.3, 3.3),
                title={"title": r"$t\bar{t}$", "x": 0.93, "y": 0.95},
                yrange=(0, 0.21),
            )

            plot_ratio_histogram(
                data={
                    "part": get_eta(gen_jets),
                    "reco": get_eta(det_jets),
                    model_label: get_eta(sample_jets),
                },
                reference_key="part",
                xlabel=r"\eta_{\text{jet}}",
                bins_range=torch.tensor([-2.4, 2.4], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                legend_loc="lower center",
                title=r"$t\bar{t}$",
            )

            plot_ratio_histogram(
                data={
                    "part": get_mass(gen_jets),
                    "reco": get_mass(det_jets),
                    model_label: get_mass(sample_jets),
                },
                reference_key="part",
                xlabel=r"m_{\text{jet}} \text{ [GeV]}",
                bins_range=torch.tensor([140, 200], dtype=torch.float32),
                no_ratio_keys=["reco"],
                file=file,
                legend_loc=None,
                title={"title": r"$t\bar{t}$", "x": 0.93, "y": 0.95},
            )

        for i in range(1, exp.cfg.plotting.n_pt + 1):
            gen_const_i = get_constituent(truth.x_gen, truth.x_gen_ptr, i)
            det_const_i = get_constituent(truth.x_det, truth.x_det_ptr, i)
            sample_const_i = get_constituent(samples.x_gen, samples.x_gen_ptr, i)
            plot_ratio_histogram(
                data={
                    "part": get_pt(gen_const_i),
                    "reco": get_pt(det_const_i),
                    model_label: get_pt(sample_const_i),
                },
                reference_key="part",
                xlabel=rf"p_{{T,{i}}}",
                bins_range=get_range(
                    [
                        get_pt(gen_const_i),
                        get_pt(det_const_i),
                        get_pt(sample_const_i),
                    ]
                ),
                no_ratio_keys=["reco"],
                logy=True,
                legend_loc={"loc": "upper right", "bbox_to_anchor": (1.0, 0.9)},
                file=file,
                title={"title": r"$t\bar{t}$", "x": 0.93, "y": 0.95},
            )
            plot_ratio_histogram(
                data={
                    "part": get_phi(gen_const_i),
                    "reco": get_phi(det_const_i),
                    model_label: get_phi(sample_const_i),
                },
                reference_key="part",
                xlabel=rf"\phi_{i}",
                bins_range=torch.tensor([-torch.pi, torch.pi], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                legend_loc="lower right",
                xrange=(-3.3, 3.3),
                file=file,
                title={"title": r"$t\bar{t}$", "x": 0.93, "y": 0.95},
                yrange=[0, 0.21],
            )
            plot_ratio_histogram(
                data={
                    "part": get_eta(gen_const_i),
                    "reco": get_eta(det_const_i),
                    model_label: get_eta(sample_const_i),
                },
                reference_key="part",
                xlabel=rf"\eta_{i}",
                bins_range=torch.tensor([-2.4, 2.4], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                legend_loc="lower center",
                file=file,
                title={"title": r"$t\bar{t}$", "x": 0.93, "y": 0.95},
            )

        if "nsubjettiness" in exp.cfg.plotting.observables and NSUB_AVAIL:
            LOGGER.info("Calculating n-subjettiness")
            gen_tau1 = tau(truth.x_gen, truth.x_gen_batch, N=1, R0=0.4)
            det_tau1 = tau(truth.x_det, truth.x_det_batch, N=1, R0=0.4)
            sample_tau1 = tau(samples.x_gen, samples.x_gen_batch, N=1, R0=0.4)

            gen_tau2 = tau(truth.x_gen, truth.x_gen_batch, N=2, R0=0.4)
            det_tau2 = tau(truth.x_det, truth.x_det_batch, N=2, R0=0.4)
            sample_tau2 = tau(samples.x_gen, samples.x_gen_batch, N=2, R0=0.4)

            gen_tau21 = torch.where(
                gen_tau1 > 0, gen_tau2 / gen_tau1, torch.tensor(0.0)
            )
            det_tau21 = torch.where(
                det_tau1 > 0, det_tau2 / det_tau1, torch.tensor(0.0)
            )
            sample_tau21 = torch.where(
                sample_tau1 > 0, sample_tau2 / sample_tau1, torch.tensor(0.0)
            )

            plot_ratio_histogram(
                data={
                    "part": gen_tau1,
                    "reco": det_tau1,
                    model_label: sample_tau1,
                },
                reference_key="part",
                xlabel=r"w",
                bins_range=torch.tensor([0.15, 1.05], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                legend_loc=None,
                title={"title": r"$t\bar{t}$", "x": 0.93, "y": 0.95},
            )

            plot_ratio_histogram(
                data={
                    "part": gen_tau2,
                    "reco": det_tau2,
                    model_label: sample_tau2,
                },
                reference_key="part",
                xlabel=r"\tau_2^{\beta=1}",
                bins_range=torch.tensor([0.03, 0.7], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                legend_loc=None,
                title={"title": r"$t\bar{t}$", "x": 0.93, "y": 0.95},
            )

            plot_ratio_histogram(
                data={
                    "part": gen_tau21,
                    "reco": det_tau21,
                    model_label: sample_tau21,
                },
                reference_key="part",
                xlabel=r"\tau_{21}^{\beta=1}",
                bins_range=torch.tensor([0.07, 0.9], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                legend_loc=None,
                title={"title": r"$t\bar{t}$", "x": 0.93, "y": 0.95},
            )

        if "softdropmass" in exp.cfg.plotting.observables and SOFTDROP_AVAIL:
            LOGGER.info("Calculating softdrop mass")
            gen_rho = sd_mass(truth.x_gen, truth.x_gen_batch, R0=0.8)
            det_rho = sd_mass(truth.x_det, truth.x_det_batch, R0=0.8)
            sample_rho = sd_mass(samples.x_gen, samples.x_gen_batch, R0=0.8)

            plot_ratio_histogram(
                data={
                    "part": gen_rho,
                    "reco": det_rho,
                    model_label: sample_rho,
                },
                reference_key="part",
                xlabel=r"\log \rho ",
                bins_range=torch.tensor([-23, -6], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                legend_loc=None,
                file=file,
                title={"title": r"$t\bar{t}$", "x": 0.93, "y": 0.95},
            )

        if "momentumfraction" in exp.cfg.plotting.observables and SOFTDROP_AVAIL:
            LOGGER.info("Calculating groomed jet momentum fraction")
            gen_zg = compute_zg(truth.x_gen, truth.x_gen_batch, R0=0.8)
            det_zg = compute_zg(truth.x_det, truth.x_det_batch, R0=0.8)
            sample_zg = compute_zg(samples.x_gen, samples.x_gen_batch, R0=0.8)

            plot_ratio_histogram(
                data={
                    "part": gen_zg,
                    "reco": det_zg,
                    model_label: sample_zg,
                },
                reference_key="part",
                xlabel=r"z_g",
                bins_range=torch.tensor([0.1, 0.5], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                xrange=(0.09, 0.51),
                legend_loc=None,
                title={"title": r"$t\bar{t}$", "x": 0.93, "y": 0.95},
                yrange=(0, 3.7),
            )

        if "eec" in exp.cfg.plotting.observables:
            LOGGER.info("Calculating EEC")
            gen_eecs = calculate_eec(truth.x_gen, truth.x_gen_ptr)
            det_eecs = calculate_eec(truth.x_det, truth.x_det_ptr)
            sample_eecs = calculate_eec(samples.x_gen, samples.x_gen_ptr)

            plot_ratio_histogram(
                data={
                    "part": gen_eecs[:, 0],
                    "reco": det_eecs[:, 0],
                    model_label: sample_eecs[:, 0],
                },
                reference_key="part",
                no_ratio_keys=["reco"],
                xlabel=r"z",
                ylabel=r"\text{Normalized EEC}",
                bins_range=torch.tensor([1e-6, 1e-1], dtype=torch.float32),
                logx=True,
                logy=True,
                file=file,
                legend_loc="lower right",
                title={"title": r"$t\bar{t}$", "x": 0.18, "y": 0.95},
                weights={
                    "part": gen_eecs[:, 1],
                    "reco": det_eecs[:, 1],
                    model_label: sample_eecs[:, 1],
                },
            )

        for i, j in exp.cfg.plotting.angles:
            gen_dphi = get_dphi(truth.x_gen, truth.x_gen_ptr, i, j)
            det_dphi = get_dphi(truth.x_det, truth.x_det_ptr, i, j)
            sample_dphi = get_dphi(samples.x_gen, samples.x_gen_ptr, i, j)
            plot_ratio_histogram(
                data={
                    "part": gen_dphi,
                    "reco": det_dphi,
                    model_label: sample_dphi,
                },
                reference_key="part",
                xlabel=rf"\Delta \phi_{{{i},{j}}}",
                bins_range=torch.tensor([-0.1, 0.1], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                title={"title": r"$t\bar{t}$", "x": 0.18, "y": 0.95},
            )
            gen_deta = get_deta(truth.x_gen, truth.x_gen_ptr, i, j)
            det_deta = get_deta(truth.x_det, truth.x_det_ptr, i, j)
            sample_deta = get_deta(samples.x_gen, samples.x_gen_ptr, i, j)
            plot_ratio_histogram(
                data={
                    "part": gen_deta,
                    "reco": det_deta,
                    model_label: sample_deta,
                },
                reference_key="part",
                xlabel=rf"\Delta \eta_{{{i},{j}}}",
                bins_range=torch.tensor([-0.1, 0.1], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                title={"title": r"$t\bar{t}$", "x": 0.18, "y": 0.95},
            )
            gen_dr = get_dr(truth.x_gen, truth.x_gen_ptr, i, j)
            det_dr = get_dr(truth.x_det, truth.x_det_ptr, i, j)
            sample_dr = get_dr(samples.x_gen, samples.x_gen_ptr, i, j)
            plot_ratio_histogram(
                data={
                    "part": gen_dr,
                    "reco": det_dr,
                    model_label: sample_dr,
                },
                reference_key="part",
                xlabel=rf"\Delta R_{{{i},{j}}}",
                bins_range=torch.tensor([0, 0.07], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                legend_loc=(
                    "upper right"
                    if i < 3
                    else {"loc": "upper left", "bbox_to_anchor": (0.15, 0.7)}
                ),
                title={
                    "title": r"$t\bar{t}$",
                    "x": 0.2 if i < 3 else 0.85,
                    "y": 0.18 if i < 3 else 0.95,
                },
            )
