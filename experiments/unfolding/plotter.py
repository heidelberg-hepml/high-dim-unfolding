import numpy as np
import math
from matplotlib.backends.backend_pdf import PdfPages


from experiments.base_plots import plot_loss, plot_metric
from experiments.unfolding.plots import (
    plot_histogram,
    plot_calibration,
    simple_histogram,
    plot_roc,
    plot_correlations,
)
from experiments.unfolding.utils import get_range, fourmomenta_to_jetmomenta


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


def plot_fourmomenta(exp, filename, model_label, weights=None, mask_dict=None):

    with PdfPages(filename) as file:
        for name in exp.obs.keys():
            extract = exp.obs[name]
            det_lvl = (
                extract(
                    exp.data_raw["gen"].x_det,
                    exp.data_raw["gen"].x_det_batch,
                    exp.data_raw["gen"].x_gen_batch,
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
                    exp.data_raw["gen"].x_gen,
                    exp.data_raw["gen"].x_gen_batch,
                    exp.data_raw["gen"].x_det_batch,
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
            # xranges = exp.obs_ranges[name]["fourmomenta"]
            for channel in range(4):
                xlabel = obs_names[channel]
                # xrange = xranges[channel]
                xrange = np.array(
                    get_range(
                        [
                            part_lvl[..., channel],
                            det_lvl[..., channel],
                            model[..., channel],
                        ]
                    )
                )
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
        for name in exp.obs.keys():
            extract = exp.obs[name]
            det_lvl = extract(
                exp.data_raw["gen"].x_det,
                exp.data_raw["gen"].x_det_batch,
                exp.data_raw["gen"].x_gen_batch,
            )
            part_lvl = extract(
                exp.data_raw["truth"].x_gen,
                exp.data_raw["truth"].x_gen_batch,
                exp.data_raw["truth"].x_det_batch,
            )[: len(det_lvl)]
            model = extract(
                exp.data_raw["gen"].x_gen,
                exp.data_raw["gen"].x_gen_batch,
                exp.data_raw["gen"].x_det_batch,
            )
            part_lvl = fourmomenta_to_jetmomenta(part_lvl).cpu().detach()
            det_lvl = fourmomenta_to_jetmomenta(det_lvl).cpu().detach()
            model = fourmomenta_to_jetmomenta(model).cpu().detach()
            obs_names = [
                r"p_{T," + name + "}",
                "\phi_{" + name + "}",
                "\eta_{" + name + "}",
                "m^2_{" + name + "}",
            ]
            # xranges = exp.obs_ranges[name]["jetmomenta"]
            for channel in range(4):
                xlabel = obs_names[channel]
                # xrange = xranges[channel]
                xrange = np.array(
                    get_range(
                        [
                            part_lvl[..., channel],
                            det_lvl[..., channel],
                            model[..., channel],
                        ]
                    )
                )
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
        for name in exp.obs.keys():
            extract = exp.obs[name]
            det_lvl = extract(
                exp.data_raw["gen"].x_det,
                exp.data_raw["gen"].x_det_batch,
                exp.data_raw["gen"].x_gen_batch,
            )
            part_lvl = extract(
                exp.data_raw["truth"].x_gen,
                exp.data_raw["truth"].x_gen_batch,
                exp.data_raw["truth"].x_det_batch,
            )[: len(det_lvl)]
            model = extract(
                exp.data_raw["gen"].x_gen,
                exp.data_raw["gen"].x_gen_batch,
                exp.data_raw["gen"].x_det_batch,
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
            # xranges = exp.obs_ranges[name][coords.__class__.__name__]
            for channel in range(4):
                xlabel = obs_names[channel]
                # xrange = xranges[channel]
                xrange = np.array(
                    get_range(
                        [
                            part_lvl[..., channel],
                            det_lvl[..., channel],
                            model[..., channel],
                        ]
                    )
                )
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
