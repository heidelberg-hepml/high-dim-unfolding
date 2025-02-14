import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

from experiments.base_plots import plot_loss
from experiments.multiplicity.distributions import (
    CategoricalDistribution,
    GammaMixture,
    GaussianMixture,
)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath} \usepackage{siunitx}"
)

FONTSIZE = 14
FONTSIZE_LEGEND = 13
FONTSIZE_TICK = 12
TICKLABELSIZE = 10

colors = ["black", "#0343DE", "#A52A2A", "darkorange"]


def plot_mixer(cfg, plot_path, plot_dict):
    if cfg.plotting.loss and cfg.train:
        file = f"{plot_path}/loss.pdf"
        if cfg.training.validate_every_n_steps < cfg.training.iterations:
            plot_loss(
                file,
                [plot_dict["train_loss"], plot_dict["val_loss"]],
                plot_dict["train_lr"],
                labels=["train loss", "val loss"],
                logy=True,
            )
        else:
            plot_loss(
                file,
                [plot_dict["train_loss"], []],
                plot_dict["train_lr"],
                labels=["train loss", "val loss"],
                logy=True,
            )

    if cfg.evaluate:
        if cfg.plotting.distributions:
            file = f"{plot_path}/distributions.pdf"
            if cfg.dist.diff:
                xrange = [-15, 40]
            else:
                xrange = [0, 85]
            plot_distributions(
                file,
                plot_dict["results_test"]["params"][: cfg.plotting.n_distributions],
                plot_dict["results_test"]["samples"].numpy(),
                xrange=xrange,
                distribution_label=cfg.dist.type,
                diff=cfg.dist.diff,
                diff_min=cfg.data.diff[0],
            )

        if cfg.plotting.histogram:
            file = f"{plot_path}/main_histogram.pdf"
            plot_histogram(
                file,
                plot_dict["results_test"]["samples"][:, 1].numpy(),
                plot_dict["results_test"]["samples"][:, 0].numpy(),
                xlabel=r"\text{Multiplicity}",
                xrange=[0, 85],
                model_label=cfg.model.net._target_.rsplit(".", 1)[-1],
            )
        if cfg.plotting.diff:
            file = f"{plot_path}/diff_histogram.pdf"
            plot_histogram(
                file,
                plot_dict["results_test"]["samples"][:, 1].numpy()
                - plot_dict["results_test"]["samples"][:, 2].numpy(),
                plot_dict["results_test"]["samples"][:, 0].numpy()
                - plot_dict["results_test"]["samples"][:, 2].numpy(),
                xlabel=r"\text{Multiplicity difference}",
                xrange=[-15, 40],
                model_label=cfg.model.net._target_.rsplit(".", 1)[-1],
            )


def plot_histogram(
    file,
    train,
    model,
    xlabel,
    xrange,
    model_label,
    logy=False,
    error_range=[0.85, 1.15],
    error_ticks=[0.9, 1.0, 1.1],
):
    """
    Plotting code used for all 1d distributions

    Parameters:
    file: str
    train: np.ndarray of shape (nevents)
    model: np.ndarray of shape (nevents)
    xlabel: str
    xrange: tuple with 2 floats
    model_label: str
    logy: bool
    n_bins: int
    error_range: tuple with 2 floats
    error_ticks: tuple with 3 floats
    """
    # construct labels and colors
    labels = ["Truth", model_label]
    colors = ["black", "#A52A2A"]
    bins = np.arange(int(xrange[0]), int(xrange[1]) + 1)

    # construct histograms
    y_trn, _ = np.histogram(train, bins=bins, range=xrange)
    y_mod, _ = np.histogram(model, bins=bins)
    hists = [y_trn, y_mod]
    hist_errors = [np.sqrt(y_trn), np.sqrt(y_mod)]

    integrals = [np.sum((bins[1:] - bins[:-1]) * y) for y in hists]
    scales = [1 / integral if integral != 0.0 else 1.0 for integral in integrals]

    dup_last = lambda a: np.append(a, a[-1])

    fig, axs = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(6, 4),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.00},
    )

    for i, y, y_err, scale, label, color in zip(
        range(len(hists)), hists, hist_errors, scales, labels, colors
    ):

        axs[0].step(
            bins,
            dup_last(y) * scale,
            label=label,
            color=color,
            linewidth=1.0,
            where="post",
        )
        axs[0].step(
            bins,
            dup_last(y + y_err) * scale,
            color=color,
            alpha=0.5,
            linewidth=0.5,
            where="post",
        )
        axs[0].step(
            bins,
            dup_last(y - y_err) * scale,
            color=color,
            alpha=0.5,
            linewidth=0.5,
            where="post",
        )
        axs[0].fill_between(
            bins,
            dup_last(y - y_err) * scale,
            dup_last(y + y_err) * scale,
            facecolor=color,
            alpha=0.3,
            step="post",
        )

        if label == "Truth":
            axs[0].fill_between(
                bins,
                dup_last(y) * scale,
                0.0 * dup_last(y),
                facecolor=color,
                alpha=0.1,
                step="post",
            )
            continue

        ratio = (y * scale) / (hists[0] * scales[0])
        ratio_err = np.sqrt((y_err / y) ** 2 + (hist_errors[0] / hists[0]) ** 2)
        ratio_isnan = np.isnan(ratio)
        ratio[ratio_isnan] = 1.0
        ratio_err[ratio_isnan] = 0.0

        axs[1].step(bins, dup_last(ratio), linewidth=1.0, where="post", color=color)
        axs[1].step(
            bins,
            dup_last(ratio + ratio_err),
            color=color,
            alpha=0.5,
            linewidth=0.5,
            where="post",
        )
        axs[1].step(
            bins,
            dup_last(ratio - ratio_err),
            color=color,
            alpha=0.5,
            linewidth=0.5,
            where="post",
        )
        axs[1].fill_between(
            bins,
            dup_last(ratio - ratio_err),
            dup_last(ratio + ratio_err),
            facecolor=color,
            alpha=0.3,
            step="post",
        )

    axs[0].legend(loc="upper right", frameon=False, fontsize=FONTSIZE_LEGEND)
    axs[0].set_ylabel("Normalized", fontsize=FONTSIZE)

    if logy:
        axs[0].set_yscale("log")

    _, ymax = axs[0].get_ylim()
    axs[0].set_ylim(0.0, ymax)
    axs[0].set_xlim(xrange)
    axs[0].tick_params(axis="both", labelsize=TICKLABELSIZE)
    plt.xlabel(
        r"${%s}$" % xlabel,
        fontsize=FONTSIZE,
    )

    axs[1].set_ylabel(
        r"$\frac{\mathrm{{%s}}}{\mathrm{Truth}}$" % model_label, fontsize=FONTSIZE
    )
    axs[1].set_yticks(error_ticks)
    axs[1].set_ylim(error_range)
    axs[1].axhline(y=error_ticks[0], c="black", ls="dotted", lw=0.5)
    axs[1].axhline(y=error_ticks[1], c="black", ls="--", lw=0.7)
    axs[1].axhline(y=error_ticks[2], c="black", ls="dotted", lw=0.5)

    axs[1].tick_params(axis="both", labelsize=TICKLABELSIZE)

    plt.savefig(file, bbox_inches="tight", format="pdf")
    plt.close()


def plot_param_histograms(
    file,
    data,
    labels,
    bins=60,
    xlabel=None,
    logx=False,
    logy=False,
    xrange=None,
):
    params_labels = ["k", "theta", "coeff"]
    hists = []
    bins_list = []
    fill = True
    for params in data:
        hists_mix = []
        for mixture_component in range(params.shape[1]):
            hist_k, bins_k = np.histogram(
                params[:, mixture_component, 0], bins=bins, range=xrange
            )
            hist_theta, bins_theta = np.histogram(
                params[:, mixture_component, 1], bins=bins, range=xrange
            )
            hist_coeff, bins_coeff = np.histogram(
                params[:, mixture_component, 2], bins=bins, range=xrange
            )
            hists_mix.append([hist_k, hist_theta, hist_coeff])
            if fill:
                bins_list.append(bins_k)
                bins_list.append(bins_theta)
                bins_list.append(bins_coeff)
                fill = False
        hists.append(hists_mix)

    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    dup_last = lambda a: np.append(a, a[-1])

    for i, hists_mix in enumerate(hists):
        for i_mix, hists_param in enumerate(hists_mix):
            for j, hist in enumerate(hists_param):
                axs[j, i].step(
                    bins_list[j],
                    dup_last(hist),
                    label=f"{i_mix+1}",
                    linewidth=1.0,
                    where="post",
                )
                axs[j, i].legend(
                    loc="upper right", frameon=False, fontsize=FONTSIZE_LEGEND
                )
                axs[j, i].set_title(
                    f"{labels[i]} dataset, component {params_labels[j]}",
                    fontsize=FONTSIZE,
                )

    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()


def plot_distributions(
    file, params, samples, xrange, distribution_label, diff, diff_min, n_plots=5
):

    with PdfPages(file) as pdf:

        if distribution_label == "GammaMixture":
            distribution = GammaMixture
        elif distribution_label == "GaussianMixture":
            distribution = GaussianMixture
        elif distribution_label == "Categorical":
            distribution = CategoricalDistribution

        fig, ax = plt.subplots(figsize=(6, 4))
        if distribution_label == "Categorical":
            bins = np.arange(xrange[0], xrange[1] + 1)
            if diff:
                for logits in params:
                    ax.step(bins, logits[bins - diff_min] / logits.sum(), linewidth=0.5)
            else:
                for logits in params:
                    ax.step(bins, logits[bins] / logits.sum(), linewidth=0.5)
        else:
            x = (
                torch.linspace(xrange[0], xrange[1], 1000)
                .reshape(-1, 1)
                .repeat(1, len(params))
            )
            dist = distribution(params)
            density = dist.log_prob(x).exp().detach().numpy()
            for j in range(len(params)):
                ax.plot(x[:, j], density[:, j], linewidth=0.5)
        ax.set_xlabel("Multiplicity", fontsize=FONTSIZE)
        ax.set_ylabel("Probability", fontsize=FONTSIZE)
        ax.set_ylim((0, 0.15))
        pdf.savefig(fig, bbox_inches="tight")

        for i in range(n_plots):
            fig, ax = plt.subplots(figsize=(6, 4))
            if distribution == CategoricalDistribution:
                if diff:
                    ax.step(
                        bins,
                        params[i][bins - diff_min] / params[i].sum(),
                        label="Predicted distribution",
                        color=colors[3],
                    )
                else:
                    ax.step(
                        bins,
                        params[i][bins] / params[i].sum(),
                        label="Predicted distribution",
                        color=colors[3],
                    )
            else:
                x = torch.linspace(xrange[0], xrange[1], 1000).reshape(-1, 1)
                dist = distribution(params[i].unsqueeze(0))
                density = dist.log_prob(x).exp().detach().numpy()
                ax.plot(x, density, label=f"Predicted distribution", color=colors[3])
            if diff:
                ax.axvline(
                    samples[i, 1] - samples[i, 2],
                    c=colors[1],
                    label="Truth",
                    linestyle="dashed",
                )
            else:
                ax.axvline(
                    samples[i, 1],
                    c=colors[1],
                    label="Particle-level multiplicity",
                    linestyle="dashed",
                )
                ax.axvline(
                    samples[i, 2],
                    c=colors[2],
                    label="Detector-level multiplicity",
                    linestyle="dashed",
                )
            ax.legend(loc="upper right", frameon=False, fontsize=FONTSIZE_LEGEND)
            ax.set_xlabel("Multiplicity", fontsize=FONTSIZE)
            ax.set_ylabel("Probability", fontsize=FONTSIZE)
            ax.set_ylim((0, 0.15))
            pdf.savefig(fig, bbox_inches="tight")
