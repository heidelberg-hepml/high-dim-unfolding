import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

from experiments.base_plots import plot_loss
from experiments.multiplicity.distributions import (
    GammaMixture,
    GaussianMixture,
    CategoricalDistribution,
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


def plot_mixer(cfg, plot_path, title, plot_dict):
    if cfg.plotting.loss and cfg.train:
        file = f"{plot_path}/loss.pdf"
        plot_loss(
            file,
            [plot_dict["train_loss"], []],
            plot_dict["train_lr"],
            labels=["train loss", "val loss"],
            logy=True,
        )

    if cfg.evaluate:
        file = f"{plot_path}/multiplicity_histograms.pdf"
        plot_mult_histograms(
            file,
            [
                plot_dict["results_train"]["samples"].numpy(),
                plot_dict["results_val"]["samples"].numpy(),
                plot_dict["results_test"]["samples"].numpy(),
            ],
            ["train", "val", "test"],
        )

        if cfg.dist.type == "GammaMixture":
            file = f"{plot_path}/params_histograms.pdf"
            plot_param_histograms(
                file,
                [
                    plot_dict["results_train"]["params"].numpy(),
                    plot_dict["results_val"]["params"].numpy(),
                    plot_dict["results_test"]["params"].numpy(),
                ],
                ["train", "val", "test"],
            )
            file = f"{plot_path}/distributions.pdf"
            plot_distributions(
                file,
                [
                    (
                        plot_dict["results_train"]["params"][
                            : cfg.plotting.n_distributions
                        ],
                        plot_dict["results_train"]["samples"].numpy(),
                    ),
                    (
                        plot_dict["results_val"]["params"][
                            : cfg.plotting.n_distributions
                        ],
                        plot_dict["results_val"]["samples"].numpy(),
                    ),
                    (
                        plot_dict["results_test"]["params"][
                            : cfg.plotting.n_distributions
                        ],
                        plot_dict["results_test"]["samples"].numpy(),
                    ),
                ],
                ["train", "val", "test"],
                x_max=cfg.data.max_num_particles,
                distribution=GammaMixture,
            )
        elif cfg.dist.type == "GaussianMixture":
            file = f"{plot_path}/params_histograms.pdf"
            plot_param_histograms(
                file,
                [
                    plot_dict["results_train"]["params"].numpy(),
                    plot_dict["results_val"]["params"].numpy(),
                    plot_dict["results_test"]["params"].numpy(),
                ],
                ["train", "val", "test"],
            )
            file = f"{plot_path}/distributions.pdf"
            plot_distributions(
                file,
                [
                    (
                        plot_dict["results_train"]["params"][
                            : cfg.plotting.n_distributions
                        ],
                        plot_dict["results_train"]["samples"].numpy(),
                    ),
                    (
                        plot_dict["results_val"]["params"][
                            : cfg.plotting.n_distributions
                        ],
                        plot_dict["results_val"]["samples"].numpy(),
                    ),
                    (
                        plot_dict["results_test"]["params"][
                            : cfg.plotting.n_distributions
                        ],
                        plot_dict["results_test"]["samples"].numpy(),
                    ),
                ],
                ["train", "val", "test"],
                x_max=cfg.data.max_num_particles,
                distribution=GaussianMixture,
            )
        elif cfg.dist.type == "Categorical":
            file = f"{plot_path}/distributions.pdf"
            plot_distributions(
                file,
                [
                    (
                        plot_dict["results_train"]["params"][
                            : cfg.plotting.n_distributions
                        ],
                        plot_dict["results_train"]["samples"].numpy(),
                    ),
                    (
                        plot_dict["results_val"]["params"][
                            : cfg.plotting.n_distributions
                        ],
                        plot_dict["results_val"]["samples"].numpy(),
                    ),
                    (
                        plot_dict["results_test"]["params"][
                            : cfg.plotting.n_distributions
                        ],
                        plot_dict["results_test"]["samples"].numpy(),
                    ),
                ],
                ["train", "val", "test"],
                x_max=cfg.data.max_num_particles,
                distribution="categorical",
            )
        file = f"{plot_path}/main_histogram.pdf"
        plot_histogram(
            file,
            plot_dict["results_train"]["samples"][:, 1].numpy(),
            plot_dict["results_test"]["samples"][:, 1].numpy(),
            plot_dict["results_test"]["samples"][:, 0].numpy(),
            title=title,
            xlabel=r"\text{Multiplicity}",
            xrange=[1, 152],
            model_label="Model",
        )
        file = f"{plot_path}/main_histogram2.pdf"
        plot_histograms(
            file,
            [
                plot_dict["results_test"]["samples"][:, 1].numpy(),
                plot_dict["results_test"]["samples"][:, 0].numpy(),
            ],
            ["target", "prediction"],
        )


def plot_histogram(
    file,
    train,
    test,
    model,
    title,
    xlabel,
    xrange,
    model_label,
    logy=False,
    n_bins=60,
    error_range=[0.85, 1.15],
    error_ticks=[0.9, 1.0, 1.1],
    weights=None,
    mask_dict=None,
):
    """
    Plotting code used for all 1d distributions

    Parameters:
    file: str
    train: np.ndarray of shape (nevents)
    test: np.ndarray of shape (nevents)
    model: np.ndarray of shape (nevents)
    title: str
    xlabel: str
    xrange: tuple with 2 floats
    model_label: str
    logy: bool
    n_bins: int
    error_range: tuple with 2 floats
    error_ticks: tuple with 3 floats
    weights: np.ndarray of shape (nevents)
    mask_dict: dict
        mask (np.ndarray), condition (str)
    """
    # construct labels and colors
    labels = ["Train", "Test", model_label]
    colors = ["black", "#0343DE", "#A52A2A"]
    bins = np.arange(int(xrange[0]), int(xrange[1]) + 1)

    # construct histograms
    y_trn, _ = np.histogram(train, bins=bins, range=xrange)
    y_tst, _ = np.histogram(test, bins=bins)
    y_mod, _ = np.histogram(model, bins=bins)
    hists = [y_trn, y_tst, y_mod]
    hist_errors = [np.sqrt(y_trn), np.sqrt(y_tst), np.sqrt(y_mod)]

    if weights is not None:
        labels.append(f"Rew. {model_label}")
        colors.append("darkorange")
        assert model.shape == weights.shape
        y_weighted = np.histogram(model, bins=bins, weights=weights)[0]
        hists.append(y_weighted)
        hist_errors.append(np.sqrt(y_weighted))

    if mask_dict is not None:
        labels.append(f"{model_label} {mask_dict['condition']}")
        colors.append("violet")
        y_masked = np.histogram(model[mask_dict["mask"]], bins=bins)[0]
        hists.append(y_masked)
        hist_errors.append(np.sqrt(y_masked))

    integrals = [np.sum((bins[1:] - bins[:-1]) * y) for y in hists]
    scales = [1 / integral if integral != 0.0 else 1.0 for integral in integrals]

    dup_last = lambda a: np.append(a, a[-1])

    if mask_dict is None:
        fig, axs = plt.subplots(
            3,
            1,
            sharex=True,
            figsize=(6, 4),
            gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.00},
        )
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        axs = [ax]

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

        if label == "Train":
            axs[0].fill_between(
                bins,
                dup_last(y) * scale,
                0.0 * dup_last(y),
                facecolor=color,
                alpha=0.1,
                step="post",
            )
            continue

        if mask_dict is not None:
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

        delta = np.fabs(ratio - 1) * 100
        delta_err = ratio_err * 100

        markers, caps, bars = axs[2].errorbar(
            (bins[:-1] + bins[1:]) / 2,
            delta,
            yerr=delta_err,
            ecolor=color,
            color=color,
            elinewidth=0.5,
            linewidth=0,
            fmt=".",
            capsize=2,
        )
        [cap.set_alpha(0.5) for cap in caps]
        [bar.set_alpha(0.5) for bar in bars]

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
    axs[0].text(
        0.02,
        0.95,
        s=title,
        horizontalalignment="left",
        verticalalignment="top",
        transform=axs[0].transAxes,
        fontsize=FONTSIZE,
    )

    if mask_dict is None:
        axs[1].set_ylabel(
            r"$\frac{\mathrm{{%s}}}{\mathrm{Test}}$" % model_label, fontsize=FONTSIZE
        )
        axs[1].set_yticks(error_ticks)
        axs[1].set_ylim(error_range)
        axs[1].axhline(y=error_ticks[0], c="black", ls="dotted", lw=0.5)
        axs[1].axhline(y=error_ticks[1], c="black", ls="--", lw=0.7)
        axs[1].axhline(y=error_ticks[2], c="black", ls="dotted", lw=0.5)

        axs[2].set_ylim((0.05, 20))
        axs[2].set_yscale("log")
        axs[2].set_yticks([0.1, 1.0, 10.0])
        axs[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
        axs[2].set_yticks(
            [
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
            ],
            minor=True,
        )

        axs[2].axhline(y=1.0, linewidth=0.5, linestyle="--", color="grey")
        axs[2].axhspan(0, 1.0, facecolor="#cccccc", alpha=0.3)
        axs[2].set_ylabel(r"$\delta [\%]$", fontsize=FONTSIZE)

        axs[1].tick_params(axis="both", labelsize=TICKLABELSIZE)
        axs[2].tick_params(axis="both", labelsize=TICKLABELSIZE)

    plt.savefig(file, bbox_inches="tight", format="pdf")
    plt.close()


def plot_histograms(
    file,
    data,
    labels,
    bins=60,
    xlabel=None,
    title=None,
    logx=False,
    logy=False,
    xrange=None,
    ratio_range=[0.85, 1.15],
    ratio_ticks=[0.9, 1.0, 1.1],
):
    hists = []
    bins = np.arange(1, np.max(np.array(data)) + 1)
    for dat in data:
        hist, bins = np.histogram(dat, bins=bins, range=xrange)
        hists.append(hist)
    integrals = [np.sum((bins[1:] - bins[:-1]) * hist) for hist in hists]
    scales = [1 / integral if integral != 0.0 else 1.0 for integral in integrals]
    dup_last = lambda a: np.append(a, a[-1])

    fig, axs = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(6, 4),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0},
    )
    for i, hist, scale, label, color in zip(
        range(len(hists)), hists, scales, labels, colors
    ):
        axs[0].step(
            bins,
            dup_last(hist) * scale,
            label=label,
            color=color,
            linewidth=1.0,
            where="post",
        )
        if i == 0:
            axs[0].fill_between(
                bins,
                dup_last(hist) * scale,
                0.0 * dup_last(hist),
                facecolor=color,
                alpha=0.1,
                step="post",
            )
            continue

        ratio = np.divide(
            hist * scale, hists[0] * scales[0], where=hists[0] * scales[0] != 0
        )  # sets denominator=0 terms to 0
        axs[1].step(bins, dup_last(ratio), linewidth=1.0, where="post", color=color)

    if logx:
        axs[0].set_xscale("log")

    axs[0].legend(loc="upper right", frameon=False, fontsize=FONTSIZE_LEGEND)
    axs[0].set_ylabel("Normalized", fontsize=FONTSIZE)
    axs[1].set_xlabel(xlabel, fontsize=FONTSIZE)

    _, ymax = axs[0].get_ylim()
    axs[0].set_ylim(0.0, ymax)
    axs[0].tick_params(axis="both", labelsize=FONTSIZE_TICK)
    axs[1].tick_params(axis="both", labelsize=FONTSIZE_TICK)
    axs[0].text(
        0.04,
        0.95,
        s=title,
        horizontalalignment="left",
        verticalalignment="top",
        transform=axs[0].transAxes,
        fontsize=FONTSIZE,
    )

    axs[1].set_yticks(ratio_ticks)
    axs[1].set_ylim(ratio_range)
    axs[1].axhline(y=ratio_ticks[0], c="black", ls="dotted", lw=0.5)
    axs[1].axhline(y=ratio_ticks[1], c="black", ls="--", lw=0.7)
    axs[1].axhline(y=ratio_ticks[2], c="black", ls="dotted", lw=0.5)

    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()


def plot_param_histograms(
    file,
    data,
    labels,
    bins=60,
    xlabel=None,
    title=None,
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


def plot_mult_histograms(
    file,
    data,
    labels,
):
    bins_max = int(max([np.max(dat).item() for dat in data]))
    bins = np.arange(1, bins_max + 1)
    hists = []
    for dat in data:
        hist_sample, _ = np.histogram(dat[:, 0], bins=bins)
        hist_target, _ = np.histogram(dat[:, 1], bins=bins)
        hist_diff_sample, bins_diff = np.histogram(dat[:, 0] - dat[:, 2], bins=60)
        hist_diff_target, _ = np.histogram(dat[:, 1] - dat[:, 2], bins=bins_diff)
        hists.append(
            [
                hist_sample / dat.shape[0],
                hist_target / dat.shape[0],
                hist_diff_sample / dat.shape[0],
                hist_diff_target / dat.shape[0],
            ]
        )

    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    dup_last = lambda a: np.append(a, a[-1])
    for i, label in enumerate(labels):
        axs[0, i].step(
            bins, dup_last(hists[i][0]), label="sample", linewidth=1.0, where="post"
        )
        axs[0, i].step(
            bins, dup_last(hists[i][1]), label="target", linewidth=1.0, where="post"
        )
        axs[0, i].legend(loc="upper right", frameon=False, fontsize=FONTSIZE_LEGEND)
        axs[0, i].set_title(f"{label} dataset", fontsize=FONTSIZE)
        axs[1, i].step(
            bins_diff,
            dup_last(hists[i][2]),
            label="sample",
            linewidth=1.0,
            where="post",
        )
        axs[1, i].step(
            bins_diff,
            dup_last(hists[i][3]),
            label="target",
            linewidth=1.0,
            where="post",
        )
        axs[1, i].legend(loc="upper right", frameon=False, fontsize=FONTSIZE_LEGEND)
        axs[1, i].set_title(f"{label} dataset", fontsize=FONTSIZE)

    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()


def plot_distributions(file, data, labels, x_max, distribution, n_plots=6):

    fig, axs = plt.subplots(n_plots + 1, 3, figsize=(15, 20))

    for i, dat, label in zip(range(3), data, labels):
        params, target = dat
        if distribution == "categorical":
            bins = np.arange(1, x_max + 1)
            for logits in params:
                axs[0, i].step(bins, logits)
        else:
            x = torch.linspace(0, x_max, 1000).reshape(-1, 1).repeat(1, len(params))
            dist = distribution(params)
            pdf = dist.log_prob(x).exp().detach().numpy()
            for j in range(len(params)):
                axs[0, i].plot(x[:, j], pdf[:, j])
        axs[0, i].set_title(f"{label} dataset", fontsize=FONTSIZE)

    params = data[0][0]
    target = data[0][1]
    for i in range(3):
        for j in range(n_plots):
            if distribution == "categorical":
                axs[j + 1, i].step(bins, params[i + 3 * j], label="predicted mult dist")
            else:
                x = torch.linspace(0, x_max, 1000).reshape(-1, 1).repeat(1, len(params))
                dist = distribution(params)
                pdf = dist.log_prob(x).exp().detach().numpy()
                axs[j + 1, i].plot(
                    x[:, i + 3 * j], pdf[:, i + 3 * j], label=f"predicted mult dist"
                )
            axs[j + 1, i].axvline(target[i + 3 * j, 1], c="red", label="target mult")
            axs[j + 1, i].axvline(target[i + 3 * j, 2], c="green", label="base mult")
            axs[j + 1, i].legend(
                loc="upper right", frameon=False, fontsize=FONTSIZE_LEGEND
            )

    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()
