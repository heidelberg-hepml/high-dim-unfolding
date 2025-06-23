import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from experiments.unfolding.coordinates import LogPtPhiEtaLogM2
from experiments.unfolding.utils import get_range, fourmomenta_to_jetmomenta

# load fonts
import matplotlib.font_manager as font_manager

font_dir = ["src/utils/bitstream-charter-ttf/Charter/"]
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)
font_manager.findSystemFonts(fontpaths=None, fontext="ttf")

# setup matplotlib
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath}"
)

# fontsize
FONTSIZE = 18
FONTSIZE_LEGEND = FONTSIZE
TICKLABELSIZE = 10


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
    labels = ["Particle-level", "Detector-level", model_label]
    colors = ["black", "#0343DE", "#A52A2A"]

    # construct histograms
    y_trn, bins = np.histogram(train, bins=n_bins, range=xrange)
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

        if label == "Particle-level":
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

        if label == model_label:
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
    axs[0].set_ylabel("Density", fontsize=FONTSIZE)

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


def plot_histogram_2d(
    file,
    test,
    model,
    title,
    xlabel,
    ylabel,
    xrange,
    yrange,
    model_label,
    n_bins=100,
):
    data = [test, model]
    weights = [None, None]
    subtitles = ["Test", model_label]

    fig, axs = plt.subplots(1, len(data), figsize=(4 * len(data), 4))
    for ax, dat, weight, subtitle in zip(axs, data, weights, subtitles):
        ax.set_title(subtitle)
        ax.hist2d(
            dat[:, 0],
            dat[:, 1],
            bins=n_bins,
            range=[xrange, yrange],
            rasterized=True,
            weights=weight,
        )
        ax.set_xlabel(r"${%s}$" % xlabel)
        ax.set_ylabel(r"${%s}$" % ylabel)
    fig.suptitle(title)
    plt.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()


def plot_calibration(file, prob_true, prob_pred):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(
        prob_true, prob_pred, color="#A52A2A", marker="o", markersize=3, linewidth=1
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("classifier probability for true events", fontsize=FONTSIZE)
    ax.set_ylabel("true fraction of true events", fontsize=FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICKLABELSIZE)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    plt.savefig(file, bbox_inches="tight", format="pdf")
    plt.close()


def plot_roc(file, tpr, fpr, auc):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#A52A2A", linewidth=1.0)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("false positive rate", fontsize=FONTSIZE)
    ax.set_ylabel("true positive rate", fontsize=FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICKLABELSIZE)
    ax.text(
        0.95,
        0.05,
        f"AUC = {auc:.4f}",
        verticalalignment="bottom",
        horizontalalignment="right",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )
    plt.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()


def simple_histogram(
    file, data, labels, xrange, xlabel, logx=False, logy=False, n_bins=80
):
    assert len(data) == 2 and len(labels) == 2
    colors = ["#0343DE", "#A52A2A"]
    dup_last = lambda a: np.append(a, a[-1])

    data = [np.clip(data_i.clone(), xrange[0], xrange[1]) for data_i in data]
    if logx:
        data = [np.log(data_i) for data_i in data]
        xrange = np.log(xrange)

    bins = np.histogram(data[0], bins=n_bins, range=xrange)[1]
    hists = [np.histogram(data_i, bins=bins, range=xrange)[0] for data_i in data]
    integrals = [np.sum((bins[1:] - bins[:-1]) * y) for y in hists]
    scales = [1 / integral if integral != 0.0 else 1.0 for integral in integrals]
    if logx:
        bins = np.exp(bins)
        xrange = np.exp(xrange)

    fig, ax = plt.subplots(figsize=(5, 4))
    for y, scale, label, color in zip(hists, scales, labels, colors):
        ax.step(
            bins,
            dup_last(y) * scale,
            label=label,
            color=color,
            linewidth=1.0,
            where="post",
        )
    ax.legend(loc="upper right", frameon=False, fontsize=FONTSIZE_LEGEND)
    ax.set_ylabel("Density", fontsize=FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)

    if logy:
        ax.set_yscale("log")
    else:
        _, ymax = ax.get_ylim()
        ax.set_ylim(0.0, ymax)
    if logx:
        ax.set_xscale("log")
    ax.set_xlim(xrange)
    ax.tick_params(axis="both", labelsize=TICKLABELSIZE)
    plt.savefig(file, bbox_inches="tight", format="pdf")
    plt.close()


def plot_kinematics(path, samples, targets, base):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    labels = ["Energy", "p_x", "p_y", "p_z"]
    xrange = [[0, 1000], [-400, 400], [-400, 400], [-750, 750]]
    for i, ax in enumerate(axs.flatten()):
        bins = np.linspace(xrange[i][0], xrange[i][1], 100)
        ax.hist(
            samples[:, i].cpu(),
            bins=bins,
            range=None,
            label="samples",
            density=True,
            histtype="step",
        )
        ax.hist(
            targets[:, i].cpu(),
            bins=bins,
            range=None,
            alpha=0.5,
            label="targets",
            density=True,
            histtype="step",
        )
        ax.hist(
            base[:, i].cpu(),
            bins=bins,
            range=None,
            alpha=0.5,
            label="base",
            density=True,
            histtype="step",
        )
        ax.set_xlabel(labels[i], fontsize=FONTSIZE)
        ax.legend(loc="upper right", frameon=False, fontsize=FONTSIZE_LEGEND)
    plt.tight_layout()
    plt.savefig(path + "/kinematics.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    jet_samples = fourmomenta_to_jetmomenta(samples)
    jet_targets = fourmomenta_to_jetmomenta(targets)
    jet_base = fourmomenta_to_jetmomenta(base)
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    labels = ["pt", "phi", "eta", "m"]
    xrange = [[300, 1000], [-np.pi, np.pi], [-3, 3], [0, 300]]
    for i, ax in enumerate(axs.flatten()):
        bins = np.linspace(xrange[i][0], xrange[i][1], 100)
        ax.hist(
            jet_samples[:, i].cpu(),
            bins=bins,
            range=None,
            label="samples",
            density=True,
            histtype="step",
        )
        ax.hist(
            jet_targets[:, i].cpu(),
            bins=bins,
            range=None,
            alpha=0.5,
            label="targets",
            density=True,
            histtype="step",
        )
        ax.hist(
            jet_base[:, i].cpu(),
            bins=bins,
            range=None,
            alpha=0.5,
            label="base",
            density=True,
            histtype="step",
        )
        ax.set_xlabel(labels[i], fontsize=FONTSIZE)
        ax.legend(loc="upper right", frameon=False, fontsize=FONTSIZE_LEGEND)
    plt.tight_layout()
    plt.savefig(path + "/kinematics_jet.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    coords = LogPtPhiEtaLogM2(pt_min=0.0, units=1.0)
    samples = coords.fourmomenta_to_x(samples)
    targets = coords.fourmomenta_to_x(targets)
    base = coords.fourmomenta_to_x(base)
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    labels = ["log pt", "phi", "eta", "log m2"]
    xrange = [[-10, 10], [-np.pi, np.pi], [-3, 3], [-5, -4.2]]
    for i, ax in enumerate(axs.flatten()):
        bins = np.linspace(xrange[i][0], xrange[i][1], 100)
        ax.hist(
            samples[:, i].cpu(),
            bins=bins,
            range=None,
            label="samples",
            density=True,
            histtype="step",
        )
        ax.hist(
            targets[:, i].cpu(),
            bins=bins,
            range=None,
            alpha=0.5,
            label="targets",
            density=True,
            histtype="step",
        )
        ax.hist(
            base[:, i].cpu(),
            bins=bins,
            range=None,
            alpha=0.5,
            label="base",
            density=True,
            histtype="step",
        )
        ax.set_xlabel(labels[i], fontsize=FONTSIZE)
        ax.legend(loc="upper right", frameon=False, fontsize=FONTSIZE_LEGEND)
    plt.tight_layout()
    plt.savefig(path + "/kinematics_prep.pdf", format="pdf", bbox_inches="tight")
    plt.close()


def plot_2d_histogram(file, x1, y1, x2, y2, xlabel, ylabel, range, model_label):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    if isinstance(range, tuple):
        xrange = range[0]
        yrange = range[1]
    else:
        xrange = range
        yrange = range
    bins = (
        np.linspace(xrange[0], xrange[1], 50),
        np.linspace(yrange[0], yrange[1], 50),
    )
    hist = ax1.hist2d(x1, y1, bins=bins, norm=mcolors.LogNorm(), rasterized=True)
    vmin, vmax = hist[-1].get_clim()
    ax1.set_xlabel(
        r"${%s}$" % xlabel,
        fontsize=FONTSIZE,
    )
    ax1.set_ylabel(
        r"${%s}$" % ylabel,
        fontsize=FONTSIZE,
    )
    ax1.set_title("Truth", fontsize=FONTSIZE)
    ax1.grid(False)

    ax2.hist2d(
        x2, y2, bins=bins, norm=mcolors.LogNorm(vmax=vmax, vmin=vmin), rasterized=True
    )
    ax2.set_xlabel(
        r"${%s}$" % xlabel,
        fontsize=FONTSIZE,
    )
    ax2.set_ylabel(
        r"${%s}$" % ylabel,
        fontsize=FONTSIZE,
    )
    ax2.set_title(model_label, fontsize=FONTSIZE)
    ax2.grid(False)

    plt.savefig(file, bbox_inches="tight", format="pdf")
    plt.close()


def plot_data(gen_jet, det_jet, filename):
    gen_jet, det_jet = gen_jet.cpu().detach(), det_jet.cpu().detach()
    if gen_jet.ndim == 3:
        gen_jet = gen_jet.reshape(-1, 4)
    if det_jet.ndim == 3:
        det_jet = det_jet.reshape(-1, 4)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    # xlims = [[0, 600], [-3, 3], [-3.5, 3.5], [0, 1500]]

    for i in range(4):
        xlims = np.array(get_range([gen_jet[..., i], det_jet[..., i]]))
        bins = np.linspace(xlims[0], xlims[1], 100)
        axs[i // 2, i % 2].hist(
            gen_jet[:, i],
            bins=bins,
            alpha=0.5,
            label="gen_jet",
            histtype="step",
            density=True,
        )
        axs[i // 2, i % 2].hist(
            det_jet[:, i],
            bins=bins,
            alpha=0.5,
            label="det_jet",
            histtype="step",
            density=True,
        )
        axs[i // 2, i % 2].set_title(f"Variable {i}")
        axs[i // 2, i % 2].set_xlim(xlims)
        axs[i // 2, i % 2].legend()
    axs[0, 0].set_yscale("log")
    axs[1, 1].set_yscale("log")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", format="pdf")
    plt.close()
