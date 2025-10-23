from operator import truth
import torch
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from torch_geometric.utils import scatter

from experiments.base_plots import (
    plot_loss,
    plot_metric,
    plot_ratio_histogram,
)
from experiments.kinematics.observables import (
    calculate_eec,
    tau,
    sd_mass,
    compute_zg,
    NSUB_AVAIL,
    SOFTDROP_AVAIL,
)
from experiments.utils import get_range, get_pt, get_phi, get_eta, get_mass
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
            gen_tau1 = tau(truth.x_gen, truth.x_gen_batch, N=1, R0=0.4, axis_mode=1)
            det_tau1 = tau(truth.x_det, truth.x_det_batch, N=1, R0=0.4, axis_mode=1)
            sample_tau1 = tau(
                samples.x_gen, samples.x_gen_batch, N=1, R0=0.4, axis_mode=1
            )

            gen_tau2 = tau(truth.x_gen, truth.x_gen_batch, N=2, R0=0.4, axis_mode=1)
            det_tau2 = tau(truth.x_det, truth.x_det_batch, N=2, R0=0.4, axis_mode=1)
            sample_tau2 = tau(
                samples.x_gen, samples.x_gen_batch, N=2, R0=0.4, axis_mode=1
            )

            gen_tau3 = tau(truth.x_gen, truth.x_gen_batch, N=3, R0=0.4, axis_mode=1)
            det_tau3 = tau(truth.x_det, truth.x_det_batch, N=3, R0=0.4, axis_mode=1)
            sample_tau3 = tau(
                samples.x_gen, samples.x_gen_batch, N=3, R0=0.4, axis_mode=1
            )

            gen_tau21 = torch.where(
                gen_tau1 > 0, gen_tau2 / gen_tau1, torch.tensor(0.0)
            )
            det_tau21 = torch.where(
                det_tau1 > 0, det_tau2 / det_tau1, torch.tensor(0.0)
            )
            sample_tau21 = torch.where(
                sample_tau1 > 0, sample_tau2 / sample_tau1, torch.tensor(0.0)
            )

            gen_tau32 = torch.where(
                gen_tau2 > 0, gen_tau3 / gen_tau2, torch.tensor(0.0)
            )
            det_tau32 = torch.where(
                det_tau2 > 0, det_tau3 / det_tau2, torch.tensor(0.0)
            )
            sample_tau32 = torch.where(
                sample_tau2 > 0, sample_tau3 / sample_tau2, torch.tensor(0.0)
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
                yrange=[0, 17],
            )

            plot_ratio_histogram(
                data={
                    "part": gen_tau3,
                    "reco": det_tau3,
                    model_label: sample_tau3,
                },
                reference_key="part",
                xlabel=r"\tau_3^{\beta=1}",
                bins_range=torch.tensor([0, 0.18], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                title={"title": "Z+jets", "x": 0.18, "y": 0.95},
                yrange=[0, 21],
            )

            plot_ratio_histogram(
                data={
                    "part": gen_tau21,
                    "reco": det_tau21,
                    model_label: sample_tau21,
                },
                reference_key="part",
                xlabel=r"\tau_{21}^{\beta=1}",
                bins_range=torch.tensor([0.1, 0.93], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                legend_loc=None,
                file=file,
                title={"title": "Z+jets", "x": 0.18, "y": 0.95},
            )

            plot_ratio_histogram(
                data={
                    "part": gen_tau32,
                    "reco": det_tau32,
                    model_label: sample_tau32,
                },
                reference_key="part",
                xlabel=r"\tau_{32}^{\beta=1}",
                # bins_range=torch.tensor([0, 1.4], dtype=torch.float32),
                # yrange=(0, 2.1),
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
            gen_tau1 = tau(truth.x_gen, truth.x_gen_batch, N=1, R0=0.4, axis_mode=1)
            det_tau1 = tau(truth.x_det, truth.x_det_batch, N=1, R0=0.4, axis_mode=1)
            sample_tau1 = tau(
                samples.x_gen, samples.x_gen_batch, N=1, R0=0.4, axis_mode=1
            )

            gen_tau2 = tau(truth.x_gen, truth.x_gen_batch, N=2, R0=0.4, axis_mode=1)
            det_tau2 = tau(truth.x_det, truth.x_det_batch, N=2, R0=0.4, axis_mode=1)
            sample_tau2 = tau(
                samples.x_gen, samples.x_gen_batch, N=2, R0=0.4, axis_mode=1
            )

            gen_tau3 = tau(truth.x_gen, truth.x_gen_batch, N=3, R0=0.4, axis_mode=1)
            det_tau3 = tau(truth.x_det, truth.x_det_batch, N=3, R0=0.4, axis_mode=1)
            sample_tau3 = tau(
                samples.x_gen, samples.x_gen_batch, N=3, R0=0.4, axis_mode=1
            )

            gen_tau4 = tau(truth.x_gen, truth.x_gen_batch, N=4, R0=0.4, axis_mode=1)
            det_tau4 = tau(truth.x_det, truth.x_det_batch, N=4, R0=0.4, axis_mode=1)
            sample_tau4 = tau(
                samples.x_gen, samples.x_gen_batch, N=4, R0=0.4, axis_mode=1
            )

            gen_tau21 = torch.where(
                gen_tau1 > 0, gen_tau2 / gen_tau1, torch.tensor(0.0)
            )
            det_tau21 = torch.where(
                det_tau1 > 0, det_tau2 / det_tau1, torch.tensor(0.0)
            )
            sample_tau21 = torch.where(
                sample_tau1 > 0, sample_tau2 / sample_tau1, torch.tensor(0.0)
            )

            gen_tau32 = torch.where(
                gen_tau2 > 0, gen_tau3 / gen_tau2, torch.tensor(0.0)
            )
            det_tau32 = torch.where(
                det_tau2 > 0, det_tau3 / det_tau2, torch.tensor(0.0)
            )
            sample_tau32 = torch.where(
                sample_tau2 > 0, sample_tau3 / sample_tau2, torch.tensor(0.0)
            )

            gen_tau43 = torch.where(
                gen_tau3 > 0, gen_tau4 / gen_tau3, torch.tensor(0.0)
            )
            det_tau43 = torch.where(
                det_tau3 > 0, det_tau4 / det_tau3, torch.tensor(0.0)
            )
            sample_tau43 = torch.where(
                sample_tau3 > 0, sample_tau4 / sample_tau3, torch.tensor(0.0)
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
                    "part": gen_tau3,
                    "reco": det_tau3,
                    model_label: sample_tau3,
                },
                reference_key="part",
                xlabel=r"\tau_3^{\beta=1}",
                # bins_range=torch.tensor([0.03, 0.7], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                legend_loc=None,
                title={"title": r"$t\bar{t}$", "x": 0.93, "y": 0.95},
            )

            plot_ratio_histogram(
                data={
                    "part": gen_tau4,
                    "reco": det_tau4,
                    model_label: sample_tau4,
                },
                reference_key="part",
                xlabel=r"\tau_4^{\beta=1}",
                # bins_range=torch.tensor([0.03, 0.7], dtype=torch.float32),
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

            plot_ratio_histogram(
                data={
                    "part": gen_tau32,
                    "reco": det_tau32,
                    model_label: sample_tau32,
                },
                reference_key="part",
                xlabel=r"\tau_{32}^{\beta=1}",
                # bins_range=torch.tensor([0.07, 0.9], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                legend_loc=None,
                title={"title": r"$t\bar{t}$", "x": 0.93, "y": 0.95},
            )

            plot_ratio_histogram(
                data={
                    "part": gen_tau43,
                    "reco": det_tau43,
                    model_label: sample_tau43,
                },
                reference_key="part",
                xlabel=r"\tau_{43}^{\beta=1}",
                # bins_range=torch.tensor([0.07, 0.9], dtype=torch.float32),
                no_ratio_keys=["reco"],
                logy=False,
                file=file,
                legend_loc=None,
                title={"title": r"$t\bar{t}$", "x": 0.09, "y": 0.95},
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
                title={"title": r"$t\bar{t}$", "x": 0.09, "y": 0.95},
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
