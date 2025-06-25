import pytest
import torch
from omegaconf import OmegaConf

from experiments import coordinates as c
from experiments.dataset import load_zplusjet, load_cms


TOLERANCES = dict(atol=1e-3, rtol=1e-4)
n_batches = 10


@pytest.mark.parametrize(
    "coordinates",
    [
        c.Fourmomenta,
        c.PPPM2,
        c.EPhiPtPz,
        c.PtPhiEtaE,
        c.PtPhiEtaM2,
        c.PPPLogM2,
        c.StandardPPPLogM2,
        c.LogPtPhiEtaE,
        c.PtPhiEtaLogM2,
        c.LogPtPhiEtaM2,
        c.LogPtPhiEtaLogM2,
        c.StandardLogPtPhiEtaLogM2,
        c.StandardPtPhiEtaLogM2,
        c.IndividualStandardLogPtPhiEtaLogM2,
        c.JetScaledPtPhiEtaM2,
        c.IndividualStandardJetScaledLogPtPhiEtaLogM2,
        c.StandardPPPM2,
        c.StandardJetScaledLogPtPhiEtaLogM2,
    ],
)
def test_invertibility(coordinates):
    """test invertibility of forward() and inverse() methods"""
    cfg = OmegaConf.create({"length": 100, "add_pid": False, "mass": 0.001})
    device = torch.device("cpu")
    dtype = torch.float64  # sometimes fails with float32

    datasets = {}
    datasets["zplusjet"] = load_zplusjet("data/zplusjet", cfg, dtype)
    datasets["cms"] = load_cms("data/cms", cfg, dtype)

    for key, data in datasets.items():
        if key == "cms":
            fixed_dims = []
        else:
            fixed_dims = [3]
        coord = coordinates(pt_min=0.0, fixed_dims=fixed_dims)

        particles = data["det_particles"]
        jets = c.fourmomenta_to_jetmomenta(particles.sum(dim=1))
        mask = torch.arange(particles.shape[1])[None, :] < data["det_mults"][:, None]
        ptr = torch.cumsum(
            torch.cat([torch.zeros(1, dtype=torch.long), data["det_mults"]]), dim=0
        )

        jets = torch.repeat_interleave(jets, ptr.diff(), dim=0)

        coord.init_fit(particles, mask=mask, ptr=ptr, jet=jets)

        fourmomenta_original = particles[mask]

        # forward and inverse transform
        x_original = coord.fourmomenta_to_x(fourmomenta_original, ptr=ptr, jet=jets)
        fourmomenta_transformed = coord.x_to_fourmomenta(x_original, ptr=ptr, jet=jets)
        x_transformed = coord.fourmomenta_to_x(
            fourmomenta_transformed, ptr=ptr, jet=jets
        )

        torch.testing.assert_close(
            fourmomenta_original, fourmomenta_transformed, **TOLERANCES
        )
        torch.testing.assert_close(x_original, x_transformed, **TOLERANCES)


@pytest.mark.parametrize(
    "coordinates",
    [
        c.Fourmomenta,
        c.PPPM2,
        c.EPhiPtPz,
        c.PtPhiEtaE,
        c.PtPhiEtaM2,
        c.PPPLogM2,
        c.StandardPPPLogM2,
        c.LogPtPhiEtaE,
        c.PtPhiEtaLogM2,
        c.LogPtPhiEtaM2,
        c.LogPtPhiEtaLogM2,
        c.StandardLogPtPhiEtaLogM2,
        c.StandardPtPhiEtaLogM2,
        c.IndividualStandardLogPtPhiEtaLogM2,
        c.JetScaledPtPhiEtaM2,
        c.IndividualStandardJetScaledLogPtPhiEtaLogM2,
        c.StandardPPPM2,
        c.StandardJetScaledLogPtPhiEtaLogM2,
    ],
)
def test_velocity(coordinates):
    """test velocity_forward() and velocity_inverse() methods"""
    cfg = OmegaConf.create({"length": 100, "add_pid": False, "mass": 0.001})
    device = torch.device("cpu")
    dtype = torch.float64  # sometimes fails with float32

    datasets = {}
    datasets["zplusjet"] = load_zplusjet("data/zplusjet", cfg, dtype)
    datasets["cms"] = load_cms("data/cms", cfg, dtype)

    for key, data in datasets.items():
        if key == "cms":
            fixed_dims = []
        else:
            fixed_dims = [3]
        coord = coordinates(pt_min=0.0, fixed_dims=fixed_dims)

        particles = data["det_particles"]
        jets = c.fourmomenta_to_jetmomenta(particles.sum(dim=1))
        mask = torch.arange(particles.shape[1])[None, :] < data["det_mults"][:, None]
        ptr = torch.cumsum(
            torch.cat([torch.zeros(1, dtype=torch.long), data["det_mults"]]), dim=0
        )

        jets = torch.repeat_interleave(jets, ptr.diff(), dim=0)

        coord.init_fit(particles, mask=mask, ptr=ptr, jet=jets)

        x = particles[mask]
        x.requires_grad_()
        v_x = torch.randn_like(x)
        v_y, y = coord.velocity_fourmomenta_to_x(v_x, x)
        v_z, z = coord.velocity_x_to_fourmomenta(v_y, y)

        # jacobians from autograd
        jac_fw_autograd, jac_inv_autograd = [], []
        for i in range(4):
            grad_outputs = torch.zeros_like(x)
            grad_outputs[..., i] = 1.0
            fw_autograd = torch.autograd.grad(
                y,
                x,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            inv_autograd = torch.autograd.grad(
                z,
                y,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            jac_fw_autograd.append(fw_autograd)
            jac_inv_autograd.append(inv_autograd)
        jac_fw_autograd = torch.stack(jac_fw_autograd, dim=-2)
        jac_inv_autograd = torch.stack(jac_inv_autograd, dim=-2)

        v_y_autograd = torch.einsum("...ij,...j->...i", jac_fw_autograd, v_x)
        v_z_autograd = torch.einsum("...ij,...j->...i", jac_inv_autograd, v_y)

        # compare to autograd
        torch.testing.assert_close(v_y, v_y_autograd, **TOLERANCES)
        torch.testing.assert_close(v_z, v_z_autograd, **TOLERANCES)
