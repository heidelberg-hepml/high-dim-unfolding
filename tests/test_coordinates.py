import pytest
import torch
from omegaconf import OmegaConf

from experiments import coordinates as c
from experiments.dataset import load_zplusjet, load_cms, load_ttbar


TOLERANCES = dict(atol=1e-3, rtol=1e-4)


def all_subclasses(cls):
    """Recursively find all subclasses of a given class."""
    subclasses = set()
    for subclass in cls.__subclasses__():
        subclasses.add(subclass)
        subclasses.update(all_subclasses(subclass))
    return subclasses


coordinates_classes = [
    # c.StandardLogPtPhiEtaLogM2,
    c.StandardJetScaledLogPtPhiEtaLogM2
]


@pytest.mark.parametrize("coordinates", coordinates_classes)
@pytest.mark.parametrize(
    "dataset, data_fn",
    [
        # ("zplusjet", load_zplusjet),
        # ("cms", load_cms),
        ("ttbar", load_ttbar),
    ],
)
@pytest.mark.parametrize("mass", [1.0])
def test_invertibility(coordinates, dataset, data_fn, mass):
    """test invertibility of forward() and inverse() methods"""
    cfg = OmegaConf.create(
        {"length": -1, "add_pid": False, "mass": mass, "min_mult": 1}
    )
    dtype = torch.float64

    data = data_fn("data/" + dataset, cfg, dtype)
    if dataset == "cms":
        fixed_dims = []
    else:
        fixed_dims = [3]

    coord = coordinates(pt_min=0.0, fixed_dims=fixed_dims)

    particles = data["gen_particles"]
    jets = c.fourmomenta_to_jetmomenta(particles.sum(dim=1))
    mask = torch.arange(particles.shape[1])[None, :] < data["gen_mults"][:, None]
    ptr = torch.cumsum(
        torch.cat([torch.zeros(1, dtype=torch.long), data["gen_mults"]]), dim=0
    )

    jets = torch.repeat_interleave(jets, ptr.diff(), dim=0)

    coord.init_fit(particles, mask=mask, ptr=ptr, jet=jets)

    fourmomenta_original = particles[mask]

    # fourmomenta_original = data["gen_particles"].sum(dim=1)
    # ptr = torch.arange(fourmomenta_original.shape[0] + 1, dtype=torch.long)
    # jets = c.fourmomenta_to_jetmomenta(fourmomenta_original)

    x_original = coord.fourmomenta_to_x(fourmomenta_original, ptr=ptr, jet=jets)
    fourmomenta_transformed = coord.x_to_fourmomenta(x_original, ptr=ptr, jet=jets)
    x_transformed = coord.fourmomenta_to_x(fourmomenta_transformed, ptr=ptr, jet=jets)

    torch.testing.assert_close(
        fourmomenta_original, fourmomenta_transformed, **TOLERANCES
    )
    torch.testing.assert_close(x_original, x_transformed, **TOLERANCES)


@pytest.mark.parametrize("coordinates", coordinates_classes)
@pytest.mark.parametrize(
    "dataset, data_fn",
    [
        # ("zplusjet", load_zplusjet),
        # ("cms", load_cms),
        ("ttbar", load_ttbar),
    ],
)
@pytest.mark.parametrize("mass", [1.0])
def test_velocity(coordinates, dataset, data_fn, mass):
    """test velocity_forward() and velocity_inverse() methods"""
    cfg = OmegaConf.create({"length": -1, "add_pid": False, "mass": mass})
    device = torch.device("cpu")
    dtype = torch.float64

    data = data_fn("data/" + dataset, cfg, dtype)
    if dataset == "cms":
        fixed_dims = []
    else:
        fixed_dims = [3]
    coord = coordinates(pt_min=0.0, fixed_dims=fixed_dims)

    particles = data["gen_particles"]
    jets = c.fourmomenta_to_jetmomenta(particles.sum(dim=1))
    mask = torch.arange(particles.shape[1])[None, :] < data["gen_mults"][:, None]
    ptr = torch.cumsum(
        torch.cat([torch.zeros(1, dtype=torch.long), data["gen_mults"]]), dim=0
    )

    jets = torch.repeat_interleave(jets, ptr.diff(), dim=0)

    coord.init_fit(particles, mask=mask, ptr=ptr, jet=jets)

    x = particles[mask]
    x.requires_grad_()
    v_x = torch.randn_like(x)
    v_y, y = coord.velocity_fourmomenta_to_x(v_x, x, jet=jets, ptr=ptr)
    v_z, z = coord.velocity_x_to_fourmomenta(v_y, y, jet=jets, ptr=ptr)

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
