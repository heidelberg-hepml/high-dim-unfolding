import torch
import math

from experiments.eventgen.helpers import (
    get_pt,
    delta_r_fast,
    fourmomenta_to_jetmomenta,
)
import experiments.eventgen.coordinates as c

# sample a few extra events to speed up rejection sampling
SAMPLING_FACTOR = 10  # typically acceptance_rate > 0.5

from experiments.eventgen.helpers import EPS1


class BaseDistribution:
    """
    Abstract base distribution
    All child classes work in fourmomenta space,
    i.e. they generate fourmomenta and return log_prob in fourmomenta space
    """

    def sample(self, shape, device, dtype, generator=None, **kwargs):
        raise NotImplementedError

    def log_prob(self, x, **kwargs):
        raise NotImplementedError


class OnShellDistribution(BaseDistribution):
    """
    Implement on shell sampling
    """

    def __init__(
        self,
        onshell_mass,
        units,
    ):
        self.onshell_mass = torch.tensor(onshell_mass)
        self.units = units

    def sample(self, shape, device, dtype, generator=None):
        fourmomenta = self.propose(
            shape, device=device, dtype=dtype, generator=generator
        )
        self.onshell_mass = (
            self.onshell_mass.to(device, dtype=dtype).expand(fourmomenta.shape[:-1])
            / self.units
        )
        fourmomenta[..., 0] = self.onshell_mass**2 + torch.sum(
            fourmomenta[..., 1:] ** 2, dim=-1
        )
        return fourmomenta


class NaivePPP(OnShellDistribution):
    """Base distribution 1: 3-momentum from standard normal"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coordinates = c.PPPM2()

    def propose(self, shape, device, dtype, generator=None):
        eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        fourmomenta = self.coordinates.x_to_fourmomenta(eps)
        return fourmomenta

    def log_prob(self, fourmomenta):
        pppm2 = self.coordinates.fourmomenta_to_x(fourmomenta)
        log_prob = log_prob_normal(pppm2)
        log_prob[..., 3] = 0.0  # fixed mass does not contribute
        log_prob = log_prob.sum(dim=[-1, -2]).unsqueeze(-1)
        logdetjac = self.coordinates.logdetjac_x_to_fourmomenta(pppm2)[0]
        log_prob = log_prob + logdetjac
        return log_prob


class StandardPPP(OnShellDistribution):
    """Base distribution 1: 3-momentum from standard normal"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coordinates = c.StandardPPPM2()

    def propose(self, shape, device, dtype, generator=None):
        eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        fourmomenta = self.coordinates.x_to_fourmomenta(eps)
        return fourmomenta

    def log_prob(self, fourmomenta):
        pppm2 = self.coordinates.fourmomenta_to_x(fourmomenta)
        log_prob = log_prob_normal(pppm2)
        log_prob[..., 3] = 0.0  # fixed mass does not contribute
        log_prob = log_prob.sum(dim=[-1, -2]).unsqueeze(-1)
        logdetjac = self.coordinates.logdetjac_x_to_fourmomenta(pppm2)[0]
        log_prob = log_prob + logdetjac
        return log_prob


class StandardLogPtPhiEta(RejectionDistribution):
    """Base distribution 4: phi uniform; eta, log(pt) and log(mass) from fitted normal"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.use_pt_min
        ), f"use_pt_min=False not implemented for distribution StandardLogPtPhiEtaLogM2"
        self.coordinates = c.StandardLogPtPhiEtaLogM2(self.pt_min, self.units)

    def propose(self, shape, device, dtype, generator=None):
        """Base distribution for precisesiast: pt, eta gaussian; phi uniform; mass shifted gaussian"""
        # sample (logpt, phi, eta, logmass)
        shape = list(shape)
        shape[0] = int(shape[0] * SAMPLING_FACTOR)
        eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)

        eps = self.coordinates.transforms[-1].inverse(eps)
        eps[..., 1] = math.pi * (
            2 * torch.rand(shape[:-1], device=device, dtype=dtype, generator=generator)
            - 1
        )  # sample phi uniformly

        for t in self.coordinates.transforms[:-1][::-1]:
            eps = t.inverse(eps)
        return eps

    def log_prob(self, fourmomenta):
        logptphietalogm2 = self.coordinates.fourmomenta_to_x(fourmomenta)
        log_prob = log_prob_normal(logptphietalogm2)
        log_prob[..., 1] = -math.log(
            2 * math.pi
        )  # normalization factor for uniform phi distribution: 1/(2 pi)
        log_prob[..., 3] = 0.0
        log_prob = log_prob.sum(dim=[-1, -2]).unsqueeze(-1)
        logdetjac = self.coordinates.logdetjac_x_to_fourmomenta(logptphietalogm2)[0]
        log_prob = log_prob + logdetjac
        return log_prob


def get_pt_mask(fourmomenta, pt_min):
    pt = get_pt(fourmomenta)
    pt_min = pt_min[: fourmomenta.shape[1]].to(
        fourmomenta.device, dtype=fourmomenta.dtype
    )
    mask = (pt > pt_min).all(dim=-1)
    return mask


def get_delta_r_mask(fourmomenta, delta_r_min):
    jetmomenta = fourmomenta_to_jetmomenta(fourmomenta)
    dr = delta_r_fast(jetmomenta.unsqueeze(1), jetmomenta.unsqueeze(2))

    # diagonal should not be < delta_r_min
    arange = torch.arange(jetmomenta.shape[1], device=jetmomenta.device)
    dr[..., arange, arange] = 42

    mask = (dr > delta_r_min).all(dim=[-1, -2])
    return mask


def log_prob_normal(z, mean=0.0, std=1.0):
    std_term = torch.log(std) if type(std) == torch.Tensor else math.log(std)
    return -((z - mean) ** 2) / (2 * std**2) - 0.5 * math.log(2 * math.pi) - std_term
