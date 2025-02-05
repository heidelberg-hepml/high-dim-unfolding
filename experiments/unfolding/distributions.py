import torch
import math

from experiments.eventgen.helpers import (
    get_pt,
    delta_r_fast,
    fourmomenta_to_jetmomenta,
)
import experiments.unfolding.coordinates as c

from experiments.unfolding.helpers import EPS1


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
        onshell_mass = (
            self.onshell_mass.to(device, dtype=dtype).expand(fourmomenta.shape[:-1])
            / self.units
        )
        fourmomenta[..., 0] = onshell_mass**2 + torch.sum(
            fourmomenta[..., 1:] ** 2, dim=-1
        )
        return fourmomenta

    def propose(self, shape, device, dtype, generator=None):
        raise NotImplementedError


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
        log_prob = log_prob.sum(-1).unsqueeze(-1)
        logdetjac = self.coordinates.logdetjac_x_to_fourmomenta(pppm2)[0]
        log_prob = log_prob + logdetjac
        return log_prob


class StandardPPP(OnShellDistribution):
    def __init__(self, pt_min, mean, std, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coordinates = c.StandardPPPLogM2(mean=mean, std=std)

    def propose(self, shape, device, dtype, generator=None):
        eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        fourmomenta = self.coordinates.x_to_fourmomenta(eps)
        return fourmomenta

    def log_prob(self, fourmomenta):
        pppm2 = self.coordinates.fourmomenta_to_x(fourmomenta)
        log_prob = log_prob_normal(pppm2)
        log_prob[..., 3] = 0.0  # fixed mass does not contribute
        log_prob = log_prob.sum(-1).unsqueeze(-1)
        logdetjac = self.coordinates.logdetjac_x_to_fourmomenta(pppm2)[0]
        log_prob = log_prob + logdetjac
        return log_prob


class StandardLogPtPhiEta(OnShellDistribution):
    """Base distribution 4: phi uniform; eta, log(pt) and log(mass) from fitted normal"""

    def __init__(self, pt_min, mean, std, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coordinates = c.StandardLogPtPhiEtaLogM2(pt_min, self.units, mean, std)

    def propose(self, shape, device, dtype, generator=None):
        """Base distribution for precisesiast: pt, eta gaussian; phi uniform; mass shifted gaussian"""
        # sample (logpt, phi, eta, logmass)
        eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)

        # sample phi uniformly
        eps = self.coordinates.transforms[-1].inverse(eps)
        eps[..., 1] = math.pi * (
            2 * torch.rand(shape[:-1], device=device, dtype=dtype, generator=generator)
            - 1
        )

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
        log_prob = log_prob.sum(-1).unsqueeze(-1)
        logdetjac = self.coordinates.logdetjac_x_to_fourmomenta(logptphietalogm2)[0]
        log_prob = log_prob + logdetjac
        return log_prob


def log_prob_normal(z, mean=0.0, std=1.0):
    std_term = torch.log(std) if type(std) == torch.Tensor else math.log(std)
    return -((z - mean) ** 2) / (2 * std**2) - 0.5 * math.log(2 * math.pi) - std_term
