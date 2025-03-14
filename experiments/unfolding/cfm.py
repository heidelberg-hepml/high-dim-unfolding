import torch
from torch import nn
from torch.autograd import grad

from torchdiffeq import odeint
from experiments.unfolding.distributions import (
    BaseDistribution,
    NaivePPP,
    StandardPPP,
    StandardLogPtPhiEta,
)
from experiments.unfolding.utils import TimeEmbedding, get_pt
import experiments.unfolding.coordinates as c
from experiments.unfolding.geometry import BaseGeometry, SimplePossiblyPeriodicGeometry
from experiments.logger import LOGGER


def hutchinson_trace(x_out, x_in):
    # Hutchinson's trace Jacobian estimator, needs O(1) calls to autograd
    noise = torch.randint_like(x_in, low=0, high=2).float() * 2 - 1.0
    x_out_noise = torch.sum(x_out * noise)
    gradient = grad(x_out_noise, x_in)[0].detach()
    return torch.sum(gradient * noise, dim=-1)


def autograd_trace(x_out, x_in):
    # Standard way of calculating trace of the Jacobian, needs O(n) calls to autograd
    trJ = 0.0
    for i in range(x_out.shape[-1]):
        trJ += (
            grad(x_out[..., i].sum(), x_in, retain_graph=True)[0]
            .contiguous()[..., i]
            .contiguous()
            .detach()
        )
    return trJ.contiguous()


class CFM(nn.Module):
    """
    Base class for all CFM models
    - event-generation-specific features are implemented in EventCFM
    - get_velocity is implemented by architecture-specific subclasses
    """

    def __init__(
        self,
        cfm,
        odeint={"method": "dopri5", "atol": 1e-5, "rtol": 1e-5, "options": None},
    ):
        super().__init__()
        self.t_embedding = TimeEmbedding(
            embed_dim=cfm.embed_t_dim, scale=cfm.embed_t_scale
        )
        self.trace_fn = hutchinson_trace if cfm.hutchinson else autograd_trace
        self.odeint = odeint
        self.cfm = cfm

        # initialize to base objects, this will be overwritten later
        self.distribution = BaseDistribution()
        self.coordinates = c.BaseCoordinates()
        self.condition_coordinates = c.BaseCoordinates()
        self.geometry = BaseGeometry()

        if cfm.transforms_float64:
            c.DTYPE = torch.float64
        else:
            c.DTYPE = torch.float32

    def init_distribution(self):
        raise NotImplementedError

    def init_coordinates(self):
        raise NotImplementedError

    def init_geometry(self):
        raise NotImplementedError

    def sample_base(self, shape, device, dtype, generator=None):
        fourmomenta = self.distribution.sample(
            shape, device, dtype, generator=generator
        )
        return fourmomenta

    def get_velocity(self, x, t):
        """
        Parameters
        ----------
        x : torch.tensor with shape (batchsize, 4)
        t : torch.tensor with shape (batchsize, 1)
        """
        # implemented by architecture-specific subclasses
        raise NotImplementedError

    def handle_velocity(self, v):
        # default: do nothing
        return v

    def batch_loss(self, batch):
        """
        Construct the conditional flow matching objective

        Parameters
        ----------
        batch : tuple of Batch graphs
            Target space particles in fourmomenta space

        Returns
        -------
        loss : torch.tensor with shape (1)
        """
        x0_fourmomenta = batch.x_gen
        t = torch.rand(
            batch.num_graphs,
            1,
            dtype=x0_fourmomenta.dtype,
            device=x0_fourmomenta.device,
        )
        t = torch.repeat_interleave(t, batch.x_gen_batch.bincount(), dim=0)
        x1_fourmomenta = self.sample_base(
            x0_fourmomenta.shape, x0_fourmomenta.device, x0_fourmomenta.dtype
        )

        # construct target trajectories
        x0_straight = self.coordinates.fourmomenta_to_x(x0_fourmomenta)
        x1_straight = self.coordinates.fourmomenta_to_x(x1_fourmomenta)

        xt_straight, vt_straight = self.geometry.get_trajectory(
            x0_straight, x1_straight, t
        )
        vp_straight = self.get_velocity(xt_straight, t, batch)

        # evaluate conditional flow matching objective
        distance = self.geometry.get_metric(
            vp_straight, vt_straight, xt_straight
        ).mean()
        distance_particlewise = [
            ((vp_straight - vt_straight) ** 2)[..., i].mean() / 2 for i in range(4)
        ]
        return distance, distance_particlewise

    def sample(self, batch, device, dtype):
        """
        Sample from CFM model
        Solve an ODE using a NN-parametrized velocity field

        Parameters
        ----------
        batch : tuple of Batch graphs
        device : torch.device
        dtype : torch.dtype

        Returns
        -------
        x0_fourmomenta : torch.tensor with shape shape = (batchsize, 4)
            Generated events
        """

        sample_batch = batch.clone()

        def velocity(t, xt_straight):
            xt_straight = self.geometry._handle_periodic(xt_straight)
            t = t * torch.ones(
                shape[0], 1, dtype=xt_straight.dtype, device=xt_straight.device
            )
            vt_straight = self.get_velocity(xt_straight, t, batch)
            vt_straight = self.handle_velocity(vt_straight)
            return vt_straight

        # sample fourmomenta from base distribution
        shape = batch.x_gen.shape
        x1_fourmomenta = self.sample_base(shape, device, dtype)
        x1_straight = self.coordinates.fourmomenta_to_x(x1_fourmomenta)

        # solve ODE in straight space
        x0_straight = odeint(
            velocity,
            x1_straight,
            torch.tensor([1.0, 0.0], device=x1_straight.device),
            **self.odeint,
        )[-1]
        x0_straight = self.geometry._handle_periodic(x0_straight)

        # the infamous nan remover
        # (MLP sometimes returns nan for single events,
        # and all components of the event are nan...
        # just sample another event in this case)
        mask = torch.isfinite(x0_straight).all(dim=-1)
        if (~mask).any():
            mask2 = torch.isfinite(x0_straight)
            x0_straight = x0_straight[mask, ...]
            x1_fourmomenta = x1_fourmomenta[mask, ...]
            LOGGER.warning(
                f"Found {(~mask2).sum(dim=0).numpy()} nan events while sampling"
            )

        # transform generated event back to fourmomenta
        x0_fourmomenta = self.coordinates.x_to_fourmomenta(x0_straight)

        pt = get_pt(x0_fourmomenta).unsqueeze(-1)

        x_perm = torch.argsort(pt, dim=0, descending=True)
        x0_fourmomenta = x0_fourmomenta.take_along_dim(x_perm, dim=0)
        index = batch.x_gen_batch.unsqueeze(-1).take_along_dim(x_perm, dim=0)
        index_perm = torch.argsort(index, dim=0, stable=True)
        x0_fourmomenta = x0_fourmomenta.take_along_dim(index_perm, dim=0)

        sample_batch.x_gen = x0_fourmomenta

        return sample_batch

    def log_prob(self, batch):
        """
        Evaluate log_prob for existing target samples in a CFM model
        Solve ODE involving the trace of the velocity field, this is more expensive than normal sampling
        The 'self.hutchinson' parameter controls if the trace should be evaluated
        with the hutchinson trace estimator that needs O(1) calls to the network,
        as opposed to the exact autograd trace that needs O(n_particles) calls to the network
        Note: Could also have a sample_and_log_prob method, but we have no use case for this

        Parameters
        ----------

        batch : tuple of Batch graphs

        Returns
        -------
        log_prob_fourmomenta : torch.tensor with shape (batchsize)
            log_prob of each event in x0, evaluated in fourmomenta space
        """

        x0_fourmomenta = batch.x_gen

        def net_wrapper(t, state):
            with torch.set_grad_enabled(True):
                xt_straight = state[0]
                xt_straight = self.geometry._handle_periodic(xt_straight)
                xt_straight = xt_straight.detach().requires_grad_(True)
                t = t * torch.ones(
                    xt_straight.shape[0],
                    1,
                    1,
                    dtype=xt_straight.dtype,
                    device=xt_straight.device,
                )
                vt_straight = self.get_velocity(xt_straight, t, batch)
                vt_straight = self.handle_velocity(vt_straight)
                dlogp_dt_straight = -self.trace_fn(vt_straight, xt_straight).unsqueeze(
                    -1
                )
            return vt_straight.detach(), dlogp_dt_straight.detach()

        # solve ODE in coordinates
        x0_straight = self.coordinates.fourmomenta_to_x(x0_fourmomenta)
        logdetjac0_cfm_straight = torch.zeros(
            (x0_straight.shape[0], 1),
            dtype=x0_straight.dtype,
            device=x0_straight.device,
        )
        state0 = (x0_straight, logdetjac0_cfm_straight)
        xt_straight, logdetjact_cfm_straight = odeint(
            net_wrapper,
            state0,
            torch.tensor(
                [0.0, 1.0], dtype=x0_straight.dtype, device=x0_straight.device
            ),
            **self.odeint,
        )
        logdetjac_cfm_straight = logdetjact_cfm_straight[-1].detach()
        x1_straight = xt_straight[-1].detach()

        # the infamous nan remover
        # (MLP sometimes returns nan for single events,
        # just remove these events from the log_prob computation)
        mask = torch.isfinite(x1_straight).all(dim=-1)
        if (~mask).any():
            mask2 = torch.isfinite(x1_straight)
            logdetjac_cfm_straight = logdetjac_cfm_straight[mask]
            x1_straight = x1_straight[mask]
            x0_fourmomenta = x0_fourmomenta[mask]
            LOGGER.warning(
                f"Found {(~mask2).sum(dim=0).numpy()} nan events while sampling"
            )

        x1_fourmomenta = self.coordinates.x_to_fourmomenta(x1_straight)
        logdetjac_forward = self.coordinates.logdetjac_fourmomenta_to_x(x0_fourmomenta)[
            0
        ]
        logdetjac_inverse = -self.coordinates.logdetjac_fourmomenta_to_x(
            x1_fourmomenta
        )[0]

        # collect log_probs
        log_prob_base_fourmomenta = self.distribution.log_prob(x1_fourmomenta)
        log_prob_fourmomenta = (
            log_prob_base_fourmomenta
            - logdetjac_cfm_straight
            - logdetjac_forward
            - logdetjac_inverse
        )
        return log_prob_fourmomenta


class EventCFM(CFM):
    """
    Add event-generation-specific methods to CFM classes:
    - Save information at the wrapper level
    - Handle base distribution and coordinates for RFM
    - Wrapper-specific preprocessing and undo_preprocessing
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_physics(self, units, pt_min, base_type, onshell_mass, device):
        """
        Pass physics information to the CFM class

        Parameters
        ----------
        units: float
            Scale of dimensionful quantities
            I call it 'units' because we can really choose it arbitrarily without losing anything
            Hard-coded in EventGenerationExperiment
        pt_min: List[float]
            Minimum pt value for each particle
            Hard-coded in EventGenerationExperiment
        mean: Torch.Tensor
        std: Torch.Tensor
        base_type: int
            Which base distribution to use
        """
        self.units = units
        self.pt_min = pt_min
        self.base_type = base_type
        self.onshell_mass = onshell_mass

    def init_distribution(self):
        args = [
            self.onshell_mass,
            self.pt_min,
            self.units,
        ]
        if self.base_type == 1:
            self.distribution = NaivePPP(*args)
        elif self.base_type == 2:
            self.distribution = StandardPPP(*args)
        elif self.base_type == 3:
            self.distribution = StandardLogPtPhiEta(*args)
        else:
            raise ValueError(f"base_type={self.base_type} not implemented")

    def init_coordinates(self):
        self.coordinates = self._init_coordinates(self.cfm.coordinates)
        self.condition_coordinates = self._init_coordinates(
            self.cfm.condition_coordinates
        )

    def _init_coordinates(self, coordinates_label):
        if coordinates_label == "Fourmomenta":
            coordinates = c.Fourmomenta()
        elif coordinates_label == "PPPM2":
            coordinates = c.PPPM2()
        elif coordinates_label == "PPPLogM2":
            coordinates = c.PPPLogM2()
        elif coordinates_label == "StandardPPPLogM2":
            coordinates = c.StandardPPPLogM2()
        elif coordinates_label == "EPhiPtPz":
            coordinates = c.EPhiPtPz()
        elif coordinates_label == "PtPhiEtaE":
            coordinates = c.PtPhiEtaE()
        elif coordinates_label == "PtPhiEtaM2":
            coordinates = c.PtPhiEtaM2()
        elif coordinates_label == "LogPtPhiEtaE":
            coordinates = c.LogPtPhiEtaE(self.pt_min, self.units)
        elif coordinates_label == "LogPtPhiEtaM2":
            coordinates = c.LogPtPhiEtaM2(self.pt_min, self.units)
        elif coordinates_label == "PtPhiEtaLogM2":
            coordinates = c.PtPhiEtaLogM2()
        elif coordinates_label == "LogPtPhiEtaLogM2":
            coordinates = c.LogPtPhiEtaLogM2(self.pt_min, self.units)
        elif coordinates_label == "StandardLogPtPhiEtaLogM2":
            coordinates = c.StandardLogPtPhiEtaLogM2(
                self.pt_min,
                self.units,
            )
        else:
            raise ValueError(f"coordinates={coordinates_label} not implemented")
        return coordinates

    def init_geometry(self):
        # placeholder for any initialization that needs to be done
        if self.cfm.geometry.type == "simple":
            self.geometry = SimplePossiblyPeriodicGeometry(
                contains_phi=self.coordinates.contains_phi,
                periodic=self.cfm.geometry.periodic,
            )
        else:
            raise ValueError(f"geometry={self.cfm.geometry} not implemented")

    def preprocess(self, fourmomenta):
        fourmomenta = fourmomenta / self.units
        return fourmomenta

    def undo_preprocess(self, fourmomenta):
        fourmomenta = fourmomenta * self.units
        return fourmomenta

    def sample(self, *args, **kwargs):
        fourmomenta = super().sample(*args, **kwargs)
        return fourmomenta

    def handle_velocity(self, v):
        if self.coordinates.contains_mass:
            # manually set mass velocity to zero
            v[..., 3] = 0.0
        return v
