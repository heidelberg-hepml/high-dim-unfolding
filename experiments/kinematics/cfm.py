import torch
from torch import nn
from torchdiffeq import odeint

from experiments.utils import GaussianFourierProjection
import experiments.coordinates as c
from experiments.geometry import BaseGeometry, SimplePossiblyPeriodicGeometry
from experiments.baselines import custom_rk4
from experiments.embedding import add_jet_to_sequence


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
        self.t_embedding = nn.Sequential(
            GaussianFourierProjection(
                embed_dim=cfm.embed_t_dim, scale=cfm.embed_t_scale
            ),
            nn.Linear(cfm.embed_t_dim, cfm.embed_t_dim),
        )
        self.odeint = odeint
        self.cfm = cfm

        # initialize to base objects, this will be overwritten later
        self.coordinates = c.BaseCoordinates()
        self.condition_coordinates = c.BaseCoordinates()
        self.geometry = BaseGeometry()

        if cfm.transforms_float64:
            c.DTYPE = torch.float64
        else:
            c.DTYPE = torch.float32

    def init_distribution(self):
        pass

    def init_coordinates(self):
        raise NotImplementedError

    def init_geometry(self):
        raise NotImplementedError

    def sample_base(self, x0, mask, generator=None):
        sample = torch.randn(
            x0.shape, device=x0.device, dtype=x0.dtype, generator=generator
        )
        if self.coordinates.contains_phi:
            sample[..., 1] = (
                torch.rand(
                    x0.shape[:-1], device=x0.device, dtype=x0.dtype, generator=generator
                )
                * 2
                * torch.pi
                - torch.pi
            )
        if 3 in self.cfm.masked_dims:
            sample[..., 3] = torch.mean(x0[mask][..., 3])
        return sample

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
        if self.cfm.add_jet:
            new_batch, mask = add_jet_to_sequence(batch)
        else:
            new_batch = batch
            mask = torch.arange(new_batch.x_gen.shape[0], device=new_batch.x_gen.device)

        x0 = new_batch.x_gen
        t = torch.rand(
            new_batch.num_graphs,
            1,
            dtype=x0.dtype,
            device=x0.device,
        )
        t = torch.repeat_interleave(t, new_batch.x_gen_batch.bincount(), dim=0)

        x1 = self.sample_base(x0, mask)

        if self.cfm.add_jet:
            x1[~mask] = new_batch.jet_gen

        vt = x1 - x0
        xt = self.geometry._handle_periodic(x0 + vt * t)

        attention_mask, condition_attention_mask, crossattention_mask = self.get_masks(
            new_batch
        )

        condition = self.get_condition(new_batch, condition_attention_mask)

        if self.cfm.self_condition_prob > 0.0:
            self_condition = torch.zeros_like(vt, device=vt.device, dtype=vt.dtype)
            if torch.rand(1) < self.cfm.self_condition_prob:
                self_condition = self.get_velocity(
                    xt=xt,
                    t=t,
                    batch=new_batch,
                    condition=condition,
                    attention_mask=attention_mask,
                    crossattention_mask=crossattention_mask,
                    self_condition=self_condition,
                ).detach()

            vp = self.get_velocity(
                xt=xt,
                t=t,
                batch=new_batch,
                condition=condition,
                attention_mask=attention_mask,
                crossattention_mask=crossattention_mask,
                self_condition=self_condition,
            )
        else:
            vp = self.get_velocity(
                xt=xt,
                t=t,
                batch=new_batch,
                condition=condition,
                attention_mask=attention_mask,
                crossattention_mask=crossattention_mask,
            )

        vp = self.handle_velocity(vp[mask])
        vt = self.handle_velocity(vt[mask])

        # evaluate conditional flow matching objective
        distance = ((vp - vt) ** 2).mean()
        if self.cfm.cosine_similarity_factor > 0.0:
            cosine_similarity = (
                1 - (vp * vt).sum(dim=-1) / (vp.norm(dim=-1) * vt.norm(dim=-1))
            ).mean()
            loss = (
                1 - self.cfm.cosine_similarity_factor
            ) * distance + self.cfm.cosine_similarity_factor * cosine_similarity
        else:
            loss = distance
        distance_particlewise = ((vp - vt) ** 2).mean(dim=0)
        return loss, distance_particlewise

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
        x0 : torch.tensor with shape shape = (batchsize, 4)
            Generated events
        """

        if self.cfm.add_jet:
            new_batch, jet_mask = add_jet_to_sequence(batch)
        else:
            new_batch = batch
            jet_mask = torch.ones(
                new_batch.x_gen.shape[0],
                device=new_batch.x_gen.device,
                dtype=torch.bool,
            )

        sample_batch = batch.clone()

        attention_mask, condition_attention_mask, crossattention_mask = self.get_masks(
            new_batch
        )

        condition = self.get_condition(new_batch, condition_attention_mask)

        def velocity(t, xt, self_condition=None):
            xt = self.geometry._handle_periodic(xt)
            t = t * torch.ones(shape[0], 1, dtype=xt.dtype, device=xt.device)
            vt = self.get_velocity(
                xt=xt,
                t=t,
                batch=new_batch,
                condition=condition,
                attention_mask=attention_mask,
                crossattention_mask=crossattention_mask,
                self_condition=self_condition,
            )
            vt = self.handle_velocity(vt)  # manually set mass velocity to zero
            return vt

        # sample fourmomenta from base distribution
        shape = new_batch.x_gen.shape
        x1 = self.sample_base(new_batch.x_gen, jet_mask)

        if self.cfm.self_condition_prob > 0.0:
            v1 = torch.zeros_like(x1, device=x1.device, dtype=x1.dtype)
            x0 = custom_rk4(
                velocity,
                (x1, v1),
                torch.tensor([1.0, 0.0], device=x1.device),
                step_size=self.odeint.options["step_size"],
            )[-1]

        else:
            x0 = odeint(
                velocity,
                x1,
                torch.tensor([1.0, 0.0], device=x1.device),
                **self.odeint,
            )[-1]

        sample_batch.x_gen = self.geometry._handle_periodic(x0[jet_mask])

        # sort generated events by pT
        # pt = x0[..., 0].unsqueeze(-1)
        # x_perm = torch.argsort(pt, dim=0, descending=True)
        # x0 = x0.take_along_dim(x_perm, dim=0)
        # index = batch.x_gen_batch.unsqueeze(-1).take_along_dim(x_perm, dim=0)
        # index_perm = torch.argsort(index, dim=0, stable=True)
        # x0 = x0.take_along_dim(index_perm, dim=0)

        return sample_batch, x1


class EventCFM(CFM):
    """
    Add event-generation-specific methods to CFM classes:
    - Save information at the wrapper level
    - Handle base distribution and coordinates for RFM
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_physics(self, pt_min, mass):
        """
        Pass physics information to the CFM class

        Parameters
        ----------
        pt_min: float
            Minimum pt value for each particle
        mass: float
        """
        self.pt_min = pt_min
        self.mass = mass

    def init_coordinates(self):
        self.coordinates = self._init_coordinates(self.cfm.coordinates)
        self.condition_coordinates = self._init_coordinates(
            self.cfm.condition_coordinates
        )
        if self.cfm.transforms_float64:
            self.coordinates.to(torch.float64)
            self.condition_coordinates.to(torch.float64)

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
            coordinates = c.LogPtPhiEtaE(self.pt_min)
        elif coordinates_label == "LogPtPhiEtaM2":
            coordinates = c.LogPtPhiEtaM2(self.pt_min)
        elif coordinates_label == "PtPhiEtaLogM2":
            coordinates = c.PtPhiEtaLogM2()
        elif coordinates_label == "LogPtPhiEtaLogM2":
            coordinates = c.LogPtPhiEtaLogM2(self.pt_min)
        elif coordinates_label == "StandardLogPtPhiEtaLogM2":
            coordinates = c.StandardLogPtPhiEtaLogM2(
                self.pt_min,
                fixed_dims=self.cfm.masked_dims,
            )
        elif coordinates_label == "JetScaledPtPhiEtaM2":
            coordinates = c.JetScaledPtPhiEtaM2()
        elif coordinates_label == "StandardJetScaledLogPtPhiEtaLogM2":
            coordinates = c.StandardJetScaledLogPtPhiEtaLogM2(
                self.pt_min, self.cfm.masked_dims
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

    def handle_velocity(self, v):
        v[..., self.cfm.masked_dims] = 0.0
        return v
