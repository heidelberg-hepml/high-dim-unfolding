import torch
from torch import nn
from torchdiffeq import odeint
from torch_geometric.utils import scatter
from torch_geometric.data import Batch, Data
import os

from experiments.utils import GaussianFourierProjection, get_batch_from_ptr
import experiments.coordinates as c
from experiments.geometry import BaseGeometry, SimplePossiblyPeriodicGeometry
from experiments.dataset import positional_encoding
from experiments.baselines import custom_rk4
from experiments.embedding import (
    add_jet_to_sequence,
    add_start_token_to_x_gen,
    add_stop_token_to_x_gen,
    stop_threshold_fn,
)
from experiments.kinematics.plots import plot_kinematics
from experiments.logger import LOGGER


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
        if cfm.mult_encoding_dim > 0:
            self.mult_encoding = nn.Sequential(
                GaussianFourierProjection(embed_dim=cfm.mult_encoding_dim, scale=30.0),
                nn.Linear(cfm.mult_encoding_dim, cfm.mult_encoding_dim),
            )
        else:
            self.mult_encoding = None

        self.odeint = odeint
        self.cfm = cfm
        self.scaling = torch.tensor(self.cfm.const_coordinates_options.scaling)

        # initialize to base objects, this will be overwritten later
        self.const_coordinates = c.BaseCoordinates()
        self.condition_const_coordinates = c.BaseCoordinates()
        self.jet_coordinates = c.BaseCoordinates()
        self.condition_jet_coordinates = c.BaseCoordinates()
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

    def sample_base(self, x0, constituents_mask=None, generator=None):
        if constituents_mask is None:
            constituents_mask = torch.ones(x0.size(0), dtype=torch.bool)
        sample = torch.randn(
            x0.shape, device=x0.device, dtype=x0.dtype, generator=generator
        )
        if self.const_coordinates.contains_phi:
            if getattr(self.cfm.const_coordinates_options, "vonmises", False):
                sample[..., 1] = (
                    self.const_coordinates.phi_dist.sample(x0.shape[:-1])
                    / self.const_coordinates.phi_std
                )
            elif "JetScaled" not in self.cfm.const_coordinates:
                sample[..., 1] = (
                    torch.rand(
                        x0.shape[:-1],
                        device=x0.device,
                        dtype=x0.dtype,
                        generator=generator,
                    )
                    * 2
                    * torch.pi
                    - torch.pi
                )
        sample = sample * self.scaling.to(x0.device, dtype=x0.dtype)
        sample[..., self.cfm.masked_dims] = x0[..., self.cfm.masked_dims]
        sample[~constituents_mask] = x0[~constituents_mask]  # keep jets fixed
        return sample

    def get_masks(self, batch):
        raise NotImplementedError

    def get_condition(self, batch, condition_attention_mask):
        raise NotImplementedError

    def get_velocity(
        self,
        xt,
        t,
        batch,
        condition,
        attention_mask,
        crossattention_mask,
        self_condition=None,
    ):
        raise NotImplementedError

    def handle_velocity(self, v):
        # default: do nothing
        return v

    def batch_loss(self, batch):
        """
        Construct the conditional flow matching objective

        Parameters
        ----------
        batch

        Returns
        -------
        loss : torch.tensor with shape (1)
        """
        if self.cfm.add_jet:
            new_batch, constituents_mask, det_constituents_mask = add_jet_to_sequence(
                batch
            )
        else:
            new_batch = batch
            constituents_mask = torch.ones(
                new_batch.x_gen.shape[0],
                device=new_batch.x_gen.device,
                dtype=torch.bool,
            )
            det_constituents_mask = torch.ones(
                new_batch.x_det.shape[0],
                device=new_batch.x_det.device,
                dtype=torch.bool,
            )

        x0 = new_batch.x_gen
        t = torch.rand(
            new_batch.num_graphs,
            1,
            dtype=x0.dtype,
            device=x0.device,
        )
        t = torch.repeat_interleave(t, new_batch.x_gen_ptr.diff(), dim=0)

        x1 = self.sample_base(x0, constituents_mask)

        xt, vt = self.geometry.get_trajectory(
            x_target=x0,
            x_base=x1,
            t=t,
        )

        attention_mask, condition_attention_mask, crossattention_mask = self.get_masks(
            new_batch
        )

        condition = self.get_condition(new_batch, condition_attention_mask)

        vp = self.get_velocity(
            xt=xt,
            t=t,
            batch=new_batch,
            condition=condition,
            attention_mask=attention_mask,
            crossattention_mask=crossattention_mask,
        )
        vp = self.handle_velocity(vp[constituents_mask])
        vt = self.handle_velocity(vt[constituents_mask])

        # evaluate conditional flow matching objective
        distance = self.geometry.get_metric(vp, vt, xt[constituents_mask])
        loss = distance

        loss = scatter(
            loss, new_batch.x_gen_batch[constituents_mask], dim=0, reduce="mean"
        )

        loss = loss.mean()

        distance_particlewise = ((vp - vt) ** 2).mean(dim=0) / 2
        return loss, distance_particlewise

    def sample(self, batch, device, dtype):
        """
        Sample from CFM model
        Solve an ODE using a NN-parametrized velocity field

        Parameters
        ----------
        batch : Batch with gen and det graphs
        device : torch.device
        dtype : torch.dtype

        Returns
        -------
        x0 : torch.tensor with shape (batchsize, 4)
            Generated events
        """

        if self.cfm.add_jet:
            new_batch, constituents_mask, det_constituents_mask = add_jet_to_sequence(
                batch
            )
        else:
            new_batch = batch.clone()
            constituents_mask = torch.ones(
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
            t = t.view(1, 1).expand(xt.shape[0], -1)

            vt = self.get_velocity(
                xt=xt,
                t=t,
                batch=new_batch,
                condition=condition,
                attention_mask=attention_mask,
                crossattention_mask=crossattention_mask,
                self_condition=self_condition,
            )

            vt = torch.where(
                constituents_mask.unsqueeze(-1), self.handle_velocity(vt), 0.0
            )

            return vt

        # sample from base distribution
        x1 = self.sample_base(new_batch.x_gen, constituents_mask)

        x0 = odeint(
            velocity,
            x1,
            torch.tensor([1.0, 0.0], device=x1.device, dtype=x1.dtype),
            **self.odeint,
        )[-1]

        sample_batch.x_gen = self.geometry._handle_periodic(x0[constituents_mask])

        if self.cfm.sort:
            # sort generated events by pT
            pt = sample_batch.x_gen[..., 0].unsqueeze(-1)
            x_perm = torch.argsort(pt, dim=0, descending=True)
            sample_batch.x_gen = sample_batch.x_gen.take_along_dim(x_perm, dim=0)
            index = sample_batch.x_gen_batch.unsqueeze(-1).take_along_dim(x_perm, dim=0)
            index_perm = torch.argsort(index, dim=0, stable=True)
            sample_batch.x_gen = sample_batch.x_gen.take_along_dim(index_perm, dim=0)

        return sample_batch.detach()  # , x1.detach()


class EventCFM(CFM):
    """
    Add event-generation-specific methods to CFM classes:
    - Save information at the wrapper level
    - Handle base distribution and coordinates for RFM
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        velocity_mask = torch.ones(1, 4, dtype=torch.bool)
        velocity_mask[:, self.cfm.masked_dims] = False
        self.register_buffer("velocity_mask", velocity_mask, persistent=False)

    def init_coordinates(self):
        self.const_coordinates = getattr(c, self.cfm.const_coordinates)(
            **self.cfm.const_coordinates_options
        )
        self.condition_const_coordinates = getattr(c, self.cfm.const_coordinates)(
            **self.cfm.const_coordinates_options
        )
        self.jet_coordinates = getattr(c, self.cfm.jet_coordinates)(
            **self.cfm.jet_coordinates_options
        )
        self.condition_jet_coordinates = getattr(c, self.cfm.jet_coordinates)(
            **self.cfm.jet_coordinates_options
        )
        if getattr(self.cfm.const_coordinates_options, "vonmises", False):
            self.condition_const_coordinates.vonmises = True
            self.const_coordinates.vonmises = True
        if getattr(self.cfm.jet_coordinates_options, "vonmises", False):
            self.jet_coordinates.vonmises = True
            self.condition_jet_coordinates.vonmises = True
        if self.cfm.transforms_float64:
            self.const_coordinates.to(torch.float64)
            self.condition_const_coordinates.to(torch.float64)
            self.jet_coordinates.to(torch.float64)
            self.condition_jet_coordinates.to(torch.float64)

    def init_geometry(self):
        if getattr(self.cfm.const_coordinates_options, "vonmises", False):
            scale = self.scaling[1].item() / self.const_coordinates.phi_std
        else:
            scale = self.scaling[1].item()
        if self.cfm.geometry.type == "simple":
            self.geometry = SimplePossiblyPeriodicGeometry(
                contains_phi=self.const_coordinates.contains_phi,
                periodic=self.cfm.geometry.periodic,
                scale=scale,
            )
        else:
            raise ValueError(f"geometry={self.cfm.geometry} not implemented")

    def handle_velocity(self, v):
        return v * self.velocity_mask


class JetCFM(EventCFM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaling = torch.tensor(self.cfm.jet_coordinates_options.scaling)

    def sample_base(self, x0, generator=None):
        sample = torch.randn(
            x0.shape, device=x0.device, dtype=x0.dtype, generator=generator
        )
        if self.jet_coordinates.contains_phi:
            if getattr(self.cfm.jet_coordinates_options, "vonmises", False):
                sample[..., 1] = self.jet_coordinates.phi_dist.sample(x0.shape[:-1])
            else:
                sample[..., 1] = (
                    torch.rand(
                        x0.shape[:-1],
                        device=x0.device,
                        dtype=x0.dtype,
                        generator=generator,
                    )
                    * 2
                    * torch.pi
                    - torch.pi
                )
        sample = sample * self.scaling.to(x0.device, dtype=x0.dtype)
        return sample

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

        if self.cfm.add_constituents:
            new_batch, _, _ = add_jet_to_sequence(batch)
        else:
            new_batch = batch.clone()

        x0 = new_batch.jet_gen
        t = torch.rand(
            new_batch.num_graphs,
            1,
            dtype=x0.dtype,
            device=x0.device,
        )

        x1 = self.sample_base(x0)

        xt, vt = self.geometry.get_trajectory(
            x_target=x0,
            x_base=x1,
            t=t,
        )

        attention_mask, condition_attention_mask, crossattention_mask = self.get_masks(
            new_batch
        )

        condition = self.get_condition(new_batch, condition_attention_mask)

        vp = self.get_velocity(
            xt=xt,
            t=t,
            batch=new_batch,
            condition=condition,
            attention_mask=attention_mask,
            crossattention_mask=crossattention_mask,
        )
        vp = self.handle_velocity(vp)
        vt = self.handle_velocity(vt)

        # evaluate conditional flow matching objective
        distance = self.geometry.get_metric(vp, vt, xt)

        loss = distance

        distance_particlewise = ((vp - vt) ** 2).mean(dim=0) / 2

        return loss.mean(), distance_particlewise

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

        if self.cfm.add_constituents:
            new_batch, _, _ = add_jet_to_sequence(batch)
        else:
            new_batch = batch.clone()

        sample_batch = batch.clone()

        attention_mask, condition_attention_mask, crossattention_mask = self.get_masks(
            new_batch
        )

        condition = self.get_condition(new_batch, condition_attention_mask)

        def velocity(t, xt, self_condition=None):
            xt = self.geometry._handle_periodic(xt)
            t = t * torch.ones(xt.shape[0], 1, dtype=xt.dtype, device=xt.device)

            vt = self.get_velocity(
                xt=xt,
                t=t,
                batch=new_batch,
                condition=condition,
                attention_mask=attention_mask,
                crossattention_mask=crossattention_mask,
                self_condition=self_condition,
            )

            vt = self.handle_velocity(vt)

            return vt

        # sample from base distribution
        x1 = self.sample_base(new_batch.jet_gen)

        x0 = odeint(
            velocity,
            x1,
            torch.tensor([1.0, 0.0], device=x1.device),
            **self.odeint,
        )[-1]

        sample_batch.jet_gen = self.geometry._handle_periodic(x0)

        return sample_batch  # , x1
