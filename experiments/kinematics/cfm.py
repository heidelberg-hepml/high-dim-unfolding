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
        vp = self.handle_velocity(vp[constituents_mask])
        vt = self.handle_velocity(vt[constituents_mask])

        # evaluate conditional flow matching objective
        distance = self.geometry.get_metric(vp, vt, xt[constituents_mask])

        if self.cfm.cosine_similarity_factor > 0.0:
            cosine_similarity = (
                1 - (vp * vt).sum(dim=-1) / (vp.norm(dim=-1) * vt.norm(dim=-1))
            ).mean()
            loss = (
                1 - self.cfm.cosine_similarity_factor
            ) * distance + self.cfm.cosine_similarity_factor * cosine_similarity
        else:
            loss = distance

        loss = scatter(
            loss, new_batch.x_gen_batch[constituents_mask], dim=0, reduce="mean"
        )

        if self.cfm.weight > 0.0:
            cost = self.geometry.get_distance(
                x0[constituents_mask], x1[constituents_mask]
            )
            cost = scatter(
                cost, new_batch.x_gen_batch[constituents_mask], dim=0, reduce="mean"
            )
            loss = loss * torch.exp(-self.cfm.weight * cost)

        loss = loss.mean()

        # if self.cfm.add_mass:
        #     gen_jets = torch.repeat_interleave(
        #         batch.jet_gen, batch.x_gen_ptr.diff(), dim=0
        #     )

        #     def get_mass(x, ptr):
        #         y = self.coordinates.x_to_fourmomenta(x, ptr=ptr, jet=gen_jets)
        #         new_jets = scatter(y, batch.x_gen_batch, dim=0, reduce="sum")
        #         jet_mass = new_jets[..., 0] ** 2 - (new_jets[..., 1:] ** 2).sum(dim=-1)
        #         jet_mass = torch.clamp(jet_mass, min=0.0).sqrt()
        #         return jet_mass

        #     true_v_mass = get_mass(
        #         fix_mass(x1[constituents_mask]), batch.x_gen_ptr
        #     ) - get_mass(fix_mass(x0[constituents_mask]), batch.x_gen_ptr)
        #     x = fix_mass(xt[constituents_mask])
        #     x.requires_grad_(True)
        #     deriv = torch.autograd.grad(
        #         outputs=get_mass(x, batch.x_gen_ptr).sum(),
        #         inputs=x,
        #         retain_graph=True,
        #     )[0]
        #     pred_v_mass = torch.einsum("ij,ij->i", deriv, vp)
        #     pred_v_mass = scatter(pred_v_mass, batch.x_gen_batch, dim=0, reduce="sum")
        #     extra_loss = ((pred_v_mass - true_v_mass) ** 2).mean()
        #     LOGGER.info(f"Mass loss: {extra_loss.item():.4f}, loss: {loss.item():.4f}")
        #     LOGGER.info(
        #         f"mass velocity: {pred_v_mass.mean().item():.4f}, "
        #         f"true mass velocity: {true_v_mass.mean().item():.4f}"
        #     )
        #     LOGGER.info(
        #         f"True mass: {get_mass(fix_mass(x1[constituents_mask]), batch.x_gen_ptr).mean().item():.4f}"
        #     )
        #     LOGGER.info(
        #         f"Base mass: {get_mass(fix_mass(x0[constituents_mask]), batch.x_gen_ptr).mean().item():.4f}"
        #     )
        #     LOGGER.info(f"xt mass: {get_mass(x, batch.x_gen_ptr).mean().item():.4f}")
        #     loss = loss + 1e-6 * extra_loss

        distance_particlewise = ((vp - vt) ** 2).mean(dim=0) / 2
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
        if self.cfm.const_coordinates_options.vonmises:
            self.condition_const_coordinates.vonmises = True
            self.const_coordinates.vonmises = True
        if self.cfm.jet_coordinates_options.vonmises:
            self.jet_coordinates.vonmises = True
            self.condition_jet_coordinates.vonmises = True
        if self.cfm.transforms_float64:
            self.const_coordinates.to(torch.float64)
            self.condition_const_coordinates.to(torch.float64)
            self.jet_coordinates.to(torch.float64)
            self.condition_jet_coordinates.to(torch.float64)

    def init_geometry(self):

        # placeholder for any initialization that needs to be done
        if self.cfm.const_coordinates_options.vonmises:
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
        vp = self.handle_velocity(vp)
        vt = self.handle_velocity(vt)

        # evaluate conditional flow matching objective
        distance = self.geometry.get_metric(vp, vt, xt)

        if self.cfm.cosine_similarity_factor > 0.0:
            cosine_similarity = (
                1 - (vp * vt).sum(dim=-1) / (vp.norm(dim=-1) * vt.norm(dim=-1))
            ).mean()
            loss = (
                1 - self.cfm.cosine_similarity_factor
            ) * distance + self.cfm.cosine_similarity_factor * cosine_similarity
        else:
            loss = distance

        if self.cfm.weight > 0.0:
            cost = self.geometry.get_distance(x0, x1)
            loss = loss * torch.exp(-self.cfm.weight * cost)

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

        sample_batch.jet_gen = self.geometry._handle_periodic(x0)

        return sample_batch  # , x1


class JetMLPCFM(EventCFM):

    def __init__(
        self,
        net,
        cfm,
        odeint,
    ):
        super().__init__(
            cfm,
            odeint,
        )
        self.net = net

    def batch_loss(self, batch):

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

        if self.cfm.self_condition_prob > 0.0:
            self_condition = torch.zeros_like(vt, device=vt.device, dtype=vt.dtype)
            if torch.rand(1) < self.cfm.self_condition_prob:
                self_condition = self.get_velocity(
                    xt=xt,
                    t=t,
                    batch=new_batch,
                    self_condition=self_condition,
                ).detach()

            vp = self.get_velocity(
                xt=xt,
                t=t,
                batch=new_batch,
                self_condition=self_condition,
            )
        else:
            vp = self.get_velocity(
                xt=xt,
                t=t,
                batch=new_batch,
            )
        vp = self.handle_velocity(vp)
        vt = self.handle_velocity(vt)

        # evaluate conditional flow matching objective
        distance = self.geometry.get_metric(vp, vt, xt)

        if self.cfm.cosine_similarity_factor > 0.0:
            cosine_similarity = (
                1 - (vp * vt).sum(dim=-1) / (vp.norm(dim=-1) * vt.norm(dim=-1))
            ).mean()
            loss = (
                1 - self.cfm.cosine_similarity_factor
            ) * distance + self.cfm.cosine_similarity_factor * cosine_similarity
        else:
            loss = distance

        distance_particlewise = ((vp - vt) ** 2).mean(dim=0) / 2
        return loss.mean(), distance_particlewise

    def sample(self, batch, device, dtype):

        new_batch = batch.clone()

        sample_batch = batch.clone()

        def velocity(t, xt, self_condition=None):
            xt = self.geometry._handle_periodic(xt)
            t = t * torch.ones(xt.shape[0], 1, dtype=xt.dtype, device=xt.device)

            vt = self.get_velocity(
                xt=xt,
                t=t,
                batch=new_batch,
                self_condition=self_condition,
            )

            vt = self.handle_velocity(vt)

            return vt

        # sample from base distribution
        x1 = self.sample_base(new_batch.jet_gen)

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

        sample_batch.jet_gen = self.geometry._handle_periodic(x0)

        return sample_batch  # , x1

    def get_velocity(
        self,
        xt,
        t,
        batch,
        self_condition=None,
    ):
        inputs_list = [xt, batch.jet_scalars_gen, self.t_embedding(t)]
        if not self.cfm.unconditional:
            inputs_list += [batch.jet_det, batch.jet_scalars_det]
        if self_condition is not None:
            inputs_list.append(self_condition)

        input = torch.cat(inputs_list, dim=-1)
        vp = self.net(input)
        return vp


class AutoregressiveCFM(EventCFM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jet_scaling = torch.tensor(self.cfm.jet_coordinates_options.scaling)
        self.const_scaling = torch.tensor(self.cfm.const_coordinates_options.scaling)

    def get_condition(
        self, batch, attention_mask, condition_attention_mask, crossattention_mask
    ):
        """
        Return a fixed_sized condition for the velocity field based on all previously generated tokens
        """
        raise NotImplementedError

    def get_velocity(self, xt, t, batch, condition, self_condition=None):
        """
        Return the velocity field for a batch of tokens xt conditioned on a fixed-sized condition for each one
        """
        raise NotImplementedError

    def sample_base_jet(self, x0, generator=None):
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
        sample = sample * self.jet_scaling.to(x0.device, dtype=x0.dtype)
        return sample

    def sample_base_const(self, x0, generator=None):
        sample = torch.randn(
            x0.shape, device=x0.device, dtype=x0.dtype, generator=generator
        )
        if self.const_coordinates.contains_phi and getattr(
            self.cfm.const_coordinates_options, "vonmises", False
        ):
            sample[..., 1] = self.const_coordinates.phi_dist.sample(x0.shape[:-1])
        elif (
            self.const_coordinates.contains_phi
            and "JetScaled" not in self.cfm.const_coordinates
        ):
            sample[..., 1] = (
                torch.rand(
                    x0.shape[:-1], device=x0.device, dtype=x0.dtype, generator=generator
                )
                * 2
                * torch.pi
                - torch.pi
            )
        sample = sample * self.const_scaling.to(x0.device, dtype=x0.dtype)
        sample[..., self.cfm.masked_dims] = x0[..., self.cfm.masked_dims]
        return sample

    def sample_base(self, x0, sequence_mask, constituents_mask, generator=None):
        sample = torch.zeros(x0.shape, device=x0.device, dtype=x0.dtype)
        sample[~sequence_mask] = x0[~sequence_mask]  # keep padding fixed
        new_sequence = x0[sequence_mask].clone()
        new_sequence[constituents_mask] = self.sample_base_const(
            new_sequence[constituents_mask], generator=generator
        )
        new_sequence[~constituents_mask] = self.sample_base_jet(
            new_sequence[~constituents_mask], generator=generator
        )
        sample[sequence_mask] = new_sequence
        return sample

    def handle_velocity(self, v):
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
        new_batch, constituents_mask, det_constituents_mask = add_jet_to_sequence(batch)
        new_batch, sequence_mask = add_start_token_to_x_gen(new_batch)
        if self.cfm.stop_token:
            new_batch, sequence_mask = add_stop_token_to_x_gen(new_batch)
            sequence_mask[new_batch.x_gen_ptr[:-1]] = False  # flag start tokens

        x0 = new_batch.x_gen
        t = torch.rand(
            new_batch.num_graphs,
            1,
            dtype=x0.dtype,
            device=x0.device,
        )
        t = torch.repeat_interleave(t, new_batch.x_gen_ptr.diff(), dim=0)

        x1 = self.sample_base(x0, sequence_mask, constituents_mask)

        plot_kinematics(
            "./runs/autor_test",
            new_batch.x_det[new_batch.x_det_ptr[:-1]],
            new_batch.x_gen[new_batch.x_gen_ptr[:-1]],
            x1[new_batch.x_gen_ptr[:-1]],
            filename="jets_det.pdf",
        )
        plot_kinematics(
            "./runs/autor_test",
            new_batch.x_gen[new_batch.x_gen_ptr[:-1]],
            new_batch.x_gen[new_batch.x_gen_ptr[:-1] + 1],
            x1[new_batch.x_gen_ptr[:-1] + 1],
            filename="jets_gen.pdf",
        )
        plot_kinematics(
            "./runs/autor_test",
            new_batch.x_det[det_constituents_mask],
            new_batch.x_gen[sequence_mask][constituents_mask],
            x1[sequence_mask][constituents_mask],
            filename="constituents_step.pdf",
        )

        xt, vt = self.geometry.get_trajectory(
            x_target=x0,
            x_base=x1,
            t=t,
        )

        attention_mask, condition_attention_mask, crossattention_mask = self.get_masks(
            new_batch
        )

        condition = self.get_condition(
            new_batch, attention_mask, condition_attention_mask, crossattention_mask
        )

        if not self.cfm.stop_token:
            stop_channels = condition[..., -1]
            target_stop_channels = torch.zeros_like(
                stop_channels, dtype=torch.float32, device=stop_channels.device
            )
            target_stop_channels[new_batch.x_gen_ptr[1:] - 1] = 1.0
            num_pos = target_stop_channels.sum()
            num_tokens = target_stop_channels.numel()
            pos_weight = (num_tokens - num_pos) / num_pos
            stop_loss = nn.BCEWithLogitsLoss(pos_weight)(
                stop_channels, target_stop_channels
            )
            condition = condition[..., :-1]

        if self.cfm.self_condition_prob > 0.0:
            self_condition = torch.zeros_like(vt, device=vt.device, dtype=vt.dtype)
            if torch.rand(1) < self.cfm.self_condition_prob:
                self_condition = self.get_velocity(
                    xt=xt,
                    t=t,
                    batch=new_batch,
                    condition=condition,
                    self_condition=self_condition,
                ).detach()

            vp = self.get_velocity(
                xt=xt,
                t=t,
                batch=new_batch,
                condition=condition,
                self_condition=self_condition,
            )
        else:
            vp = self.get_velocity(
                xt=xt,
                t=t,
                batch=new_batch,
                condition=condition,
            )
        vp = self.handle_velocity(vp[sequence_mask])
        vt = self.handle_velocity(vt[sequence_mask])

        # evaluate conditional flow matching objective
        distance = self.geometry.get_metric(vp, vt, xt[sequence_mask])

        # if self.cfm.cosine_similarity_factor > 0.0:
        #     cosine_similarity = (
        #         1 - (vp * vt).sum(dim=-1) / (vp.norm(dim=-1) * vt.norm(dim=-1))
        #     ).mean()
        #     loss = (
        #         1 - self.cfm.cosine_similarity_factor
        #     ) * distance + self.cfm.cosine_similarity_factor * cosine_similarity
        # else:
        #     loss = distance

        jet_loss = distance[~constituents_mask]
        const_loss = scatter(
            distance[constituents_mask],
            new_batch.x_gen_batch[sequence_mask][constituents_mask],
            dim=0,
            reduce="mean",
        )

        # if self.cfm.weight > 0.0:
        #     cost = self.geometry.get_distance(x0[sequence_mask], x1[sequence_mask])
        #     cost = scatter(
        #         cost, new_batch.x_gen_batch[sequence_mask], dim=0, reduce="mean"
        #     )
        #     loss = loss * torch.exp(-self.cfm.weight * cost)

        # loss = loss.mean()

        loss = (
            self.cfm.const_scale * const_loss.mean()
            + self.cfm.jet_scale * jet_loss.mean()
        )

        const_metrics = ((vp[constituents_mask] - vt[constituents_mask]) ** 2).mean(
            dim=0
        ) / 2
        jet_metrics = ((vp[~constituents_mask] - vt[~constituents_mask]) ** 2).mean(
            dim=0
        ) / 2
        metrics = torch.cat([const_metrics, jet_metrics], dim=0)

        if self.cfm.stop_scale > 0.0 and not self.cfm.stop_token:
            loss = loss + self.cfm.stop_scale * stop_loss
            metrics = torch.cat([metrics, stop_loss.unsqueeze(0)], dim=0)

        return loss, metrics

    def sample(self, batch, device=None, dtype=None):
        """
        Autoregressive sampling that supports:
        - STOP TOKEN mode (self.cfm.stop_token=True): EOS is a token; 3 flag channels.
        - STOP CHANNEL mode (self.cfm.stop_token=False): EOS is a channel logit; 2 flag channels.
        """
        B = len(batch.x_gen_ptr) - 1
        device = device or batch.x_gen.device
        dtype = dtype or batch.x_gen.dtype

        # channels
        pe_dim = 8
        n_flag = (
            3 if self.cfm.stop_token else 2
        )  # <-- one fewer channel without stop token
        pos_enc = positional_encoding(
            seq_length=self.cfm.max_seq_len, pe_dim=pe_dim
        ).to(
            device, dtype
        )  # (max_len, pe_dim)
        max_len = pos_enc.size(0)

        def make_scalar_row(pos_idx, is_jet_det=False, is_jet_gen=False):
            """Build scalars_gen row with correct channel count."""
            pe = (
                torch.zeros(pe_dim, device=device, dtype=dtype)
                if pos_idx is None
                else pos_enc[pos_idx]
            )
            if self.cfm.stop_token:
                # [pos8, after_jet_gen, start_flag, end_flag]
                if is_jet_det:
                    flags = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
                elif is_jet_gen:
                    flags = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
                else:
                    flags = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)
            else:
                # [pos8, after_jet_gen, start_flag]  (no EOS one-hot)
                if is_jet_det:
                    flags = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
                elif is_jet_gen:
                    flags = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
                else:
                    flags = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
            return torch.cat([pe, flags], dim=0).unsqueeze(0)

        # Seed per sequence with jet_det
        generated_tokens = [
            [batch.jet_det[i].to(device, dtype).unsqueeze(0)] for i in range(B)
        ]
        generated_scalars = [[make_scalar_row(None, is_jet_det=True)] for _ in range(B)]
        active = torch.arange(B, device=device)

        # tmp_batch scaffold
        new_batch, _, _ = add_jet_to_sequence(batch)
        tmp_batch = new_batch.clone()

        # Pre-loop: sample jet_gen with sample_base_jet
        prefix_x = torch.cat(
            [torch.cat(generated_tokens[i], dim=0) for i in active], dim=0
        )
        ptr = [0]
        for i in active.tolist():
            ptr.append(ptr[-1] + len(generated_tokens[i]))
        tmp_batch.x_gen = prefix_x
        tmp_batch.x_gen_ptr = torch.tensor(ptr, device=device, dtype=torch.long)
        tmp_batch.x_gen_batch = get_batch_from_ptr(tmp_batch.x_gen_ptr)
        tmp_batch.scalars_gen = torch.cat(
            [torch.cat(generated_scalars[i], dim=0) for i in active], dim=0
        )

        attn_mask, cond_attn_mask, cross_attn_mask = self.get_masks(tmp_batch)
        condition = self.get_condition(
            tmp_batch, attn_mask, cond_attn_mask, cross_attn_mask
        )

        last_token_idx = [ptr[i + 1] - 1 for i in range(len(ptr) - 1)]
        last_condition = condition[last_token_idx]

        if not self.cfm.stop_token:
            last_condition = last_condition[..., :-1]

        x1 = self.sample_base_jet(
            torch.cat([generated_tokens[i][-1] for i in active], dim=0)
        )

        def velocity_jet(t, xt, self_condition=None):
            xt = self.geometry._handle_periodic(xt)
            t = t.view(1, 1).expand(xt.shape[0], -1)
            vt = self.get_velocity(
                xt=xt,
                t=t,
                batch=tmp_batch,
                condition=last_condition,
                self_condition=self_condition,
            )
            return self.handle_velocity(vt)

        if self.cfm.self_condition_prob > 0.0:
            v1 = torch.zeros_like(x1)
            x_token = custom_rk4(
                velocity_jet,
                (x1, v1),
                torch.tensor([1.0, 0.0], device=device),
                step_size=self.odeint.options["step_size"],
            )[-1]
        else:
            x_token = odeint(
                velocity_jet,
                x1,
                torch.tensor([1.0, 0.0], device=device, dtype=dtype),
                **self.odeint,
            )[-1]
        x_token = self.geometry._handle_periodic(x_token)

        # append jet_gen to every active sequence
        for j, seq_idx in enumerate(active.tolist()):
            generated_tokens[seq_idx].append(x_token[j].unsqueeze(0))
            generated_scalars[seq_idx].append(make_scalar_row(None, is_jet_gen=True))

        # Loop: subsequent tokens with sample_base_const
        stop_threshold = getattr(self.cfm, "stop_threshold", 0.5)

        while len(active) > 0:
            prefix_x = torch.cat(
                [torch.cat(generated_tokens[i], dim=0) for i in active], dim=0
            )
            ptr = [0]
            for i in active.tolist():
                ptr.append(ptr[-1] + len(generated_tokens[i]))
            tmp_batch.x_gen = prefix_x
            tmp_batch.x_gen_ptr = torch.tensor(ptr, device=device, dtype=torch.long)
            tmp_batch.x_gen_batch = get_batch_from_ptr(tmp_batch.x_gen_ptr)
            tmp_batch.scalars_gen = torch.cat(
                [torch.cat(generated_scalars[i], dim=0) for i in active], dim=0
            )

            det_mask = torch.isin(new_batch.x_det_batch, active)

            tmp_batch.x_det = new_batch.x_det[det_mask]
            tmp_batch.scalars_det = new_batch.scalars_det[det_mask]

            det_mults = new_batch.x_det_ptr.diff()
            tmp_batch.x_det_ptr = torch.cat(
                [
                    torch.zeros(1, device=device, dtype=torch.long),
                    torch.cumsum(det_mults[active], dim=0),
                ]
            )
            tmp_batch.x_det_batch = get_batch_from_ptr(tmp_batch.x_det_ptr)

            attn_mask, cond_attn_mask, cross_attn_mask = self.get_masks(tmp_batch)
            condition = self.get_condition(
                tmp_batch, attn_mask, cond_attn_mask, cross_attn_mask
            )

            last_token_idx = [ptr[i + 1] - 1 for i in range(len(ptr) - 1)]
            last_condition = condition[last_token_idx]

            if not self.cfm.stop_token:
                stop_channel = last_condition[..., -1]
                last_condition = last_condition[..., :-1]

            # propose next token from last generated token
            x1 = self.sample_base_const(
                torch.cat([generated_tokens[i][-1] for i in active], dim=0)
            )

            def velocity_const(t, xt, self_condition=None):
                xt = self.geometry._handle_periodic(xt)
                t = t.view(1, 1).expand(xt.shape[0], -1)
                vt = self.get_velocity(
                    xt=xt,
                    t=t,
                    batch=tmp_batch,
                    condition=last_condition,
                    self_condition=self_condition,
                )
                return self.handle_velocity(vt)

            if self.cfm.self_condition_prob > 0.0:
                v1 = torch.zeros_like(x1)
                x_token = custom_rk4(
                    velocity_const,
                    (x1, v1),
                    torch.tensor([1.0, 0.0], device=device),
                    step_size=self.odeint.options["step_size"],
                )[-1]
            else:
                x_token = odeint(
                    velocity_const,
                    x1,
                    torch.tensor([1.0, 0.0], device=device, dtype=dtype),
                    **self.odeint,
                )[-1]

            x_token = self.geometry._handle_periodic(x_token)

            # decide which sequences remain active
            still_active = []
            for j, seq_idx in enumerate(active.tolist()):
                # check termination
                if self.cfm.stop_token:
                    # EOS detected from the token itself
                    if stop_threshold_fn(x_token[j]):  # your existing EOS test
                        continue
                else:
                    # EOS decision from channel logit on *current last position*
                    stop_logit = stop_channel[j]
                    stop_prob = torch.sigmoid(stop_logit)
                    if stop_prob.item() > stop_threshold:
                        continue

                # otherwise accept token
                tok = x_token[j].unsqueeze(0)
                pos_idx = len(generated_tokens[seq_idx]) - 2  # 0-based after jet_gen
                if pos_idx >= max_len:
                    # reached positional budget
                    continue

                generated_tokens[seq_idx].append(tok)
                generated_scalars[seq_idx].append(make_scalar_row(pos_idx))
                still_active.append(seq_idx)

            active = (
                torch.tensor(still_active, device=device, dtype=torch.long)
                if len(still_active) > 0
                else torch.tensor([], device=device, dtype=torch.long)
            )

        # ---------- Pack final graphs (drop jet_det and jet_gen) ----------
        data_list = []
        for i in range(B):
            seq = torch.cat(generated_tokens[i], dim=0)
            scal = torch.cat(generated_scalars[i], dim=0)
            if seq.size(0) < 3:  # need at least jet_det + jet_gen + one token
                continue
            x_core = seq[2:]  # drop jet_det, jet_gen
            s_core = scal[2:]  # drop their scalar rows
            jet_gen = seq[1].unsqueeze(0)
            data_list.append(
                Data(
                    x_gen=x_core,
                    scalars_gen=s_core,
                    jet_gen=jet_gen,
                    x_det=batch.x_det[batch.x_det_batch == i],
                    scalars_det=batch.scalars_det[batch.x_det_batch == i],
                    jet_det=batch.jet_det[i].unsqueeze(0),
                )
            )

        return Batch.from_data_list(data_list, follow_batch=["x_gen", "x_det"])
