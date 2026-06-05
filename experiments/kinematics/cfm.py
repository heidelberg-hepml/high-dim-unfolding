import torch
from torch import nn
from torch_geometric.utils import scatter
from torchdiffeq import odeint

import experiments.coordinates as c
from experiments.baselines import custom_rk4
from experiments.embedding import (
    add_jet_to_sequence,
)
from experiments.geometry import BaseGeometry, SimplePossiblyPeriodicGeometry
from experiments.misc import GaussianFourierProjection


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
            GaussianFourierProjection(embed_dim=cfm.embed_t_dim, scale=cfm.embed_t_scale),
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
        sample = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
        if self.const_coordinates.contains_phi:
            if getattr(self.cfm.const_coordinates_options, "vonmises", False):
                sample[..., 1] = (
                    self.const_coordinates.phi_dist.sample(x0.shape[:-1])
                    / self.const_coordinates.phi_std
                )
            elif "JetScaled" in self.cfm.const_coordinates:
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
            new_batch, constituents_mask, _ = add_jet_to_sequence(batch)
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
        )  # .clamp(1e-5, 1 - 1e-5)
        t = torch.repeat_interleave(t, new_batch.x_gen_ptr.diff(), dim=0)

        x1 = self.sample_base(x0, constituents_mask)

        xt, vt = self.geometry.get_trajectory(
            x_target=x0,
            x_base=x1,
            t=t,
        )

        attention_mask, condition_attention_mask, crossattention_mask = self.get_masks(new_batch)

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

        loss = scatter(loss, new_batch.x_gen_batch[constituents_mask], dim=0, reduce="mean")

        if self.cfm.weight > 0.0:
            cost = self.geometry.get_distance(x0[constituents_mask], x1[constituents_mask])
            cost = scatter(cost, new_batch.x_gen_batch[constituents_mask], dim=0, reduce="mean")
            loss = loss * torch.exp(-self.cfm.weight * cost)

        loss = loss.mean()

        # add velocity loss in the mass direction
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
        batch : Batch with gen and det graphs
        device : torch.device
        dtype : torch.dtype

        Returns
        -------
        x0 : torch.tensor with shape (batchsize, 4)
            Generated events
        """

        if self.cfm.add_jet:
            new_batch, constituents_mask, det_constituents_mask = add_jet_to_sequence(batch)
        else:
            new_batch = batch.clone()
            constituents_mask = torch.ones(
                new_batch.x_gen.shape[0],
                device=new_batch.x_gen.device,
                dtype=torch.bool,
            )

        sample_batch = batch.clone()

        attention_mask, condition_attention_mask, crossattention_mask = self.get_masks(new_batch)

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

            vt = torch.where(constituents_mask.unsqueeze(-1), self.handle_velocity(vt), 0.0)

            return vt

        # sample from base distribution
        x1 = self.sample_base(new_batch.x_gen, constituents_mask)

        if self.cfm.self_condition_prob > 0.0:
            v1 = torch.zeros_like(x1, device=x1.device, dtype=x1.dtype)
            x0 = custom_rk4(
                velocity,
                (x1, v1),
                torch.tensor([1.0, 0.0], device=x1.device, dtype=x1.dtype),
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
        sample = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
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

        attention_mask, condition_attention_mask, crossattention_mask = self.get_masks(new_batch)

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

        attention_mask, condition_attention_mask, crossattention_mask = self.get_masks(new_batch)

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
