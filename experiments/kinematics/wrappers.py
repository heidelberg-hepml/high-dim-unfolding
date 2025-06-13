import torch
import numpy as np
from torchdiffeq import odeint

from experiments.utils import xformers_cond_mask
from experiments.kinematics.cfm import EventCFM
from lgatr.interface import embed_vector, extract_vector


class ConditionalTransformerCFM(EventCFM):
    """
    Base class for all CFM models
    - event-generation-specific features are implemented in EventCFM
    - get_velocity is implemented by architecture-specific subclasses
    """

    def __init__(
        self,
        net,
        net_condition,
        cfm,
        odeint,
    ):
        # See GATrCFM.__init__ for documentation
        super().__init__(
            cfm,
            odeint,
        )
        self.net = net
        self.net_condition = net_condition

    def get_masks(self, batch):
        return xformers_cond_mask(
            batch.x_gen_batch, materialize=not torch.cuda.is_available()
        ), xformers_cond_mask(
            batch.x_gen_batch,
            batch.x_det_batch,
            materialize=not torch.cuda.is_available(),
        )

    def get_condition(self, batch):
        mask = xformers_cond_mask(
            batch.x_det_batch, materialize=not torch.cuda.is_available()
        )
        return self.net_condition(batch.x_det.unsqueeze(0), mask)

    def get_velocity(self, x, t, condition, attention_mask, crossattention_mask):
        input = torch.cat([x, self.t_embedding(t)], dim=-1)
        vp = self.net(
            x=input.unsqueeze(0),
            processed_condition=condition,
            attention_mask=attention_mask,
            crossattention_mask=crossattention_mask,
        ).squeeze(0)
        return self.geometry._handle_periodic(vp)


class ConditionalGATrCFM(EventCFM):
    """
    GATr velocity network
    """

    def __init__(
        self,
        net,
        net_condition,
        cfm,
        scalar_dims,
        odeint,
        cfg_data,
    ):
        """
        Parameters
        ----------
        net : torch.nn.Module
        net_condition : torch.nn.Module
        cfm : Dict
            Information about how to set up CFM (used in parent classes)
        scalar_dims : List[int]
            Components within the used parametrization
            for which the equivariantly predicted velocity (using multivector channels)
            is overwritten by a scalar network output (using scalar channels)
            This is required when cfm.coordinates contains log-transforms
        odeint : Dict
            ODE solver settings to be passed to torchdiffeq.odeint
        cfg_data : Dict
            Data settings to be passed to the CFM
        """
        super().__init__(
            cfm,
            odeint,
        )
        self.scalar_dims = scalar_dims
        assert (np.array(scalar_dims) < 4).all() and (np.array(scalar_dims) >= 0).all()
        self.cfg_data = cfg_data
        self.net = net
        self.net_condition = net_condition

    def get_masks(self, batch):
        return xformers_cond_mask(
            batch.x_gen_batch, materialize=not torch.cuda.is_available()
        ), xformers_cond_mask(
            batch.x_gen_batch,
            batch.x_det_batch,
            materialize=not torch.cuda.is_available(),
        )

    def get_condition(self, batch):
        attention_mask = xformers_cond_mask(batch.x_det_batch)
        mv, s = batch.x_det.unsqueeze(0), batch.scalars_det.unsqueeze(0)
        fixed_t = torch.zeros(s.shape[1], 1, dtype=s.dtype, device=s.device)
        t = self.t_embedding(fixed_t).unsqueeze(0)
        s = torch.cat([s, t], dim=-1)
        condition_mv, condition_s = self.net_condition(mv, s, attention_mask)
        return condition_mv, condition_s

    def get_velocity(
        self, xt, t, batch, condition, attention_mask, crossattention_mask
    ):
        assert self.coordinates is not None

        fourmomenta = self.coordinates.x_to_fourmomenta(xt, batch.x_gen_ptr)
        condition_mv, condition_s = condition

        mv, s = self.embed_into_ga(fourmomenta, batch.scalars_gen, t)

        mv_outputs, s_outputs = self.net(
            multivectors=mv.unsqueeze(0),
            multivectors_condition=condition_mv,
            scalars=s.unsqueeze(0),
            scalars_condition=condition_s,
            attention_mask=attention_mask,
            crossattention_mask=crossattention_mask,
        )
        mv_outputs = mv_outputs.squeeze(0)
        s_outputs = s_outputs.squeeze(0)

        v_fourmomenta, v_s = self.extract_from_ga(mv_outputs, s_outputs)

        v_straight = self.coordinates.velocity_fourmomenta_to_x(
            v_fourmomenta,
            fourmomenta,
            batch.x_gen_ptr,
        )[0]

        # Overwrite transformed velocities with scalar outputs
        # (this is specific to GATr to avoid large jacobians from from log-transforms)
        v_straight[..., self.scalar_dims] = v_s[..., self.scalar_dims]

        return v_straight

    def batch_loss(self, batch):
        x0 = batch.x_gen
        t = torch.rand(
            batch.num_graphs,
            1,
            dtype=x0.dtype,
            device=x0.device,
        )
        t = torch.repeat_interleave(t, batch.x_gen_ptr.diff(), dim=0)

        if 3 in self.cfm.masked_dims:
            mass = self.mass
        else:
            mass = None
        x1 = self.sample_base(x0.shape, x0.device, x0.dtype, mass)

        vt = x1 - x0
        xt = self.geometry._handle_periodic(x0 + vt * t)

        condition = self.get_condition(batch)

        attention_mask, crossattention_mask = self.get_masks(batch)

        vp = self.get_velocity(
            xt, t, batch, condition, attention_mask, crossattention_mask
        )

        # vp = self.handle_velocity(vp, batch.x_gen_ptr)
        # vt = self.handle_velocity(vt, batch.x_gen_ptr)

        # evaluate conditional flow matching objective
        distance = ((vp - vt) ** 2).mean()
        distance_particlewise = ((vp - vt) ** 2).mean(dim=0)
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

        condition = self.get_condition(batch)

        attention_mask, crossattention_mask = self.get_masks(batch)

        def velocity(t, xt_straight):
            xt_straight = self.geometry._handle_periodic(xt_straight)
            t = t * torch.ones(
                shape[0], 1, dtype=xt_straight.dtype, device=xt_straight.device
            )
            vt_straight = self.get_velocity(
                xt_straight, t, batch, condition, attention_mask, crossattention_mask
            )
            vt_straight = self.handle_velocity(
                vt_straight, batch.x_gen_ptr
            )  # manually set mass velocity to zero
            return vt_straight

        # sample fourmomenta from base distribution
        shape = batch.x_gen.shape
        if 3 in self.cfm.masked_dims:
            mass = self.mass
        else:
            mass = None
        x1 = self.sample_base(shape, device, dtype, mass)

        # solve ODE in straight space
        x0 = odeint(
            velocity,
            x1,
            torch.tensor([1.0, 0.0], device=x1.device),
            **self.odeint,
        )[-1]

        sample_batch.x_gen = self.geometry._handle_periodic(x0)

        # sort generated events by pT
        # pt = x0[..., 0].unsqueeze(-1)
        # x_perm = torch.argsort(pt, dim=0, descending=True)
        # x0 = x0.take_along_dim(x_perm, dim=0)
        # index = batch.x_gen_batch.unsqueeze(-1).take_along_dim(x_perm, dim=0)
        # index_perm = torch.argsort(index, dim=0, stable=True)
        # x0 = x0.take_along_dim(index_perm, dim=0)

        return sample_batch

    def embed_into_ga(self, fourmomenta, scalars, t):

        # scalar embedding
        t = self.t_embedding(t)
        s = torch.cat([scalars, t], dim=-1)

        mv = embed_vector(fourmomenta).unsqueeze(-2)

        return mv, s

    def extract_from_ga(self, mv, s):
        v = extract_vector(mv).squeeze(dim=-2)
        return v, s
