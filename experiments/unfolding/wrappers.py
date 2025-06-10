import torch
import numpy as np
from torchdiffeq import odeint

from experiments.unfolding.autoregression import (
    add_start_tokens,
    start_sequence,
    insert_tokens,
    remove_extra,
    create_block_mask,
)
from experiments.unfolding.cfm import EventCFM
from experiments.unfolding.utils import ensure_angle, mask_dims
from gatr.interface import embed_vector, extract_vector
from experiments.logger import LOGGER

from xformers.ops.fmha.attn_bias import BlockDiagonalMask


def xformers_sa_mask(batch, batch_condition=None, materialize=False):
    """
    Construct attention mask that makes sure that objects only attend to each other
    within the same batch element, and not across batch elements

    Parameters
    ----------
    batch: torch.tensor
        batch object in the torch_geometric.data naming convention
        contains batch index for each event in a sparse tensor
    materialize: bool
        Decides whether a xformers or ('materialized') torch.tensor mask should be returned
        The xformers mask allows to use the optimized xformers attention kernel, but only runs on gpu

    Returns
    -------
    mask: xformers.ops.fmha.attn_bias.BlockDiagonalMask or torch.tensor
        attention mask, to be used in xformers.ops.memory_efficient_attention
        or torch.nn.functional.scaled_dot_product_attention
    """
    bincounts = torch.bincount(batch).tolist()
    if batch_condition is not None:
        bincounts_condition = torch.bincount(batch_condition).tolist()
    else:
        bincounts_condition = bincounts
        batch_condition = batch
    mask = BlockDiagonalMask.from_seqlens(bincounts, bincounts_condition)
    if materialize:
        # materialize mask to torch.tensor (only for testing purposes)
        mask = mask.materialize(shape=(len(batch), len(batch_condition))).to(
            batch.device
        )

    return mask


@torch.compile(dynamic=True)
def full_self_attention_mask(batch):
    def masking(b, h, q_idx, kv_idx):
        return batch[q_idx] == batch[kv_idx]

    return create_block_mask(
        masking, B=None, H=None, Q_LEN=len(batch), KV_LEN=len(batch)
    )


@torch.compile(dynamic=True)
def causal_self_attention_mask(batch):
    def masking(b, h, q_idx, kv_idx):
        return torch.where(q_idx < kv_idx, True, False) * torch.where(
            batch[q_idx] == batch[kv_idx], True, False
        )

    return create_block_mask(
        masking, B=None, H=None, Q_LEN=len(batch), KV_LEN=len(batch)
    )


@torch.compile(dynamic=True)
def cross_attention_mask(Q_batch, KV_batch):
    def masking(b, h, q_idx, kv_idx):
        return torch.where(Q_batch[q_idx] == KV_batch[kv_idx], True, False)

    return create_block_mask(
        masking, B=None, H=None, Q_LEN=len(Q_batch), KV_LEN=len(KV_batch)
    )


class ConditionalMLPCFM(EventCFM):
    """
    Conditional Transformer velocity network
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

    def get_condition(self, batch):
        # condition_x = self.condition_coordinates.fourmomenta_to_x(batch.x_det)
        # condition = torch.cat([condition_x, batch.scalars_det], dim=-1)
        # processed_condition = self.net_condition(
        #     inputs=condition,
        # )
        return torch.zeros_like(batch.x_gen)

    def get_velocity(self, xt, t, batch, processed_condition):
        t_embedding = self.t_embedding(t)

        x = torch.cat([xt, batch.scalars_gen, t_embedding], dim=-1)

        v = self.net(
            inputs=x,
        )

        return v


class ConditionalAutoregressiveTransformerCFM(EventCFM):
    """
    Conditional Autoregressive Transformer velocity network
    """

    def __init__(
        self,
        autoregressive_tr,
        net_condition,
        mlp,
        cfm,
        odeint,
    ):
        # See GATrCFM.__init__ for documentation
        super().__init__(
            cfm,
            odeint,
        )
        self.autoregressive_tr = autoregressive_tr
        self.net_condition = net_condition
        self.mlp = mlp

    def get_condition(self, batch, add_start=True):
        if add_start:
            new_batch = add_start_tokens(batch)
        else:
            new_batch = batch.clone()

        x = self.coordinates.fourmomenta_to_x(new_batch.x_gen)
        condition_x = self.condition_coordinates.fourmomenta_to_x(new_batch.x_det)
        condition = torch.cat([condition_x, new_batch.scalars_det], dim=-1)

        attention_mask = causal_self_attention_mask(new_batch.x_gen_batch)
        attention_mask_condition = full_self_attention_mask(new_batch.x_det_batch)
        crossattention_mask = cross_attention_mask(
            new_batch.x_gen_batch, new_batch.x_det_batch
        )

        processed_condition = self.net_condition(
            condition.unsqueeze(0), attention_mask_condition
        )

        autoregressive_condition = self.autoregressive_tr(
            x=x.unsqueeze(0),
            processed_condition=processed_condition,
            attention_mask=attention_mask,
            crossattention_mask=crossattention_mask,
        ).squeeze(0)
        new_batch.x_gen = autoregressive_condition
        return new_batch

    def get_velocity(self, xt, t, batch, batch_with_cond):

        t_embedding = self.t_embedding(t)

        condition = remove_extra(batch_with_cond, batch.x_gen_ptr)
        input = torch.cat([xt, t_embedding, condition.x_gen], dim=-1)
        # input = torch.cat([xt, t_embedding], dim=-1)

        v = self.mlp(input)
        return v

    def sample(self, batch, device, dtype):

        max_constituents = torch.bincount(batch.x_gen_batch).max().item()
        sequence = start_sequence(batch)
        shape = (batch.x_gen_batch[-1].item() + 1, *batch.x_gen.shape[1:])

        for i in range(max_constituents):
            new_batch = self.get_condition(sequence, add_start=False)
            condition = new_batch.x_gen[sequence.x_gen_ptr[1:] - 1]

            def velocity(t, xt_straight):
                xt_straight = self.geometry._handle_periodic(xt_straight)
                t = t * torch.ones(
                    shape[0], 1, dtype=xt_straight.dtype, device=xt_straight.device
                )
                t_embedding = self.t_embedding(t)
                input = torch.cat([xt_straight, t_embedding, condition], dim=-1)
                # input = torch.cat([xt_straight, t_embedding], dim=-1)

                v = self.mlp(input)
                v = self.handle_velocity(v, new_batch.x_gen_ptr)
                return v

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

            # transform generated event back to fourmomenta
            x0_fourmomenta = self.coordinates.x_to_fourmomenta(x0_straight)

            sequence = insert_tokens(sequence, x0_fourmomenta)

        samples = remove_extra(sequence, batch.x_gen_ptr, remove_start=True)
        return samples


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
        return xformers_sa_mask(
            batch.x_gen_batch, materialize=not torch.cuda.is_available()
        ), xformers_sa_mask(
            batch.x_gen_batch,
            batch.x_det_batch,
            materialize=not torch.cuda.is_available(),
        )

    def sample_base(self, shape, device, dtype, mass=None, generator=None):
        sample = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        sample[..., 1] = (
            torch.rand(shape[:-1], device=device, dtype=dtype, generator=generator)
            * 2
            * torch.pi
            - torch.pi
        )
        if mass is not None:
            sample[..., 3] = mass
        return sample

    def get_condition(self, batch):
        mask = xformers_sa_mask(
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
        # return ensure_angle(vp)
        return vp

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
        x0 = batch.x_gen
        t = torch.rand(
            batch.num_graphs,
            1,
            dtype=x0.dtype,
            device=x0.device,
        )
        t = torch.repeat_interleave(t, batch.x_gen_ptr.diff(), dim=0)

        if 3 in self.cfm.masked_dims:
            mass = self.onshell_mass
        else:
            mass = None
        x1 = self.sample_base(x0.shape, x0.device, x0.dtype, mass)

        if self.cfm.mask_jets:
            x1[batch.x_gen_ptr[:-1]] = x0[batch.x_gen_ptr[:-1]]

        vt = x1 - x0
        xt = self.geometry._handle_periodic(x0 + vt * t)

        condition = self.get_condition(batch)

        attention_mask, crossattention_mask = self.get_masks(batch)

        vp = self.get_velocity(xt, t, condition, attention_mask, crossattention_mask)

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
                xt_straight, t, condition, attention_mask, crossattention_mask
            )
            vt_straight = mask_dims(vt_straight, self.cfm.masked_dims)
            vt_straight = self.handle_velocity(
                vt_straight, batch.x_gen_ptr
            )  # manually set mass velocity to zero
            return vt_straight

        # sample fourmomenta from base distribution
        shape = batch.x_gen.shape
        if 3 in self.cfm.masked_dims:
            mass = self.onshell_mass
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
        raise NotImplementedError


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

    def sample_base(self, shape, device, dtype, mass=None, generator=None):
        sample = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        sample[..., 1] = (
            torch.rand(shape[:-1], device=device, dtype=dtype, generator=generator)
            * 2
            * torch.pi
            - torch.pi
        )
        if mass is not None:
            sample[..., 3] = mass
        return sample

    def get_masks(self, batch):
        return xformers_sa_mask(
            batch.x_gen_batch, materialize=not torch.cuda.is_available()
        ), xformers_sa_mask(
            batch.x_gen_batch,
            batch.x_det_batch,
            materialize=not torch.cuda.is_available(),
        )

    def get_condition(self, batch):
        attention_mask = xformers_sa_mask(batch.x_det_batch)
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
            mass = self.onshell_mass
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
            vt_straight = mask_dims(vt_straight, self.cfm.masked_dims)
            vt_straight = self.handle_velocity(
                vt_straight, batch.x_gen_ptr
            )  # manually set mass velocity to zero
            return vt_straight

        # sample fourmomenta from base distribution
        shape = batch.x_gen.shape
        if 3 in self.cfm.masked_dims:
            mass = self.onshell_mass
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
