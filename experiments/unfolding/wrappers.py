import torch
import numpy as np

from experiments.unfolding.cfm import EventCFM
from experiments.unfolding.embedding import embed_into_ga_with_spurions
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
    mask = BlockDiagonalMask.from_seqlens(bincounts, bincounts_condition)
    if materialize:
        # materialize mask to torch.tensor (only for testing purposes)
        mask = mask.materialize(shape=(len(batch), len(batch_condition))).to(
            batch.device
        )

    return mask


class ConditionalCFMForGA(EventCFM):
    def __init__(self, scalar_dims, cfg_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scalar_dims = scalar_dims
        assert (np.array(scalar_dims) < 4).all() and (np.array(scalar_dims) >= 0).all()
        self.cfg_data = cfg_data

    def get_velocity(self, xt, t, batch):
        assert self.coordinates is not None
        input_batch, condition_batch = batch

        fourmomenta = self.coordinates.x_to_fourmomenta(xt)
        condition_fourmomenta = self.coordinates.x_to_fourmomenta(condition_batch.x)
        mv, s = self.embed_into_ga(fourmomenta, input_batch.scalars, t)
        condition_mv, condition_s, condition_batch_indices = (
            embed_into_ga_with_spurions(
                condition_fourmomenta,
                condition_batch.scalars,
                condition_batch.ptr,
                self.cfg_data,
            )
        )

        attention_mask = xformers_sa_mask(input_batch.batch)
        attention_mask_condition = xformers_sa_mask(condition_batch_indices)
        crossattention_mask = xformers_sa_mask(
            input_batch.batch, condition_batch_indices
        )

        mv_outputs, s_outputs = self.net(
            multivectors=mv.unsqueeze(0),
            multivectors_condition=condition_mv.unsqueeze(0),
            scalars=s.unsqueeze(0),
            scalars_condition=condition_s.unsqueeze(0),
            attention_mask=attention_mask,
            attention_mask_condition=attention_mask_condition,
            crossattention_mask=crossattention_mask,
        )
        mv_outputs = mv_outputs.squeeze(0)
        s_outputs = s_outputs.squeeze(0)

        v_fourmomenta, v_s = self.extract_from_ga(mv_outputs, s_outputs)

        v_straight = self.coordinates.velocity_fourmomenta_to_x(
            v_fourmomenta,
            fourmomenta,
        )[0]

        # Overwrite transformed velocities with scalar outputs
        # (this is specific to GATr to avoid large jacobians from from log-transforms)
        v_straight[..., self.scalar_dims] = v_s[..., self.scalar_dims]
        return v_straight


class ConditionalTransformerCFM(EventCFM):
    """
    Conditional Transformer velocity network
    """

    def __init__(
        self,
        net,
        cfm,
        odeint,
    ):
        # See GATrCFM.__init__ for documentation
        super().__init__(
            cfm,
            odeint,
        )
        self.net = net

    def get_velocity(self, xt, t, batch):
        input_batch, condition_batch = batch

        t_embedding = self.t_embedding(t)

        x = torch.cat([xt, input_batch.scalars, t_embedding], dim=-1)
        condition = torch.cat([condition_batch.x, condition_batch.scalars], dim=-1)

        attention_mask = xformers_sa_mask(input_batch.batch)
        attention_mask_condition = xformers_sa_mask(condition_batch.batch)
        crossattention_mask = xformers_sa_mask(input_batch.batch, condition_batch.batch)

        v = self.net(
            x=x.unsqueeze(0),
            condition=condition.unsqueeze(0),
            attention_mask=attention_mask,
            attention_mask_condition=attention_mask_condition,
            crossattention_mask=crossattention_mask,
        )
        v = v.squeeze(0)
        return v


class ConditionalGATrCFM(ConditionalCFMForGA):
    """
    GATr velocity network
    """

    def __init__(
        self,
        net,
        cfm,
        scalar_dims,
        odeint,
        cfg_data,
    ):
        """
        Parameters
        ----------
        net : torch.nn.Module
        cfm : Dict
            Information about how to set up CFM (used in parent classes)
        scalar_dims : List[int]
            Components within the used parametrization
            for which the equivariantly predicted velocity (using multivector channels)
            is overwritten by a scalar network output (using scalar channels)
            This is required when cfm.coordinates contains log-transforms
        odeint : Dict
            ODE solver settings to be passed to torchdiffeq.odeint
        """
        super().__init__(
            scalar_dims,
            cfg_data,
            cfm,
            odeint,
        )
        self.net = net

    def embed_into_ga(self, fourmomenta, scalars, t):

        # scalar embedding
        t = self.t_embedding(t)
        s = torch.cat([scalars, t], dim=-1)

        mv = embed_vector(fourmomenta).unsqueeze(-2)

        return mv, s

    def extract_from_ga(self, mv, s):
        v = extract_vector(mv).squeeze(dim=-2)
        return v, s
