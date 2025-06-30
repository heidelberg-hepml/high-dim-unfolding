import torch
import numpy as np
from lgatr.interface import extract_vector

from experiments.utils import xformers_mask
from experiments.kinematics.cfm import EventCFM
from experiments.embedding import embed_data_into_ga
from experiments.coordinates import jetmomenta_to_fourmomenta
from experiments.logger import LOGGER


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
        self.use_xformers = torch.cuda.is_available()

    def get_masks(self, batch):
        attention_mask = xformers_mask(
            batch.x_gen_batch, materialize=not self.use_xformers
        )
        condition_attention_mask = xformers_mask(
            batch.x_det_batch, materialize=not self.use_xformers
        )
        cross_attention_mask = xformers_mask(
            batch.x_gen_batch,
            batch.x_det_batch,
            materialize=not self.use_xformers,
        )
        return attention_mask, condition_attention_mask, cross_attention_mask

    def get_condition(self, batch, attention_mask):
        input = torch.cat([batch.x_det, batch.scalars_det], dim=-1)
        attn_kwargs = {
            "attn_bias" if self.use_xformers else "attn_mask": attention_mask
        }
        return self.net_condition(input.unsqueeze(0), **attn_kwargs)

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
        if self_condition is not None:
            input = torch.cat(
                [xt, batch.scalars_gen, self.t_embedding(t), self_condition], dim=-1
            )
        else:
            input = torch.cat([xt, batch.scalars_gen, self.t_embedding(t)], dim=-1)
        vp = self.net(
            x=input.unsqueeze(0),
            processed_condition=condition,
            attn_kwargs={
                "attn_bias" if self.use_xformers else "attn_mask": attention_mask
            },
            crossattn_kwargs={
                "attn_bias" if self.use_xformers else "attn_mask": crossattention_mask
            },
        ).squeeze(0)
        return vp


class ConditionalLGATrCFM(EventCFM):
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
        GA_config,
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
        self.ga_cfg = GA_config
        self.net = net
        self.net_condition = net_condition
        self.use_xformers = torch.cuda.is_available()

    def init_coordinates(self):
        self.coordinates = self._init_coordinates(self.cfm.coordinates)
        self.condition_coordinates = self._init_coordinates("Fourmomenta")
        if self.cfm.transforms_float64:
            self.coordinates.to(torch.float64)
            self.condition_coordinates.to(torch.float64)

    def get_masks(self, batch):
        _, _, gen_batch_idx, _ = embed_data_into_ga(
            batch.x_gen,
            batch.scalars_gen,
            batch.x_gen_ptr,
            self.ga_cfg,
        )
        _, _, det_batch_idx, _ = embed_data_into_ga(
            batch.x_det,
            batch.scalars_det,
            batch.x_det_ptr,
            self.ga_cfg,
        )
        attention_mask = xformers_mask(gen_batch_idx, materialize=not self.use_xformers)
        condition_attention_mask = xformers_mask(
            det_batch_idx, materialize=not self.use_xformers
        )
        cross_attention_mask = xformers_mask(
            gen_batch_idx,
            det_batch_idx,
            materialize=not self.use_xformers,
        )
        return attention_mask, condition_attention_mask, cross_attention_mask

    def get_condition(self, batch, attention_mask):
        mv, s, _, _ = embed_data_into_ga(
            batch.x_det,
            batch.scalars_det,
            batch.x_det_ptr,
            self.ga_cfg,
        )
        mv = mv.unsqueeze(0)
        s = s.unsqueeze(0)
        attn_kwargs = {
            "attn_bias" if self.use_xformers else "attn_mask": attention_mask
        }
        condition_mv, condition_s = self.net_condition(mv, s, **attn_kwargs)
        return condition_mv, condition_s

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
        assert self.coordinates is not None

        jet_mask = torch.ones(xt.shape[0], dtype=torch.bool, device=xt.device)
        if self.cfm.add_jet:
            jet_mask[batch.x_gen_ptr[:-1]] = False
        gen_jets = torch.repeat_interleave(batch.jet_gen, batch.x_gen_ptr.diff(), dim=0)

        fourmomenta = torch.zeros_like(xt)
        fourmomenta[jet_mask] = self.coordinates.x_to_fourmomenta(
            xt[jet_mask],
            jet=gen_jets[jet_mask],
        )
        fourmomenta[~jet_mask] = jetmomenta_to_fourmomenta(xt[~jet_mask])

        condition_mv, condition_s = condition
        if self_condition is not None:
            scalars = torch.cat(
                [batch.scalars_gen, self.t_embedding(t), self_condition], dim=-1
            )
        else:
            scalars = torch.cat([batch.scalars_gen, self.t_embedding(t)], dim=-1)

        mv, s, _, spurions_mask = embed_data_into_ga(
            fourmomenta,
            scalars,
            batch.x_gen_ptr,
            self.ga_cfg,
        )

        mv_outputs, s_outputs = self.net(
            multivectors=mv.unsqueeze(0),
            multivectors_condition=condition_mv,
            scalars=s.unsqueeze(0),
            scalars_condition=condition_s,
            attn_kwargs={
                "attn_bias" if self.use_xformers else "attn_mask": attention_mask
            },
            crossattn_kwargs={
                "attn_bias" if self.use_xformers else "attn_mask": crossattention_mask
            },
        )
        mv_outputs = mv_outputs.squeeze(0)
        s_outputs = s_outputs.squeeze(0)

        v_fourmomenta = extract_vector(mv_outputs[spurions_mask]).squeeze(dim=-2)
        v_s = s[spurions_mask]

        LOGGER.info(
            f"Velocity output shape: {v_fourmomenta.shape}, {v_fourmomenta[jet_mask].shape}"
        )

        v_straight = torch.zeros_like(v_fourmomenta)
        v_straight[jet_mask] = self.coordinates.velocity_fourmomenta_to_x(
            v_fourmomenta[jet_mask],
            fourmomenta[jet_mask],
            jet=gen_jets[jet_mask],
        )[0]

        # Overwrite transformed velocities with scalar outputs
        # (this is specific to GATr to avoid large jacobians from from log-transforms)
        v_straight[..., self.scalar_dims] = v_s[..., self.scalar_dims]

        return v_straight
