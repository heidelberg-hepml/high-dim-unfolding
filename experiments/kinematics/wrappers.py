import torch
import numpy as np
from lgatr.interface import extract_vector

from experiments.utils import xformers_mask
from experiments.kinematics.cfm import EventCFM
from experiments.embedding import embed_data_into_ga


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
        force_xformers=True,
    ):
        # See GATrCFM.__init__ for documentation
        super().__init__(
            cfm,
            odeint,
        )
        self.net = net
        self.net_condition = net_condition
        self.force_xformers = force_xformers

    def get_masks(self, batch):
        attention_mask = xformers_mask(
            batch.x_gen_batch, materialize=not self.force_xformers
        )
        condition_attention_mask = xformers_mask(
            batch.x_det_batch, materialize=not self.force_xformers
        )
        cross_attention_mask = xformers_mask(
            batch.x_gen_batch,
            batch.x_det_batch,
            materialize=not self.force_xformers,
        )
        return attention_mask, condition_attention_mask, cross_attention_mask

    def get_condition(self, batch, attention_mask):
        input = torch.cat([batch.x_det, batch.scalars_det], dim=-1)
        return self.net_condition(input.unsqueeze(0), attention_mask=attention_mask)

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
            attention_mask=attention_mask,
            crossattention_mask=crossattention_mask,
        ).squeeze(0)
        return self.geometry._handle_periodic(vp)


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
        force_xformers=True,
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
        self.force_xformers = force_xformers

    def get_masks(self, batch):
        gen_embedding = embed_data_into_ga(
            batch.x_gen,
            batch.scalars_gen,
            batch.x_gen_ptr,
            self.ga_cfg,
        )
        det_embedding = embed_data_into_ga(
            batch.x_det,
            batch.scalars_det,
            batch.x_det_ptr,
            self.ga_cfg,
        )
        attention_mask = xformers_mask(
            gen_embedding["batch"], materialize=not self.force_xformers
        )
        condition_attention_mask = xformers_mask(
            det_embedding["batch"], materialize=not self.force_xformers
        )
        cross_attention_mask = xformers_mask(
            gen_embedding["batch"],
            det_embedding["batch"],
            materialize=not self.force_xformers,
        )
        return attention_mask, condition_attention_mask, cross_attention_mask

    def get_condition(self, batch, attention_mask):
        embedding = embed_data_into_ga(
            batch.x_det,
            batch.scalars_det,
            batch.x_det_ptr,
            self.ga_cfg,
        )
        mv = embedding["mv"].unsqueeze(0)
        s = embedding["s"].unsqueeze(0)
        condition_mv, condition_s = self.net_condition(mv, s, attn_bias=attention_mask)
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

        fourmomenta = self.coordinates.x_to_fourmomenta(
            xt,
            jet=torch.repeat_interleave(batch.jet_gen, batch.x_gen_ptr.diff(), dim=0),
        )
        condition_mv, condition_s = condition
        if self_condition is not None:
            scalars = torch.cat(
                [batch.scalars_gen, self.t_embedding(t), self_condition], dim=-1
            )
        else:
            scalars = torch.cat([batch.scalars_gen, self.t_embedding(t)], dim=-1)

        embedding = embed_data_into_ga(
            fourmomenta,
            scalars,
            batch.x_gen_ptr,
            # self.ga_cfg,
        )
        mv = embedding["mv"].unsqueeze(0)
        s = embedding["s"].unsqueeze(0)
        spurions_mask = embedding["mask"]

        mv_outputs, s_outputs = self.net(
            multivectors=mv,
            multivectors_condition=condition_mv,
            scalars=s,
            scalars_condition=condition_s,
            attn_kwargs={"attn_bias": attention_mask},
            crossattn_kwargs={"attn_bias": crossattention_mask},
        )
        mv_outputs = mv_outputs.squeeze(0)
        s_outputs = s_outputs.squeeze(0)

        v_fourmomenta, v_s = self.extract_from_ga(mv_outputs, s_outputs, spurions_mask)

        v_straight = self.coordinates.velocity_fourmomenta_to_x(
            v_fourmomenta,
            fourmomenta,
            jet=torch.repeat_interleave(batch.jet_gen, batch.x_gen_ptr.diff(), dim=0),
        )[0]

        # Overwrite transformed velocities with scalar outputs
        # (this is specific to GATr to avoid large jacobians from from log-transforms)
        v_straight[..., self.scalar_dims] = v_s[..., self.scalar_dims]

        return v_straight

    def extract_from_ga(self, mv, s, mask):
        v = extract_vector(mv[mask]).squeeze(dim=-2)
        return v, s[mask]
