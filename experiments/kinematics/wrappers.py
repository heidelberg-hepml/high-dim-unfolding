import torch
import numpy as np
from lgatr.interface import extract_vector, embed_vector

from experiments.utils import xformers_mask
from experiments.kinematics.cfm import EventCFM, JetCFM
from experiments.embedding import embed_data_into_ga
from experiments.coordinates import jetmomenta_to_fourmomenta
from experiments.dataset import positional_encoding
from experiments.logger import LOGGER
from experiments.kinematics.plots import plot_kinematics


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
        self.ga_cfg = GA_config if self.cfm.gac else None
        self.net = net
        self.net_condition = net_condition
        self.use_xformers = torch.cuda.is_available()

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
        x = batch.x_det
        constituents_mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        if self.cfm.add_jet:
            constituents_mask[batch.x_det_ptr[:-1]] = False
            ptr = batch.x_det_ptr - torch.arange(
                batch.x_det_ptr.shape[0], device=batch.x_det_ptr.device
            )
        else:
            ptr = batch.x_det_ptr

        det_jets = self.condition_jet_coordinates.x_to_fourmomenta(batch.jet_det)
        ext_det_jets = torch.repeat_interleave(det_jets, ptr.diff(), dim=0)

        fourmomenta = torch.zeros_like(x)
        fourmomenta[constituents_mask] = (
            self.condition_const_coordinates.x_to_fourmomenta(
                x[constituents_mask],
                jet=ext_det_jets,
                ptr=ptr,
            )
        )
        fourmomenta[~constituents_mask] = det_jets

        scalars = torch.cat([batch.scalars_det, x[..., [0, 3]]], dim=-1)

        mv, s, _, _ = embed_data_into_ga(
            batch.x_det,
            scalars,
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

        constituents_mask = torch.ones(xt.shape[0], dtype=torch.bool, device=xt.device)
        if self.cfm.add_jet:
            constituents_mask[batch.x_gen_ptr[:-1]] = False
            ptr = batch.x_gen_ptr - torch.arange(
                batch.x_gen_ptr.shape[0], device=batch.x_gen_ptr.device
            )
        else:
            ptr = batch.x_gen_ptr

        gen_jets = self.jet_coordinates.x_to_fourmomenta(batch.jet_gen)
        ext_gen_jets = torch.repeat_interleave(gen_jets, ptr.diff(), dim=0)

        fourmomenta = torch.zeros_like(xt)
        fourmomenta[constituents_mask] = self.const_coordinates.x_to_fourmomenta(
            xt[constituents_mask],
            jet=ext_gen_jets,
            ptr=ptr,
        )

        fourmomenta[~constituents_mask] = gen_jets

        condition_mv, condition_s = condition
        if self_condition is not None:
            scalars = torch.cat(
                [
                    batch.scalars_gen,
                    self.t_embedding(t),
                    self_condition,
                    xt[..., 0],
                    xt[..., 3],
                ],
                dim=-1,
            )
        else:
            scalars = torch.cat(
                [batch.scalars_gen, self.t_embedding(t), xt[..., [0, 3]]], dim=-1
            )

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

        v_fourmomenta = extract_vector(mv_outputs[~spurions_mask]).squeeze(dim=-2)
        v_s = s_outputs[~spurions_mask]

        if not (torch.isfinite(v_fourmomenta).all() and torch.isfinite(v_s).all()):

            event_idx = batch.x_gen_batch[
                torch.nonzero(torch.isnan(v_fourmomenta).any(dim=-1))
            ]

            assert event_idx[0] == event_idx[-1]

            event_idx = event_idx[0].item()

            LOGGER.info(f"Event index: {event_idx}")
            LOGGER.info(
                f"time: {t[batch.x_gen_ptr[event_idx]]}, time embedding: {self.t_embedding(t[batch.x_gen_ptr[event_idx]]).squeeze()}"
            )
            LOGGER.info(
                f"x0: {batch.x_gen[batch.x_gen_ptr[event_idx]:batch.x_gen_ptr[event_idx + 1]]}"
            )

            LOGGER.info(
                f"xt: {xt[batch.x_gen_ptr[event_idx]:batch.x_gen_ptr[event_idx + 1]]}"
            )
            raise ValueError("Non-finite values found in velocity outputs")

        v_straight = torch.zeros_like(v_fourmomenta)
        v_straight[constituents_mask] = (
            self.const_coordinates.velocity_fourmomenta_to_x(
                v_fourmomenta[constituents_mask],
                fourmomenta[constituents_mask],
                jet=ext_gen_jets,
                ptr=ptr,
            )[0]
        )

        # Overwrite transformed velocities with scalar outputs
        # (this is specific to GATr to avoid large jacobians from from log-transforms)
        row_indices = torch.where(constituents_mask)[0].unsqueeze(-1)
        v_straight[row_indices, self.scalar_dims] = v_s[row_indices, self.scalar_dims]

        return v_straight


class JetConditionalTransformerCFM(JetCFM):
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
        if self.cfm.transpose:
            if self.cfm.pe_dim > 0:
                self.pe = positional_encoding(seq_length=4, pe_dim=self.cfm.pe_dim)
            else:
                self.pe = torch.empty(4, 0)

    def get_masks(self, batch):
        if self.cfm.transpose:
            gen_batch_idx = torch.repeat_interleave(
                torch.arange(batch.num_graphs, device=batch.jet_gen.device), 4
            )
            det_batch_idx = torch.repeat_interleave(
                torch.arange(batch.num_graphs, device=batch.jet_det.device), 4
            )
        else:
            gen_batch_idx = torch.arange(batch.num_graphs, device=batch.jet_gen.device)
            if self.cfm.add_constituents:
                det_batch_idx = batch.x_det_batch
            else:
                det_batch_idx = torch.arange(
                    batch.num_graphs, device=batch.x_det.device
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
        if self.cfm.transpose:
            input = torch.flatten(batch.jet_det).unsqueeze(-1)
            pe = self.pe.repeat(batch.jet_det.shape[0], 1).to(
                batch.jet_det.device, dtype=batch.jet_det.dtype
            )
            input = torch.cat(
                [input, torch.repeat_interleave(batch.jet_scalars_det, 4, dim=0), pe],
                dim=1,
            )
        elif self.cfm.add_constituents:
            input = torch.cat([batch.x_det, batch.scalars_det], dim=-1)
        else:
            input = torch.cat([batch.jet_det, batch.jet_scalars_det], dim=-1)
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

        if self.cfm.transpose:
            pe = self.pe.repeat(xt.shape[0], 1).to(xt.device, dtype=xt.dtype)
            scalars = torch.repeat_interleave(
                torch.cat([batch.jet_scalars_gen, self.t_embedding(t)], dim=-1),
                4,
                dim=0,
            )
            input = torch.cat(
                [xt.flatten().unsqueeze(-1), scalars, pe],
                dim=-1,
            )
        else:
            input = torch.cat([xt, batch.jet_scalars_gen, self.t_embedding(t)], dim=-1)
        if self_condition is not None:
            if self.cfm.transpose:
                input = torch.cat(
                    [input, self_condition.flatten().unsqueeze(-1)], dim=-1
                )
            else:
                input = torch.cat([input, self_condition], dim=-1)

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
        if self.cfm.transpose:
            vp = vp.reshape(xt.shape[0], 4)
        return vp


class JetConditionalLGATrCFM(JetCFM):
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
        self.ga_cfg = GA_config if self.cfm.gac else None
        self.net = net
        self.net_condition = net_condition
        self.use_xformers = torch.cuda.is_available()

    def get_masks(self, batch):
        _, _, gen_batch_idx, _ = embed_data_into_ga(
            batch.jet_gen,
            batch.jet_scalars_gen,
            torch.arange(batch.num_graphs + 1, device=batch.jet_gen.device),
            self.ga_cfg,
        )
        if self.cfm.add_constituents:
            _, _, det_batch_idx, _ = embed_data_into_ga(
                batch.x_det,
                batch.scalars_det,
                batch.x_det_ptr,
                self.ga_cfg,
            )
        else:
            _, _, det_batch_idx, _ = embed_data_into_ga(
                batch.jet_det,
                batch.jet_scalars_det,
                torch.arange(batch.num_graphs + 1, device=batch.jet_gen.device),
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

        if self.cfm.add_constituents:
            x = batch.x_det
            constituents_mask = torch.ones(
                x.shape[0], dtype=torch.bool, device=x.device
            )

            ptr = batch.x_det_ptr
            det_jets = self.condition_jet_coordinates.x_to_fourmomenta(batch.jet_det)
            ext_det_jets = torch.repeat_interleave(det_jets, ptr.diff(), dim=0)

            fourmomenta = torch.zeros_like(x)
            fourmomenta[constituents_mask] = (
                self.condition_const_coordinates.x_to_fourmomenta(
                    x[constituents_mask],
                    jet=ext_det_jets,
                    ptr=ptr,
                )
            )
            fourmomenta[~constituents_mask] = det_jets

            scalars = torch.cat([batch.scalars_det, x[..., [0, 3]]], dim=1)

        else:
            fourmomenta = self.condition_jet_coordinates.x_to_fourmomenta(batch.jet_det)
            ptr = torch.arange(batch.num_graphs + 1, device=batch.jet_det.device)
            scalars = torch.cat(
                [batch.jet_scalars_det, batch.jet_det[..., [0, 3]]],
                dim=1,
            )

        mv, s, _, _ = embed_data_into_ga(
            fourmomenta,
            scalars,
            ptr,
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

        fourmomenta = self.jet_coordinates.x_to_fourmomenta(
            xt,
        )

        condition_mv, condition_s = condition
        if self_condition is not None:
            scalars = torch.cat(
                [
                    batch.jet_scalars_gen,
                    self.t_embedding(t),
                    self_condition,
                    xt[..., 0],
                    xt[..., 3],
                ],
                dim=-1,
            )
        else:
            scalars = torch.cat(
                [batch.jet_scalars_gen, self.t_embedding(t), xt[..., [0, 3]]],
                dim=-1,
            )

        mv, s, _, spurions_mask = embed_data_into_ga(
            fourmomenta,
            scalars,
            torch.arange(batch.num_graphs + 1, device=xt.device),
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

        v_fourmomenta = extract_vector(mv_outputs[~spurions_mask]).squeeze(dim=-2)
        v_s = s_outputs[~spurions_mask]

        v_straight = self.jet_coordinates.velocity_fourmomenta_to_x(
            v_fourmomenta, fourmomenta
        )[0]

        # Overwrite transformed velocities with scalar outputs
        # (this is specific to GATr to avoid large jacobians from from log-transforms)
        v_straight[..., self.scalar_dims] = v_s[..., self.scalar_dims]

        return v_straight
