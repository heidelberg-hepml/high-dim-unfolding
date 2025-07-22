import torch
import numpy as np
from lgatr.interface import extract_vector, embed_vector

from experiments.utils import xformers_mask
from experiments.kinematics.cfm import EventCFM
from experiments.embedding import embed_data_into_ga
from experiments.coordinates import jetmomenta_to_fourmomenta
from experiments.logger import LOGGER


class ConditionalTransformerJetTokensCFM(EventCFM):
    """
    CFM model for jet tokens where each jet's 4 coordinates (E, px, py, pz)
    are treated as 4 separate tokens of dimension 1.

    Uses batch.jet_gen and batch.jet_det instead of batch.x_gen and batch.x_det.
    Simply flattens jets (batch_size, 4) to tokens (batch_size*4, 1).
    """

    def __init__(
        self,
        net,
        net_condition,
        cfm,
        odeint,
    ):
        super().__init__(
            cfm,
            odeint,
        )
        self.net = net
        self.net_condition = net_condition
        self.use_xformers = torch.cuda.is_available()

    def _jet_to_tokens(self, jets):
        """
        Convert jets (batch_size, 4) to tokens (batch_size*4, 1)
        """
        batch_size = jets.shape[0]
        return jets.view(batch_size * 4, 1)

    def _tokens_to_jet(self, tokens):
        """
        Convert tokens (batch_size*4, 1) to jets (batch_size, 4)
        """
        return tokens.view(-1, 4)

    def _create_token_batch_indices(self, batch_size):
        """
        Create batch indices for tokens where each event has 4 consecutive tokens
        """
        # Each event contributes 4 tokens: [0,0,0,0,1,1,1,1,2,2,2,2,...]
        return torch.repeat_interleave(torch.arange(batch_size), 4)

    def get_masks(self, batch):
        batch_size = batch.jet_gen.shape[0]

        # Create token batch indices
        token_gen_batch = self._create_token_batch_indices(batch_size).to(
            batch.jet_gen.device
        )
        token_det_batch = self._create_token_batch_indices(batch_size).to(
            batch.jet_det.device
        )

        attention_mask = xformers_mask(
            token_gen_batch, materialize=not self.use_xformers
        )
        condition_attention_mask = xformers_mask(
            token_det_batch, materialize=not self.use_xformers
        )
        cross_attention_mask = xformers_mask(
            token_gen_batch,
            token_det_batch,
            materialize=not self.use_xformers,
        )
        return attention_mask, condition_attention_mask, cross_attention_mask

    def get_condition(self, batch, attention_mask):
        # Convert det jets to tokens
        det_tokens = self._jet_to_tokens(batch.jet_det)  # (batch_size*4, 1)

        # Create scalars for each token (repeat scalars 4 times per event)
        batch_size = batch.jet_det.shape[0]
        if hasattr(batch, "scalars_det"):
            scalars_det_tokens = batch.scalars_det.repeat_interleave(
                4, dim=0
            )  # (batch_size*4, scalar_dim)
        else:
            scalars_det_tokens = torch.zeros(
                batch_size * 4, 0, device=det_tokens.device
            )

        input = torch.cat([det_tokens, scalars_det_tokens], dim=-1)
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
        # xt should be in token format (batch_size*4, 1)
        batch_size = batch.jet_gen.shape[0]

        # Create scalars for each token
        if hasattr(batch, "jet_scalars_gen"):
            scalars_gen_tokens = batch.jet_scalars_gen.repeat_interleave(4, dim=0)
        else:
            scalars_gen_tokens = torch.zeros(batch_size * 4, 0, device=xt.device)

        # Time embedding for each token
        t_embed = self.t_embedding(t).repeat_interleave(4, dim=0)

        if self_condition is not None:
            self_condition_tokens = self_condition.repeat_interleave(4, dim=0)
            input = torch.cat(
                [xt, scalars_gen_tokens, t_embed, self_condition_tokens], dim=-1
            )
        else:
            input = torch.cat([xt, scalars_gen_tokens, t_embed], dim=-1)

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

    def sample(self, batch, device, dtype):
        """
        Override sample to work with jets as tokens
        """
        batch_size = batch.jet_det.shape[0]

        # Convert jets to tokens for the generative process
        jet_gen_tokens = self._jet_to_tokens(batch.jet_gen)  # (batch_size*4, 1)
        jet_det_tokens = self._jet_to_tokens(batch.jet_det)  # (batch_size*4, 1)

        # Create a modified batch with token structure
        token_batch = batch.clone()

        # Override the x_gen and x_det with jet tokens
        token_batch.x_gen = self.coordinates.fourmomenta_to_x(
            jet_gen_tokens, jet=batch.jet_gen
        )
        token_batch.x_det = self.condition_coordinates.fourmomenta_to_x(
            jet_det_tokens, jet=batch.jet_det
        )

        # Create token batch indices and pointers
        token_batch.x_gen_batch = self._create_token_batch_indices(batch_size).to(
            device
        )
        token_batch.x_det_batch = self._create_token_batch_indices(batch_size).to(
            device
        )
        token_batch.x_gen_ptr = torch.arange(0, batch_size * 4 + 1, 4, device=device)
        token_batch.x_det_ptr = torch.arange(0, batch_size * 4 + 1, 4, device=device)

        # Use parent's sample method with token batch
        sample_token_batch, sample = super().sample(token_batch, device, dtype)

        # Convert sampled tokens back to jets
        sample_gen_tokens = self.coordinates.x_to_fourmomenta(
            sample_token_batch.x_gen, jet=batch.jet_gen
        )
        sample_det_tokens = self.condition_coordinates.x_to_fourmomenta(
            sample_token_batch.x_det, jet=batch.jet_det
        )

        # Convert back to jet format
        sample_batch = batch.clone()
        sample_batch.jet_gen = self._tokens_to_jet(sample_gen_tokens)
        sample_batch.jet_det = self._tokens_to_jet(sample_det_tokens)

        return sample_batch, sample


class ConditionalLGATrJetTokensCFM(EventCFM):
    """
    GATr velocity network for jet tokens where each coordinate is a separate token.
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

    def _jet_to_tokens(self, jets):
        """Convert jets (batch_size, 4) to tokens (batch_size*4, 1)"""
        batch_size = jets.shape[0]
        return jets.view(batch_size * 4, 1)

    def _tokens_to_jet(self, tokens):
        """Convert tokens (batch_size*4, 1) to jets (batch_size, 4)"""
        return tokens.view(-1, 4)

    def _create_token_pointers(self, batch_size):
        """Create pointer arrays for the 4 tokens per event"""
        return torch.arange(0, (batch_size + 1) * 4, 4)

    def get_masks(self, batch):
        batch_size = batch.jet_gen.shape[0]

        # Convert jets to tokens for mask computation
        gen_jet_tokens = self._jet_to_tokens(batch.jet_gen)
        det_jet_tokens = self._jet_to_tokens(batch.jet_det)

        # Create pointers for GA embedding
        gen_ptrs = self._create_token_pointers(batch_size).to(batch.jet_gen.device)
        det_ptrs = self._create_token_pointers(batch_size).to(batch.jet_det.device)

        # Create scalars for tokens
        if hasattr(batch, "scalars_gen"):
            scalars_gen = batch.scalars_gen.repeat_interleave(4, dim=0)
        else:
            scalars_gen = torch.zeros(batch_size * 4, 0, device=batch.jet_gen.device)

        if hasattr(batch, "scalars_det"):
            scalars_det = batch.scalars_det.repeat_interleave(4, dim=0)
        else:
            scalars_det = torch.zeros(batch_size * 4, 0, device=batch.jet_det.device)

        # For GA embedding, we need to treat each token as a 4-vector
        # We'll expand each token to a full 4-vector for GA processing
        gen_fourmomenta = torch.zeros(batch_size * 4, 4, device=batch.jet_gen.device)
        gen_fourmomenta[:, 0] = (
            gen_jet_tokens.squeeze()
        )  # Put coordinate value in energy component

        det_fourmomenta = torch.zeros(batch_size * 4, 4, device=batch.jet_det.device)
        det_fourmomenta[:, 0] = det_jet_tokens.squeeze()

        _, _, gen_batch_idx, _ = embed_data_into_ga(
            gen_fourmomenta,
            scalars_gen,
            gen_ptrs,
            None,
        )
        _, _, det_batch_idx, _ = embed_data_into_ga(
            det_fourmomenta,
            scalars_det,
            det_ptrs,
            None,
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
        batch_size = batch.jet_det.shape[0]
        det_jet_tokens = self._jet_to_tokens(batch.jet_det)
        det_ptrs = self._create_token_pointers(batch_size).to(batch.jet_det.device)

        if hasattr(batch, "scalars_det"):
            scalars_det = batch.scalars_det.repeat_interleave(4, dim=0)
        else:
            scalars_det = torch.zeros(batch_size * 4, 0, device=batch.jet_det.device)

        # Create fourmomenta representation for GA embedding
        det_fourmomenta = torch.zeros(batch_size * 4, 4, device=batch.jet_det.device)
        det_fourmomenta[:, 0] = det_jet_tokens.squeeze()

        mv, s, _, _ = embed_data_into_ga(
            det_fourmomenta,
            scalars_det,
            det_ptrs,
            None,
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

        batch_size = batch.jet_gen.shape[0]
        gen_ptrs = self._create_token_pointers(batch_size).to(xt.device)

        # xt contains the coordinate tokens (batch_size*4, 1)
        if hasattr(batch, "scalars_gen"):
            scalars_gen = batch.scalars_gen.repeat_interleave(4, dim=0)
        else:
            scalars_gen = torch.zeros(batch_size * 4, 0, device=xt.device)

        condition_mv, condition_s = condition
        if self_condition is not None:
            self_condition_tokens = self_condition.repeat_interleave(4, dim=0)
            scalars = torch.cat(
                [
                    scalars_gen,
                    self.t_embedding(t).repeat_interleave(4, dim=0),
                    self_condition_tokens,
                ],
                dim=-1,
            )
        else:
            scalars = torch.cat(
                [scalars_gen, self.t_embedding(t).repeat_interleave(4, dim=0)], dim=-1
            )

        # Convert coordinate tokens to fourmomenta for GA processing
        fourmomenta = torch.zeros(batch_size * 4, 4, device=xt.device)
        fourmomenta[:, 0] = xt.squeeze()  # Put coordinate values in energy component

        # Embed into GA space
        mv, s, _, spurions_mask = embed_data_into_ga(
            fourmomenta,
            scalars,
            gen_ptrs,
            None,
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

        # Extract velocity from GA space
        v_fourmomenta = extract_vector(mv_outputs[spurions_mask]).squeeze(dim=-2)
        v_s = s_outputs[spurions_mask]

        # Convert velocity back to coordinate tokens (extract energy component)
        v_tokens = v_fourmomenta[:, 0:1]  # (batch_size*4, 1)

        # Overwrite with scalar outputs for specific dimensions
        if len(self.scalar_dims) > 0:
            for dim in self.scalar_dims:
                coord_indices = torch.arange(dim, batch_size * 4, 4, device=xt.device)
                if len(coord_indices) > 0 and coord_indices.max() < v_tokens.shape[0]:
                    if dim < v_s.shape[1]:
                        v_tokens[coord_indices, 0] = v_s[coord_indices, dim]

        return v_tokens

    def sample(self, batch, device, dtype):
        """
        Override sample to work with jets as tokens
        """
        batch_size = batch.jet_det.shape[0]

        # Convert jets to tokens for the generative process
        jet_gen_tokens = self._jet_to_tokens(batch.jet_gen)
        jet_det_tokens = self._jet_to_tokens(batch.jet_det)

        # Create a modified batch with token structure
        token_batch = batch.clone()

        # Override the x_gen and x_det with jet tokens
        token_batch.x_gen = self.coordinates.fourmomenta_to_x(
            jet_gen_tokens, jet=batch.jet_gen
        )
        token_batch.x_det = self.condition_coordinates.fourmomenta_to_x(
            jet_det_tokens, jet=batch.jet_det
        )

        # Create token pointers
        token_batch.x_gen_ptr = self._create_token_pointers(batch_size).to(device)
        token_batch.x_det_ptr = self._create_token_pointers(batch_size).to(device)

        # Create scalars for tokens
        if hasattr(batch, "scalars_gen"):
            token_batch.scalars_gen = batch.scalars_gen.repeat_interleave(4, dim=0)
        if hasattr(batch, "scalars_det"):
            token_batch.scalars_det = batch.scalars_det.repeat_interleave(4, dim=0)

        # Use parent's sample method with token batch
        sample_token_batch, sample = super().sample(token_batch, device, dtype)

        # Convert sampled tokens back to jets
        sample_gen_tokens = self.coordinates.x_to_fourmomenta(
            sample_token_batch.x_gen, jet=batch.jet_gen
        )
        sample_det_tokens = self.condition_coordinates.x_to_fourmomenta(
            sample_token_batch.x_det, jet=batch.jet_det
        )

        # Convert back to jet format
        sample_batch = batch.clone()
        sample_batch.jet_gen = self._tokens_to_jet(sample_gen_tokens)
        sample_batch.jet_det = self._tokens_to_jet(sample_det_tokens)

        return sample_batch, sample
