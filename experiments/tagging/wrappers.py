import torch
from lloca.framesnet.frames import Frames
from lloca.framesnet.nonequi_frames import IdentityFrames
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.utils import (
    get_batch_from_ptr,
    get_ptr_from_batch,
)
from torch import nn
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.utils import scatter

from experiments.misc import get_attention_mask
from experiments.tagging.embedding import get_tagging_features
from lgatr import embed_vector, extract_scalar


class TaggerWrapper(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        framesnet,
        add_fourmomenta_backbone: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_fourmomenta_backbone = add_fourmomenta_backbone
        self.framesnet = framesnet
        self.trafo_fourmomenta = TensorRepsTransform(TensorReps("1x1n"))

    def init_standardization(self, fourmomenta, ptr, reduce_size=None):
        # framesnet equivectors edge_attr standardization (if applicable)
        if hasattr(self.framesnet, "equivectors") and hasattr(
            self.framesnet.equivectors, "init_standardization"
        ):
            fourmomenta_reduced = (
                fourmomenta[:reduce_size] if reduce_size is not None else fourmomenta
            )
            self.framesnet.equivectors.init_standardization(fourmomenta_reduced, ptr)

    def forward(self, embedding):
        # extract embedding
        fourmomenta_withspurions = embedding["fourmomenta"]
        scalars_withspurions = embedding["scalars"]
        global_tagging_features_withspurions = embedding["tagging_features"]
        batch_withspurions = embedding["batch"]
        is_spurion = embedding["is_spurion"]
        ptr_withspurions = embedding["ptr"]
        num_graphs = embedding["num_graphs"]
        nospurion_idxs = (~is_spurion).nonzero(as_tuple=False).squeeze(-1)

        # remove spurions from the data again and recompute attributes
        fourmomenta_nospurions = fourmomenta_withspurions.index_select(0, nospurion_idxs)
        scalars_nospurions = scalars_withspurions.index_select(0, nospurion_idxs)

        batch_nospurions = batch_withspurions.index_select(0, nospurion_idxs)
        ptr_nospurions = get_ptr_from_batch(batch_nospurions)
        B = ptr_nospurions.numel() - 1

        scalars_withspurions = torch.cat(
            [scalars_withspurions, global_tagging_features_withspurions], dim=-1
        )
        frames_spurions, tracker = self.framesnet(
            fourmomenta_withspurions,
            scalars_withspurions,
            ptr=ptr_withspurions,
            return_tracker=True,
            num_graphs=num_graphs,
        )
        matrices = frames_spurions.matrices.index_select(0, nospurion_idxs)
        frames_nospurions = Frames(
            matrices,
            is_global=frames_spurions.is_global,
            det=frames_spurions.det.index_select(0, nospurion_idxs),
            inv=frames_spurions.inv.index_select(0, nospurion_idxs),
            is_identity=frames_spurions.is_identity,
            device=frames_spurions.device,
            dtype=frames_spurions.dtype,
            shape=matrices.shape,
        )

        # transform features into local frames
        fourmomenta_local_nospurions = self.trafo_fourmomenta(
            fourmomenta_nospurions, frames_nospurions
        )
        jet_nospurions = scatter(
            fourmomenta_nospurions,
            index=batch_nospurions,
            dim=0,
            reduce="sum",
            dim_size=B,
        ).index_select(0, batch_nospurions)
        jet_local_nospurions = self.trafo_fourmomenta(jet_nospurions, frames_nospurions)
        local_tagging_features_nospurions = get_tagging_features(
            fourmomenta_local_nospurions,
            jet_local_nospurions,
            tagging_features="all",
        )

        features_local_nospurions = torch.cat(
            [scalars_nospurions, local_tagging_features_nospurions], dim=-1
        )
        if self.add_fourmomenta_backbone:
            features_local_nospurions = torch.cat(
                [features_local_nospurions, fourmomenta_local_nospurions], dim=-1
            )

        # change dtype (see embedding.py fourmomenta_float64 option)
        features_local_nospurions = features_local_nospurions.to(scalars_nospurions.dtype)
        frames_nospurions.to(scalars_nospurions.dtype)

        return (
            features_local_nospurions,
            fourmomenta_local_nospurions,
            frames_nospurions,
            ptr_nospurions,
            batch_nospurions,
            tracker,
        )


class AggregatedTaggerWrapper(TaggerWrapper):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aggregator = MeanAggregation()

    def extract_score(self, features, ptr):
        B = ptr.numel() - 1
        score = self.aggregator(features, ptr=ptr, dim_size=B)
        return score


class TransformerWrapper(AggregatedTaggerWrapper):
    def __init__(
        self,
        net,
        *args,
        use_amp=False,
        attention_backend="xformers",
        mean_aggregation=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_amp = use_amp
        self.attention_backend = attention_backend
        self.mean_aggregation = mean_aggregation
        self.net = net(in_channels=self.in_channels, out_channels=self.out_channels)

        if attention_backend == "flex":
            compile_flex_attention(package_name="lloca")

    def forward(self, embedding):
        # precompute attention mask to avoid cudaStreamSynchronize
        # from .tolist() in get_xformers_attention_mask
        batch_withspurions = embedding["batch"]
        is_spurion = embedding["is_spurion"]
        nospurion_idxs = (~is_spurion).nonzero(as_tuple=False).squeeze(-1)
        batch_nospurions = batch_withspurions.index_select(0, nospurion_idxs)
        ptr_nospurions = get_ptr_from_batch(batch_nospurions)
        ptr, batch = ptr_nospurions, batch_nospurions
        if not self.mean_aggregation:
            batchsize = len(ptr) - 1
            ptr = ptr.clone()
            ptr[1:] = ptr[1:] + (torch.arange(batchsize, device=ptr.device) + 1)
            batch = get_batch_from_ptr(ptr)
        mask_kwarg = get_attention_mask(
            batch,
            dtype=embedding["scalars"].dtype,
            attention_backend=self.attention_backend,
        )

        (
            features_local,
            _,
            frames,
            ptr,
            batch,
            tracker,
        ) = super().forward(embedding)

        # handle global token
        if self.mean_aggregation:
            is_global = None
        else:
            # append global tokens to batch, ptr, features_local and frames
            # and keep a is_global mask for later extraction
            batchsize = len(ptr) - 1
            global_idxs = ptr[:-1] + torch.arange(batchsize, device=batch.device)
            is_global = torch.zeros(
                features_local.shape[0] + batchsize,
                dtype=torch.bool,
                device=ptr.device,
            )
            is_global[global_idxs] = True
            features_local_buffer = features_local.clone()
            features_local = torch.zeros(
                is_global.shape[0],
                *features_local.shape[1:],
                dtype=features_local.dtype,
                device=features_local.device,
            )
            features_local[~is_global] = features_local_buffer
            is_global_channel = torch.zeros(
                features_local.shape[0],
                1,
                dtype=features_local.dtype,
                device=features_local.device,
            )
            is_global_channel[is_global] = 1
            features_local = torch.cat((features_local, is_global_channel), dim=-1)

            # global token frames are identity
            matrices_new = (
                torch.eye(4, device=frames.device, dtype=frames.dtype)
                .unsqueeze(0)
                .expand(is_global.shape[0], -1, -1)
            ).clone()
            matrices_new[~is_global] = frames.matrices
            det_new = torch.ones(
                is_global.shape[0], device=frames.device, dtype=frames.dtype
            ).clone()
            det_new[~is_global] = frames.det
            inv_new = (
                torch.eye(4, device=frames.device, dtype=frames.dtype)
                .unsqueeze(0)
                .expand(is_global.shape[0], -1, -1)
            ).clone()
            inv_new[~is_global] = frames.inv
            frames = Frames(
                matrices_new,
                is_global=frames.is_global,
                det=det_new,
                inv=inv_new,
                is_identity=frames.is_identity,
                device=frames.device,
                dtype=frames.dtype,
                shape=matrices_new.shape,
            )

            ptr[1:] = ptr[1:] + (torch.arange(batchsize, device=ptr.device) + 1)
            batch = get_batch_from_ptr(ptr)

        # add artificial batch dimension
        features_local = features_local.unsqueeze(0)
        frames = frames.reshape(1, *frames.shape)

        # LOGGER.info(f"Transformer input: {features_local}, shape {features_local.shape}")

        # network
        with torch.autocast("cuda", enabled=self.use_amp):
            outputs = self.net(inputs=features_local, frames=frames, **mask_kwarg)

        # LOGGER.info(f"Transformer output: {outputs}")

        # aggregation
        outputs = outputs[0, ...]
        if self.mean_aggregation:
            score = self.extract_score(outputs, ptr)
        else:
            score = outputs[is_global]

        # LOGGER.info(f"Transformer score: {score}")

        return score, tracker, frames


class ConditionalTransformerWrapper(nn.Module):
    def __init__(
        self,
        net,
        net_condition,
        in_channels,
        out_channels,
        *args,
        use_amp=False,
        attention_backend="xformers",
        mean_aggregation=True,
        framesnet=None,
        **kwargs,
    ):
        super().__init__()
        self.use_amp = use_amp
        self.attention_backend = attention_backend
        self.mean_aggregation = mean_aggregation
        self.net = net(in_channels=in_channels, out_channels=out_channels)
        self.net_condition = net_condition(in_channels=in_channels)
        self.framesnet = framesnet if framesnet is not None else IdentityFrames()

        if attention_backend == "flex":
            compile_flex_attention(package_name="lloca")

    def get_masks(self, batch, condition_batch, dtype):
        mask_kwarg = get_attention_mask(
            batch,
            dtype=dtype,
            attention_backend=self.attention_backend,
        )
        condition_mask_kwarg = get_attention_mask(
            condition_batch,
            dtype=dtype,
            attention_backend=self.attention_backend,
        )
        cross_mask_kwarg = get_attention_mask(
            batch,
            condition_batch=condition_batch,
            dtype=dtype,
            attention_backend=self.attention_backend,
        )
        return mask_kwarg, condition_mask_kwarg, cross_mask_kwarg

    def forward(self, embedding):
        input = torch.cat(
            [
                embedding["gen"]["fourmomenta"],
                embedding["gen"]["scalars"],
                embedding["gen"]["tagging_features"],
            ],
            dim=-1,
        )
        condition_input = torch.cat(
            [
                embedding["det"]["fourmomenta"],
                embedding["det"]["scalars"],
                embedding["det"]["tagging_features"],
            ],
            dim=-1,
        )
        gen_batch = embedding["gen"]["batch"]
        det_batch = embedding["det"]["batch"]
        mask_kwarg, condition_mask_kwarg, cross_mask_kwarg = self.get_masks(
            gen_batch,
            det_batch,
            dtype=input.dtype,
        )
        with torch.autocast("cuda", enabled=self.use_amp):
            condition = self.net_condition(
                inputs=condition_input.unsqueeze(0), **condition_mask_kwarg
            )
            outputs = self.net(
                x=input.unsqueeze(0),
                processed_condition=condition,
                attn_kwargs=mask_kwarg,
                crossattn_kwargs=cross_mask_kwarg,
            )

        score = scatter(outputs.squeeze(0), gen_batch, dim=0, reduce="mean")

        return score, {}, None


class LGATrWrapper(nn.Module):
    def __init__(
        self,
        net,
        framesnet,
        out_channels,
        mean_aggregation=False,
        use_amp=False,
        attention_backend="xformers",
    ):
        super().__init__()
        self.use_amp = use_amp
        self.attention_backend = attention_backend
        self.net = net(out_mv_channels=out_channels)
        self.aggregator = MeanAggregation() if mean_aggregation else None

        self.framesnet = framesnet  # not actually used
        assert isinstance(framesnet, IdentityFrames)

        if attention_backend == "flex":
            compile_flex_attention(package_name="lgatr")

    def forward(self, embedding):
        # extract embedding (includes spurions)
        fourmomenta = embedding["fourmomenta"]
        scalars = torch.cat([embedding["scalars"], embedding["tagging_features"]], dim=-1)
        batch = embedding["batch"]
        ptr = embedding["ptr"]
        is_spurion = embedding["is_spurion"]

        # rescale fourmomenta (but not the spurions)
        fourmomenta[~is_spurion] = fourmomenta[~is_spurion] / 20

        # handle global token
        if self.aggregator is None:
            batchsize = len(ptr) - 1
            global_idxs = ptr[:-1] + torch.arange(batchsize, device=batch.device)
            is_global = torch.zeros(
                fourmomenta.shape[0] + batchsize,
                dtype=torch.bool,
                device=ptr.device,
            )
            is_global[global_idxs] = True
            fourmomenta_buffer = fourmomenta.clone()
            fourmomenta = torch.zeros(
                is_global.shape[0],
                *fourmomenta.shape[1:],
                dtype=fourmomenta.dtype,
                device=fourmomenta.device,
            )
            fourmomenta[~is_global] = fourmomenta_buffer
            scalars_buffer = scalars.clone()
            scalars = torch.zeros(
                fourmomenta.shape[0],
                scalars.shape[1] + 1,
                dtype=scalars.dtype,
                device=scalars.device,
            )
            token_idx = torch.nn.functional.one_hot(torch.arange(1, device=scalars.device))
            token_idx = token_idx.repeat(batchsize, 1)
            scalars[~is_global] = torch.cat(
                (
                    scalars_buffer,
                    torch.zeros(
                        scalars_buffer.shape[0],
                        token_idx.shape[1],
                        dtype=scalars.dtype,
                        device=scalars.device,
                    ),
                ),
                dim=-1,
            )
            scalars[is_global] = torch.cat(
                (
                    torch.zeros(
                        token_idx.shape[0],
                        scalars_buffer.shape[1],
                        dtype=scalars.dtype,
                        device=scalars.device,
                    ),
                    token_idx,
                ),
                dim=-1,
            )
            ptr[1:] = ptr[1:] + (torch.arange(batchsize, device=ptr.device) + 1)
            batch = get_batch_from_ptr(ptr)
        else:
            is_global = None

        fourmomenta = fourmomenta.unsqueeze(0).to(scalars.dtype)
        scalars = scalars.unsqueeze(0)

        mask_kwarg = get_attention_mask(
            batch,
            dtype=scalars.dtype,
            attention_backend=self.attention_backend,
        )

        mv = embed_vector(fourmomenta).unsqueeze(-2)
        s = scalars if scalars.shape[-1] > 0 else None

        with torch.autocast("cuda", enabled=self.use_amp):
            mv_outputs, _ = self.net(mv, s, **mask_kwarg)
        out = extract_scalar(mv_outputs)[0, :, :, 0]

        if self.aggregator is not None:
            B = ptr.numel() - 1
            logits = self.aggregator(out, index=batch, dim_size=B)
        else:
            logits = out[is_global]
        return logits, {}, None


class LGATrSlimWrapper(nn.Module):
    def __init__(
        self,
        net,
        framesnet,
        out_channels,
        mean_aggregation=False,
        attention_backend="xformers",
        use_amp=False,
    ):
        super().__init__()
        self.use_amp = use_amp
        self.attention_backend = attention_backend
        self.net = net(out_s_channels=out_channels)
        self.aggregator = MeanAggregation() if mean_aggregation else None
        self.framesnet = framesnet  # not actually used
        assert isinstance(framesnet, IdentityFrames)

        if attention_backend == "flex":
            compile_flex_attention(package_name="lgatr")

    def forward(self, embedding):
        # extract embedding (includes spurions)
        fourmomenta = embedding["fourmomenta"]
        scalars = torch.cat([embedding["scalars"], embedding["tagging_features"]], dim=-1)
        batch = embedding["batch"]
        ptr = embedding["ptr"]
        is_spurion = embedding["is_spurion"]

        # rescale fourmomenta (but not the spurions)
        fourmomenta[~is_spurion] = fourmomenta[~is_spurion] / 20

        # handle global token
        if self.aggregator is None:
            batchsize = len(ptr) - 1
            global_idxs = ptr[:-1] + torch.arange(batchsize, device=batch.device)
            is_global = torch.zeros(
                fourmomenta.shape[0] + batchsize,
                dtype=torch.bool,
                device=ptr.device,
            )
            is_global[global_idxs] = True
            fourmomenta_buffer = fourmomenta.clone()
            fourmomenta = torch.zeros(
                is_global.shape[0],
                *fourmomenta.shape[1:],
                dtype=fourmomenta.dtype,
                device=fourmomenta.device,
            )
            fourmomenta[~is_global] = fourmomenta_buffer
            scalars_buffer = scalars.clone()
            scalars = torch.zeros(
                fourmomenta.shape[0],
                scalars.shape[1] + 1,
                dtype=scalars.dtype,
                device=scalars.device,
            )
            token_idx = torch.nn.functional.one_hot(torch.arange(1, device=scalars.device))
            token_idx = token_idx.repeat(batchsize, 1)
            scalars[~is_global] = torch.cat(
                (
                    scalars_buffer,
                    torch.zeros(
                        scalars_buffer.shape[0],
                        token_idx.shape[1],
                        dtype=scalars.dtype,
                        device=scalars.device,
                    ),
                ),
                dim=-1,
            )
            scalars[is_global] = torch.cat(
                (
                    torch.zeros(
                        token_idx.shape[0],
                        scalars_buffer.shape[1],
                        dtype=scalars.dtype,
                        device=scalars.device,
                    ),
                    token_idx,
                ),
                dim=-1,
            )
            ptr[1:] = ptr[1:] + (torch.arange(batchsize, device=ptr.device) + 1)
            batch = get_batch_from_ptr(ptr)
        else:
            is_global = None

        fourmomenta = fourmomenta.unsqueeze(0).to(scalars.dtype)
        scalars = scalars.unsqueeze(0)

        mask_kwarg = get_attention_mask(
            batch,
            dtype=fourmomenta.dtype,
            attention_backend=self.attention_backend,
        )

        v = fourmomenta.unsqueeze(-2)
        s = scalars

        with torch.autocast("cuda", enabled=self.use_amp):
            _, out_s = self.net(v, s, **mask_kwarg)
        out = out_s[0, :, :]

        if self.aggregator is not None:
            logits = self.aggregator(out, index=batch)
        else:
            logits = out[is_global]
        return logits, {}, None


def compile_flex_attention(package_name="lgatr"):
    """Run torch.compile on the flex_attention function.

    However, as of today (Dec 2025, pytorch 2.9.0), torch.compile + flex_attention
    for variable-length sequences only works in a few cases:
    - CPU: Forward pass is supported, but backward pass not (https://github.com/pytorch/pytorch/issues/169224)
      To still let the code run through for tests, we skip torch.compile on CPU.
      This way the code runs through, but is super slow because it materializes the attention matrix.
      Note that we use essentially the same approach for xformers, where we fall back to default torch attention on CPU.
      On the plus side, flex_attention supports arbitrary head_dim if torch.compile is not used.
    - GPU: The docs say that only head dimensions being powers of 2 are supported.
      However, on my system only head_dim=2**n with n>=4 works, i.e. head_dim=16,32,...
      Setting head_dim=2,4,8 gives cryptic errors.
      Moreover, transformers with flex_attention are still significantly slower than
      transformers with xformers attention in our implementation.
    """
    if package_name == "lgatr":
        import lgatr.primitives.attention_backends.flex as flex
    elif package_name == "lloca":
        import lloca.backbone.attention_backends.flex as flex
    else:
        raise ValueError(f"Unknown package {package_name}")

    if torch.cuda.is_available():
        # max-autotune strongly recommended for flex-attention with variable-length sequences,
        # see https://pytorch.org/blog/flexattention-for-inference/
        flex.attention = torch.compile(
            flex.attention,
            dynamic=True,
        )
