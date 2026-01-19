import math
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn

try:
    from torch.nn.attention.flex_attention import create_block_mask
except ModuleNotFoundError:
    pass
try:
    from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask, BlockDiagonalMask
except ModuleNotFoundError:
    pass


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, input_dim=1, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.weights = nn.Parameter(
            scale * torch.randn(input_dim, embed_dim // 2), requires_grad=False
        )

    def forward(self, t):
        projection = 2 * math.pi * torch.matmul(t, self.weights)
        embedding = torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)
        return embedding


# log(x) -> log(x+EPS1)
# in (invertible) preprocessing functions to avoid being close to log(0)
EPS1 = 1e-5

# generic numerical stability cutoff
EPS2 = 1e-10

# exp(x) -> exp(x.clamp(max=CUTOFF))
CUTOFF = 15


def ensure_angle(phi):
    return torch.remainder(phi + torch.pi, 2 * torch.pi) - torch.pi


def get_batch_from_ptr(ptr):
    ptr = ptr.to(torch.int)
    return (
        torch.arange(len(ptr) - 1, device=ptr.device)
        .repeat_interleave(
            ptr.diff(),
        )
        .to(torch.int64)
    )


def get_ptr_from_batch(batch):
    return (
        torch.cat([torch.tensor([0], device=batch.device), torch.bincount(batch)])
        .cumsum(dim=0)
        .to(torch.int64)
    )


def get_pt(fourmomenta):
    return torch.sqrt(fourmomenta[..., 1] ** 2 + fourmomenta[..., 2] ** 2)


def get_phi(fourmomenta):
    return ensure_angle(torch.arctan2(fourmomenta[..., 2], fourmomenta[..., 1]))


def get_eta(fourmomenta):
    p_abs = torch.sqrt(torch.sum(fourmomenta[..., 1:] ** 2, dim=-1))
    eta = stable_arctanh(fourmomenta[..., 3] / p_abs, eps=EPS2)
    return eta


def stable_arctanh(x, eps=EPS2):
    # implementation of arctanh that avoids log(0) issues
    return 0.5 * (torch.log((1 + x).clamp(min=eps)) - torch.log((1 - x).clamp(min=eps)))


def get_mass(fourmomenta, eps=EPS2):
    m2 = fourmomenta[..., 0] ** 2 - torch.sum(fourmomenta[..., 1:] ** 2, dim=-1)
    m2 = torch.abs(m2)
    m = torch.sqrt(m2.clamp(min=eps))
    return m


def pid_encoding(float_pids: torch.Tensor) -> torch.Tensor:
    """
    Convert float PIDs to one-hot encoded tensor representation.

    Parameters
    ----------
    float_pids : torch.Tensor
        Input tensor with float PIDs on the last axis

    Returns
    -------
    torch.Tensor
        Tensor with 6-dimensional encoding on the last axis:
        [charge, is_electron, is_muon, is_photon, is_charged_hadron, is_neutral_hadron]
    """
    # Transform float to int
    rounded_pids = torch.round(float_pids * 10).to(torch.int)

    # Create encoding tensors on the same device as input
    device = float_pids.device
    dtype = float_pids.dtype

    # Pre-compute all possible encodings as a lookup table
    pid_lookup = torch.tensor(
        [
            [0, 0, 0, 1, 0, 0],  # photon (0.0)
            [1, 0, 0, 0, 1, 0],  # pi+ (0.1)
            [-1, 0, 0, 0, 1, 0],  # pi- (0.2)
            [0, 0, 0, 0, 0, 1],  # K0_L (0.3)
            [-1, 1, 0, 0, 0, 0],  # e- (0.4)
            [1, 1, 0, 0, 0, 0],  # e+ (0.5)
            [-1, 0, 1, 0, 0, 0],  # mu- (0.6)
            [1, 0, 1, 0, 0, 0],  # mu+ (0.7)
            [1, 0, 0, 0, 1, 0],  # K+ (0.8)
            [-1, 0, 0, 0, 1, 0],  # K- (0.9)
            [1, 0, 0, 0, 1, 0],  # proton (1.0)
            [-1, 0, 0, 0, 1, 0],  # anti-proton (1.1)
            [0, 0, 0, 0, 0, 1],  # neutron (1.2)
            [0, 0, 0, 0, 0, 1],  # anti-neutron (1.3)
        ],
        dtype=dtype,
        device=device,
    )

    # Get shape for reshaping
    original_shape = list(float_pids.shape)
    original_shape[-1] = 6

    # Lookup encodings and reshape to match input dimensions
    encoded = pid_lookup[rounded_pids.flatten()].reshape(original_shape)

    return encoded


def get_range(input, quantile=5e-3, boundary_scale=5e-2):
    if type(input) in [list, tuple]:
        if isinstance(input[0], np.ndarray):
            tensor = torch.cat([torch.from_numpy(element) for element in input], dim=0)
        else:
            tensor = torch.cat(input, dim=0)
    elif isinstance(input, np.ndarray):
        tensor = torch.from_numpy(input)
    elif isinstance(input, torch.Tensor):
        tensor = input
    else:
        raise ValueError("Input must be a list, tuple, numpy array, or torch tensor")
    dtype = tensor.dtype
    tensor = tensor.flatten().to(torch.float32)

    if tensor.size(0) > 1000000:
        tensor = tensor[torch.randperm(tensor.size(0))][:1000000]
    quantiles = torch.quantile(
        tensor,
        torch.tensor([quantile, 1 - quantile], device=tensor.device, dtype=tensor.dtype),
    )
    quantile_range = quantiles[1] - quantiles[0]
    quantiles[0] -= boundary_scale * quantile_range
    quantiles[1] += boundary_scale * quantile_range

    if dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        quantiles = quantiles.round().to(dtype)

    return quantiles


def flatten_dict(d, parent_key="", sep="."):
    """Flattens a nested dictionary with str keys."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, Mapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_xformers_attention_mask(
    batch, batch_condition=None, materialize=False, dtype=torch.float32
):
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
            batch.device, dtype=dtype
        )

    return mask


def xformers_causal_mask(batch, materialize=False):
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

    mask = BlockDiagonalCausalMask.from_seqlens(bincounts)
    if materialize:
        # materialize mask to torch.tensor (only for testing purposes)
        mask = mask.materialize(shape=(len(batch), len(batch))).to(batch.device)

    return mask


def get_flex_attention_mask(batch: torch.Tensor):
    """Returns a mask for the attention mechanism.

    Parameters
    ----------
    batch : torch.Tensor
        Batch vector, maps each token to its sequence in the batch.

    Returns
    -------
    BlockMask
        Block-diagonal BlockMask for flex attention, with one block per sequence.
    """
    N = batch.size(0)

    def jagged_masking(b, h, q_idx, kv_idx):
        return batch[q_idx] == batch[kv_idx]

    mask = create_block_mask(jagged_masking, None, None, N, N, device=batch.device, _compile=True)
    return mask


def get_attention_mask(
    batch: torch.Tensor,
    attention_backend: str,
    dtype: torch.dtype,
    condition_batch: torch.Tensor | None = None,
):
    """Returns the attention mask according to the backend.

    Parameters
    ----------
    batch : torch.Tensor
        Batch vector, maps each token to its sequence in the batch.
    attention_backend : str
        Attention backend to use ("xformers", "flex", or "flash").
    dtype : torch.dtype
        Data type of the attention mask (for xformers backend).

    Returns
    -------
    dict[str, torch.Tensor | BlockMask | BlockDiagonalMask]
        Attention mask for the specified backend.
    """
    on_cpu = batch.device == torch.device("cpu")
    if attention_backend == "xformers":
        mask = get_xformers_attention_mask(
            batch=batch, batch_condition=condition_batch, dtype=dtype, materialize=on_cpu
        )
        if not on_cpu:
            return {"attn_bias": mask}
        else:
            # fallback to default attention
            return {"attn_mask": mask}
    elif attention_backend == "flash":
        raise NotImplementedError("Flash attention backend is not implemented yet.")
        seqlens = torch.bincount(batch).to(torch.int32)
        maxlen = int(seqlens.max().item())
        cu_seqlens = torch.cumsum(seqlens, dim=0, dtype=torch.int32)
        cu_seqlens = torch.cat(
            [torch.tensor([0], dtype=torch.int32, device=seqlens.device), cu_seqlens], dim=0
        )
        if not on_cpu:
            return {
                "cu_seqlens_q": cu_seqlens,
                "cu_seqlens_k": cu_seqlens,
                "max_seqlen_q": maxlen,
                "max_seqlen_k": maxlen,
            }
        else:
            # fallback to default attention
            mask = get_xformers_attention_mask(batch=batch, dtype=dtype, materialize=on_cpu)
            return {"attn_mask": mask}
    elif attention_backend == "flex":
        raise NotImplementedError("Flex attention backend is not implemented yet.")
        mask = get_flex_attention_mask(batch=batch)
        return {"block_mask": mask}
    else:
        raise ValueError(
            f"Unsupported attention backend: {attention_backend}. "
            'Supported backends are "xformers", "flex", and "flash".'
        )


def get_device() -> torch.device:
    """Gets CUDA if available, CPU else."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_nd(tensor, d):
    """Make tensor n-dimensional, group extra dimensions in first."""
    return tensor.view(-1, *(1,) * (max(0, d - 1 - tensor.dim())), *tensor.shape[-(d - 1) :])


def fix_mass(constituents, mass=0.0):
    new_constituents = constituents.clone().to(torch.float64)
    new_constituents[..., 0] = torch.sqrt(
        torch.sum(new_constituents[..., 1:] ** 2, dim=-1) + mass**2
    )
    return new_constituents.to(constituents.dtype)


def kappa_from_Vc(Vc):
    R = 1 - Vc
    if R < 0.53:
        kappa = 2 * R + R**3 + 5 * R**5 / 6
    elif R < 0.85:
        kappa = -0.4 + 1.39 * R + 0.43 / (1 - R)
    elif R < 1 - 5e-3:
        kappa = 1 / (R**3 - 4 * R**2 + 3 * R)
    else:
        kappa = 1 / (2 * (1 - R + EPS2))
    return kappa
