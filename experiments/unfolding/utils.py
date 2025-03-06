import torch
from xformers.ops.fmha import BlockDiagonalMask
import math
from torch import nn

from experiments.logger import LOGGER


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


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim, input_dim=1, scale=30.0):
        super().__init__()
        self.proj = GaussianFourierProjection(embed_dim, input_dim, scale)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.linear.weight.requires_grad = False
        self.linear.bias.requires_grad = False

    def forward(self, t):
        projection = self.proj(t)
        embedding = self.linear(projection)
        return embedding


# log(x) -> log(x+EPS1)
# in (invertible) preprocessing functions to avoid being close to log(0)
EPS1 = 1e-5

# generic numerical stability cutoff
EPS2 = 1e-10

# exp(x) -> exp(x.clamp(max=CUTOFF))
CUTOFF = 10


def unpack_last(x):
    # unpack along the last dimension
    n = len(x.shape)
    return torch.permute(x, (n - 1, *list(range(n - 1))))


def fourmomenta_to_jetmomenta(fourmomenta):
    pt = get_pt(fourmomenta)
    phi = get_phi(fourmomenta)
    eta = get_eta(fourmomenta)
    mass = get_mass(fourmomenta)

    jetmomenta = torch.stack((pt, phi, eta, mass), dim=-1)
    assert torch.isfinite(jetmomenta).all()
    return jetmomenta


def jetmomenta_to_fourmomenta(jetmomenta):
    pt, phi, eta, mass = unpack_last(jetmomenta)

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta.clamp(min=-CUTOFF, max=CUTOFF))
    E = torch.sqrt(mass**2 + px**2 + py**2 + pz**2)

    fourmomenta = torch.stack((E, px, py, pz), dim=-1)
    assert torch.isfinite(fourmomenta).all()
    return fourmomenta


def ensure_angle(phi):
    return torch.remainder(phi + torch.pi, 2 * torch.pi) - torch.pi


def xformers_sa_mask(batch, materialize=False):
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
    mask = BlockDiagonalMask.from_seqlens(bincounts)
    if materialize:
        # materialize mask to torch.tensor (only for testing purposes)
        mask = mask.materialize(shape=(len(batch), len(batch))).to(batch.device)
    return mask


def get_batch_from_ptr(ptr):
    ptr = ptr.to(torch.int)
    return (
        torch.arange(len(ptr) - 1, device=ptr.device)
        .repeat_interleave(
            ptr[1:] - ptr[:-1],
        )
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
    m = torch.sqrt(m2.clamp(min=EPS2))
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
