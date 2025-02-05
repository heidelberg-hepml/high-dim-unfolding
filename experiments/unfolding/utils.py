import torch
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


def get_type_token(x_ref, type_token_channels):
    # embed token for the particle type
    type_token_raw = torch.arange(
        x_ref.shape[-2], device=x_ref.device, dtype=torch.long
    )
    type_token = nn.functional.one_hot(type_token_raw, num_classes=type_token_channels)
    type_token = type_token.expand(*x_ref.shape[:-1], type_token_channels)
    return type_token


def get_process_token(x_ref, ijet, process_token_channels):
    # embed token for the process
    process_token_raw = torch.tensor([ijet], device=x_ref.device, dtype=torch.long)
    process_token = nn.functional.one_hot(
        process_token_raw, num_classes=process_token_channels
    )
    process_token = process_token.expand(*x_ref.shape[:-1], process_token_channels)
    return process_token
