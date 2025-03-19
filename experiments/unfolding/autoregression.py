import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from experiments.unfolding.utils import get_batch_from_ptr

flex_attention = torch.compile(flex_attention, dynamic=True)
create_block_mask = torch.compile(create_block_mask, dynamic=True)

START_TOKEN = torch.tensor([2, 1, 1, 1], dtype=torch.float32)


def insert_tokens(batch, tokens, start=False):
    new_batch = batch.clone()
    sequence = new_batch.x_gen
    ptr = new_batch.x_gen_ptr

    batchsize = len(ptr) - 1
    arange = torch.arange(batchsize, device=sequence.device)

    assert len(tokens) % batchsize == 0
    n_tokens = len(tokens) // batchsize

    if start:
        token_idx = torch.stack(
            [ptr[:-1] + i for i in range(n_tokens)], dim=0
        ) + n_tokens * torch.arange(batchsize, device=ptr.device)
        token_idx = token_idx.permute(1, 0).flatten()
    else:
        token_idx = torch.stack(
            [ptr[1:] + i for i in range(n_tokens)], dim=0
        ) + n_tokens * torch.arange(batchsize, device=ptr.device)
        token_idx = token_idx.permute(1, 0).flatten()

    insert_token = torch.zeros(
        sequence.shape[0] + batchsize * n_tokens,
        dtype=torch.bool,
        device=sequence.device,
    )
    insert_token[token_idx] = True
    sequence_buffer = sequence.clone()
    sequence = torch.empty(
        insert_token.shape[0],
        *sequence.shape[1:],
        dtype=sequence.dtype,
        device=sequence.device,
    )

    sequence[~insert_token] = sequence_buffer
    sequence[insert_token] = tokens

    ptr[1:] = ptr[1:] + (arange + 1) * n_tokens
    batch_idx = get_batch_from_ptr(ptr)

    new_batch.x_gen = sequence
    new_batch.x_gen_ptr = ptr
    new_batch.x_gen_batch = batch_idx

    return new_batch


def add_start_tokens(batch):
    new_batch = batch.clone()

    start_tokens = new_batch.x_det[new_batch.x_det_ptr[:-1]]
    new_batch = insert_tokens(new_batch, start_tokens, True)

    return new_batch


def start_sequence(batch):
    new_batch = batch.clone()
    batchsize = len(new_batch.x_gen_ptr) - 1
    start_tokens = new_batch.x_det[new_batch.x_det_ptr[:-1]]
    ptr = torch.arange(batchsize + 1, device=batch.x_gen.device)
    batch_idx = torch.arange(batchsize, device=batch.x_gen.device)
    new_batch.x_gen = start_tokens
    new_batch.x_gen_ptr = ptr
    new_batch.x_gen_batch = batch_idx
    return new_batch


def remove_extra(batch, true_ptr, remove_start=False):
    new_batch = batch.clone()
    seq = new_batch.x_gen
    ptr = new_batch.x_gen_ptr

    n_constituents = true_ptr[1:] - true_ptr[:-1]  # Compute lengths of each segment
    start_indices = ptr[:-1] + remove_start  # Adjust for remove_start flag
    ranges = torch.arange(seq.shape[0], device=seq.device).unsqueeze(0)

    # Create a mask by checking which indices fall in valid ranges
    mask = (ranges >= start_indices.unsqueeze(1)) & (
        ranges < (start_indices + n_constituents).unsqueeze(1)
    )
    idx = mask.any(dim=0)  # Collapse across rows to get the final index mask

    new_batch.x_gen = seq[idx]
    new_batch.x_gen_ptr = true_ptr.clone()
    new_batch.x_gen_batch = get_batch_from_ptr(new_batch.x_gen_ptr)

    return new_batch
