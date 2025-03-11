import torch

from experiments.unfolding.utils import get_batch_from_ptr


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
    sequence = new_batch.x_gen
    ptr = new_batch.x_gen_ptr

    batchsize = len(ptr) - 1

    start_tokens = torch.zeros_like(batchsize, 4)

    new_batch = insert_tokens(new_batch, start_tokens, True)

    return new_batch


def start_sequence(batch):
    new_batch = batch.clone()
    batchsize = len(new_batch.x_gen_ptr) - 1
    start_tokens = torch.zeros((batchsize, 4))
    ptr = torch.arange(batchsize + 1, device=batch.x_gen.device)
    batch_idx = torch.arange(batchsize, device=batch.x_gen.device)
    new_batch.x_gen = start_tokens
    new_batch.x_gen_ptr = ptr
    new_batch.x_gen_batch = batch_idx
    return new_batch


def remove_extra(batch, true_ptr, remove_start=True):
    new_batch = batch.clone()
    seq = new_batch.x_gen
    ptr = new_batch.x_gen_ptr

    idx = torch.zeros(
        seq.shape[0],
        dtype=torch.bool,
        device=seq.device,
    )
    for i in range(len(ptr) - 1):
        n_constituents = true_ptr[i + 1] - true_ptr[i]
        if remove_start:
            idx[ptr[i] + 1 : ptr[i] + 1 + n_constituents] = True
        else:
            idx[ptr[i] : ptr[i] + n_constituents] = True
    new_batch.x_gen = seq[idx]
    new_batch.x_gen_ptr = true_ptr.clone()
    new_batch.x_gen_batch = get_batch_from_ptr(new_batch.x_gen_ptr)

    return new_batch
