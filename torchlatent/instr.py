from typing import List, Optional
from typing import NamedTuple

import torch
from torch import Tensor
from torchrua import accumulate_batch_sizes
from torchrua import lengths_to_ptr


class TreeReductionIndices(NamedTuple):
    xs: List[Tensor]
    ys: List[Tensor]
    zs: List[Tensor]
    head: Tensor
    last: Tensor


@torch.no_grad()
def tree_reduction_indices(lengths: Tensor, device: Optional[torch.device]) -> TreeReductionIndices:
    if device is not None:
        device = lengths.device

    batch_ptr2, token_ptr2, batch_sizes = lengths_to_ptr(
        lengths=lengths * 2 - 1,
        sorted_indices=None,
        device=device,
    )
    acc_batch_sizes = accumulate_batch_sizes(batch_sizes)
    offsets = torch.zeros_like(lengths)

    head = torch.ones_like(token_ptr2, dtype=torch.bool)
    last = acc_batch_sizes[lengths * 2 - 2] + batch_ptr2[:batch_sizes[0]]

    xs, ys, zs = [], [], []
    while (lengths != 1).any().item():
        clamp_lengths = torch.masked_fill(lengths // 2, lengths <= (lengths[0] + 1) // 2, 0)

        batch_ptr, token_ptr, _ = lengths_to_ptr(clamp_lengths, sorted_indices=None, device=device)
        base_ptr = offsets[batch_ptr] + token_ptr

        x = acc_batch_sizes[base_ptr + token_ptr + 0] + batch_ptr
        y = acc_batch_sizes[base_ptr + token_ptr + 1] + batch_ptr
        z = acc_batch_sizes[base_ptr + clamp_lengths[batch_ptr] * 2] + batch_ptr
        xs.append(x)
        ys.append(y)
        zs.append(z)

        offsets = offsets + clamp_lengths * 2
        lengths = lengths - clamp_lengths
        head = torch.scatter(head, dim=0, index=z, value=False)

    head = acc_batch_sizes[token_ptr2[head]] + batch_ptr2[head]

    return TreeReductionIndices(xs=xs, ys=ys, zs=zs, head=head, last=last)
