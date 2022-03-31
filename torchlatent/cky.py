from typing import NamedTuple, Tuple, Type

import torch
from torch import Tensor
from torch.types import Device
from torchrua import major_sizes_to_ptr, accumulate_sizes

from torchlatent.semiring import Semiring


class CkyIndices(NamedTuple):
    width_size: int
    cache_size: int

    src: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]
    tgt: Tuple[Tensor, Tensor]


@torch.no_grad()
def cky_indices(token_sizes: Tensor, device: Device = None):
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)
    acc_token_sizes = accumulate_sizes(sizes=token_sizes)

    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)
    x_ptr, z_ptr = major_sizes_to_ptr(sizes=token_ptr + 1)
    batch_ptr = batch_ptr[z_ptr]
    y_ptr = z_ptr - acc_token_sizes[batch_ptr]

    width_size = token_sizes.max().item()
    cache_size, = token_ptr.size()

    return CkyIndices(
        width_size=width_size, cache_size=cache_size,
        src=((y_ptr - x_ptr, z_ptr), (batch_ptr, x_ptr, y_ptr)),
        tgt=(token_sizes - 1, acc_token_sizes),
    )


def cky_partition(data: Tensor, indices: CkyIndices, semiring: Type[Semiring]) -> Tensor:
    width_size, cache_size, (src1, src2), tgt = indices

    tensor0 = torch.full((width_size, cache_size, *data.size()[3:]), fill_value=semiring.zero, requires_grad=False)
    tensor1 = torch.full((width_size, cache_size, *data.size()[3:]), fill_value=semiring.zero, requires_grad=False)
    tensor2 = torch.full((width_size, cache_size, *data.size()[3:]), fill_value=semiring.zero, requires_grad=False)

    tensor0[src1] = data[src2]
    tensor1[0, :] = tensor2[-1, :] = tensor0[0, :]

    for w in range(1, width_size):
        tensor1[w, :-w] = tensor2[-w - 1, w:] = semiring.mul(
            semiring.sum(semiring.mul(tensor1[:w, :-w], tensor2[-w:, w:]), dim=0),
            tensor0[w, w:],
        )

    return tensor1[tgt]
