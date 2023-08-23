from typing import Type
from typing import Union

import torch
from torch import Tensor

from torchlatent.semiring import Semiring
from torchrua import C
from torchrua import D
from torchrua import P


def cky_scores(emissions: C, targets: Union[C, D, P], semiring: Type[Semiring]) -> Tensor:
    xyz, token_sizes = targets = targets.cat()
    batch_ptr, _ = targets.ptr()

    emissions = emissions.data[batch_ptr, xyz[..., 0], xyz[..., 1], xyz[..., 2]]
    return semiring.segment_prod(emissions, token_sizes)


def cky_partitions(emissions: C, semiring: Type[Semiring]) -> Tensor:
    batch_ptr, token_ptr = emissions.ptr()
    z_ptr, x_ptr = emissions._replace(token_sizes=token_ptr + 1).ptr()
    y_ptr = token_ptr[z_ptr]

    _, token_size, *_ = emissions.size()
    cache_size, = y_ptr.size()

    src1 = y_ptr - x_ptr, x_ptr + z_ptr - y_ptr
    src2 = batch_ptr[z_ptr], x_ptr, y_ptr
    tgt = emissions.token_sizes - 1, emissions.offsets()

    size = (token_size, cache_size, *emissions.data.size()[3:])
    tensor0 = torch.full(size, fill_value=semiring.zero, device=emissions.data.device, requires_grad=False)
    tensor1 = torch.full(size, fill_value=semiring.zero, device=emissions.data.device, requires_grad=False)
    tensor2 = torch.full(size, fill_value=semiring.zero, device=emissions.data.device, requires_grad=False)

    tensor0[src1] = emissions.data[src2]
    tensor1[0, :] = tensor2[-1, :] = tensor0[0, :]

    for w in range(1, token_size):
        tensor1[w, :-w] = tensor2[-w - 1, w:] = semiring.mul(
            semiring.sum(semiring.mul(tensor1[:w, :-w], tensor2[-w:, w:]), dim=0),
            tensor0[w, :-w],
        )

    return tensor1[tgt]
