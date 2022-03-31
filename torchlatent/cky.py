from typing import Tuple, NamedTuple
from typing import Type

import torch
from torch import Tensor
from torch.distributions.utils import lazy_property
from torch.types import Device
from torchrua import major_sizes_to_ptr, accumulate_sizes

from torchlatent.abc import DistributionABC
from torchlatent.semiring import Semiring, Log, Max
from torchlatent.types import Sequence


class CkyIndices(NamedTuple):
    token_size: int
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

    token_size = token_sizes.max().item()
    cache_size, = token_ptr.size()

    return CkyIndices(
        token_size=token_size, cache_size=cache_size,
        src=((y_ptr - x_ptr, z_ptr), (batch_ptr, x_ptr, y_ptr)),
        tgt=(token_sizes - 1, acc_token_sizes),
    )


def cky_partition(data: Tensor, indices: CkyIndices, semiring: Type[Semiring]) -> Tensor:
    token_size, cache_size, (src1, src2), tgt = indices

    tensor0 = torch.full((token_size, cache_size, *data.size()[3:]), fill_value=semiring.zero, requires_grad=False)
    tensor1 = torch.full((token_size, cache_size, *data.size()[3:]), fill_value=semiring.zero, requires_grad=False)
    tensor2 = torch.full((token_size, cache_size, *data.size()[3:]), fill_value=semiring.zero, requires_grad=False)

    tensor0[src1] = data[src2]
    tensor1[0, :] = tensor2[-1, :] = tensor0[0, :]

    for w in range(1, token_size):
        tensor1[w, :-w] = tensor2[-w - 1, w:] = semiring.mul(
            semiring.sum(semiring.mul(tensor1[:w, :-w], tensor2[-w:, w:]), dim=0),
            tensor0[w, w:],
        )

    return tensor1[tgt]


class CkyDistribution(DistributionABC):
    def __init__(self, scores: Tensor, indices: CkyIndices) -> None:
        super(CkyDistribution, self).__init__(validate_args=False)

        self.scores = scores
        self.indices = indices

    def log_scores(self, targets: Sequence) -> Tensor:
        raise NotImplementedError

    @lazy_property
    def log_partitions(self) -> Tensor:
        return cky_partition(data=self.scores, indices=self.indices, semiring=Log)

    @lazy_property
    def max(self) -> Tensor:
        return cky_partition(data=self.scores, indices=self.indices, semiring=Max)

    @lazy_property
    def argmax(self) -> Tensor:
        pass

    @lazy_property
    def log_marginals(self) -> Tensor:
        pass

    @lazy_property
    def entropy(self) -> Tensor:
        pass


if __name__ == '__main__':
    ans = CkyDistribution(
        torch.randn((3, 5, 5), requires_grad=True),
        cky_indices(torch.tensor([5, 2, 3])),
    ).max
    print(ans)
