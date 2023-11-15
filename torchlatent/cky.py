from typing import Type, Union

import torch
from torch import Tensor
from torch.distributions.utils import lazy_property
from torchrua import C, D, P

from torchlatent.abc import StructuredDecoder, StructuredDistribution
from torchlatent.semiring import Log, Max, Semiring


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
    cache_size, = batch_ptr.size()

    w_ptr = y_ptr - x_ptr
    src1 = w_ptr, z_ptr - w_ptr
    # src2 = -w_ptr - 1, z_ptr

    src = batch_ptr[z_ptr], x_ptr, y_ptr
    tgt = emissions.token_sizes - 1, emissions.offsets()

    size = (token_size, cache_size, *emissions.data.size()[3:])
    score1 = emissions.data.new_full(size, fill_value=semiring.zero, requires_grad=False)
    # score2 = emissions.data.new_full(size, fill_value=semiring.zero, requires_grad=False)
    chart1 = emissions.data.new_full(size, fill_value=semiring.zero, requires_grad=False)
    chart2 = emissions.data.new_full(size, fill_value=semiring.zero, requires_grad=False)

    score1[src1] = emissions.data[src]
    # score2[src2] = emissions.data[src]
    chart1[0, :] = chart2[-1, :] = score1[0, :]

    for w in range(1, token_size):
        chart1[w, :-w] = chart2[-w - 1, w:] = semiring.mul(
            semiring.sum(semiring.mul(chart1[:w, :-w], chart2[-w:, w:]), dim=0),
            score1[w, :-w],
        )

    return chart1[tgt]


class CkyDistribution(StructuredDistribution):
    def __init__(self, emissions: C) -> None:
        super(CkyDistribution, self).__init__(emissions=emissions)

    def log_scores(self, targets: Union[C, D, P]) -> Tensor:
        return cky_scores(
            emissions=self.emissions, targets=targets,
            semiring=Log,
        )

    @lazy_property
    def log_partitions(self) -> Tensor:
        return cky_partitions(
            emissions=self.emissions._replace(data=Log.sum(self.emissions.data, dim=-1)),
            semiring=Log,
        )

    @lazy_property
    def max(self) -> Tensor:
        return cky_partitions(
            emissions=self.emissions._replace(data=Max.sum(self.emissions.data, dim=-1)),
            semiring=Max,
        )

    @lazy_property
    def argmax(self) -> C:
        mask = super(CkyDistribution, self).argmax > 0
        _, t, _, n = mask.size()

        index = torch.arange(t, device=mask.device)
        x = torch.masked_select(index[None, :, None, None], mask=mask)
        y = torch.masked_select(index[None, None, :, None], mask=mask)

        index = torch.arange(n, device=mask.device)
        z = torch.masked_select(index[None, None, None, :], mask=mask)

        return C(
            data=torch.stack([x, y, z], dim=-1),
            token_sizes=self.emissions.token_sizes * 2 - 1,
        )


class CkyDecoder(StructuredDecoder):
    def __init__(self, *, num_targets: int) -> None:
        super(CkyDecoder, self).__init__(num_targets=num_targets)

    def forward(self, emissions: C) -> CkyDistribution:
        return CkyDistribution(emissions=emissions)
