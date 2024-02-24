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
    if emissions.data.dim() == 4:
        data = semiring.sum(emissions.data, dim=-1)
        emissions = emissions._replace(data=data)

    chart = torch.full_like(emissions.data, fill_value=semiring.zero, requires_grad=False)
    b, t, _, *size = chart.size()
    c, n, m, *stride = chart.stride()

    def diag(offset: int) -> Tensor:
        return emissions.data.diagonal(offset=offset, dim1=1, dim2=2)

    def diag_scatter(tensor: Tensor, offset: int) -> None:
        chart.diagonal(offset=offset, dim1=1, dim2=2)[::] = tensor

    def left(offset: int) -> Tensor:
        return chart.as_strided(size=(b, t - offset, offset, *size), stride=(c, n + m, m, *stride))

    def right(offset: int) -> Tensor:
        return chart[:, 1:, offset:].as_strided(size=(b, t - offset, offset, *size), stride=(c, n + m, n, *stride))

    score = diag(offset=0)
    diag_scatter(score, offset=0)

    for w in range(1, t):
        score = semiring.sum(semiring.mul(left(offset=w), right(offset=w)), dim=2)
        diag_scatter(semiring.mul(score, diag(offset=w)), offset=w)

    index = torch.arange(b, dtype=torch.long, device=chart.device)
    return chart[index, 0, emissions.token_sizes - 1]


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
            emissions=self.emissions,
            semiring=Log,
        )

    @lazy_property
    def max(self) -> Tensor:
        return cky_partitions(
            emissions=self.emissions,
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
