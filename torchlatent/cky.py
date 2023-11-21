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
    b, t, _, *size = emissions.data.size()
    c, n, m, *stride = emissions.data.stride()

    chart = torch.full_like(emissions.data, fill_value=Log.zero, requires_grad=False)

    def diag() -> Tensor:
        return emissions.data.diagonal(offset=w, dim1=1, dim2=2)

    def diag_scatter(tensor: Tensor) -> None:
        chart.diagonal(offset=w, dim1=1, dim2=2)[::] = tensor

    def left() -> Tensor:
        return chart.as_strided(size=(b, t - w, w, *size), stride=(c, n + m, m, *stride))

    def right() -> Tensor:
        return chart[:, 1:, w:].as_strided(size=(b, t - w, w, *size), stride=(c, n + m, n, *stride))

    w = 0
    diag_scatter(diag())

    for w in range(1, t):
        diag_scatter(semiring.mul(semiring.sum(semiring.mul(left(), right()), dim=2), diag()))

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
