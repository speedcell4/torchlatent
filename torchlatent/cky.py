from typing import Tuple, Type, Union

import torch
from torch import Tensor
from torch.distributions.utils import lazy_property
from torchrua import C, D, P

from torchlatent.abc import StructuredDecoder, StructuredDistribution
from torchlatent.semiring import Div, ExceptionSemiring, Log, Max, Semiring, Xen


def cky_scores(logits: C, targets: Union[C, D, P], semiring: Type[Semiring]) -> Tensor:
    xyz, token_sizes = targets = targets.cat()
    batch_ptr, _ = targets.ptr()

    logits = logits.data[batch_ptr, xyz[..., 0], xyz[..., 1], xyz[..., 2]]
    return semiring.segment_prod(logits, token_sizes)


def diag(tensor: Tensor, offset: int) -> Tensor:
    return tensor.diagonal(offset=offset, dim1=1, dim2=2)


def diag_scatter(chart: Tensor, score: Tensor, offset: int) -> None:
    chart.diagonal(offset=offset, dim1=1, dim2=2)[::] = score


def left(chart: Tensor, offset: int) -> Tensor:
    b, t, _, *size = chart.size()
    c, n, m, *stride = chart.stride()
    return chart.as_strided(
        size=(b, t - offset, offset, *size),
        stride=(c, n + m, m, *stride),
    )


def right(chart: Tensor, offset: int) -> Tensor:
    b, t, _, *size = chart.size()
    c, n, m, *stride = chart.stride()
    return chart[:, 1:, offset:].as_strided(
        size=(b, t - offset, offset, *size),
        stride=(c, n + m, n, *stride),
    )


def cky_partitions(logits: C, semiring: Type[Semiring]) -> Tensor:
    chart = torch.full_like(logits.data, fill_value=semiring.zero, requires_grad=False)

    diag_scatter(chart, diag(logits.data, offset=0), offset=0)

    for w in range(1, chart.size()[1]):
        score = semiring.sum(semiring.mul(left(chart, offset=w), right(chart, offset=w)), dim=2)
        diag_scatter(chart, semiring.mul(score, diag(logits.data, offset=w)), offset=w)

    index = torch.arange(chart.size()[0], dtype=torch.long, device=chart.device)
    return chart[index, 0, logits.token_sizes - 1]


def cky_exceptions(logits1: C, logits2: C, log_prob: C,
                   semiring: Type[Semiring], exception: Type[ExceptionSemiring]) -> Tensor:
    chart1 = torch.full_like(logits1.data, fill_value=semiring.zero, requires_grad=False)
    chart2 = torch.full_like(logits2.data, fill_value=semiring.zero, requires_grad=False)
    chart3 = torch.full_like(log_prob.data, fill_value=exception.zero, requires_grad=False)

    diag_scatter(chart1, diag(logits1.data, offset=0), offset=0)
    diag_scatter(chart2, diag(logits2.data, offset=0), offset=0)
    diag_scatter(chart3, diag(log_prob.data, offset=0), offset=0)

    for w in range(1, chart3.size()[1]):
        score1 = semiring.mul(left(chart1, offset=w), right(chart1, offset=w))
        score2 = semiring.mul(left(chart2, offset=w), right(chart2, offset=w))
        tensor = exception.mul(left(chart3, offset=w), right(chart3, offset=w))

        diag_scatter(chart1, semiring.mul(
            semiring.sum(score1, dim=2),
            diag(logits1.data, offset=w),
        ), offset=w)

        diag_scatter(chart2, semiring.mul(
            semiring.sum(score2, dim=2),
            diag(logits2.data, offset=w),
        ), offset=w)

        log_p = score1 - semiring.sum(score1, dim=-1, keepdim=True)
        log_q = score2 - semiring.sum(score2, dim=-1, keepdim=True)
        diag_scatter(chart3, exception.mul(
            exception.sum(tensor, log_p=log_p, log_q=log_q, dim=2),
            diag(log_prob.data, offset=w),
        ), offset=w)

    index = torch.arange(chart3.size()[0], dtype=torch.long, device=chart3.device)
    return chart3[index, 0, log_prob.token_sizes - 1]


def masked_select(mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    _, t, _, n = mask.size()

    index = torch.arange(t, device=mask.device)
    x = torch.masked_select(index[None, :, None, None], mask=mask)
    y = torch.masked_select(index[None, None, :, None], mask=mask)

    index = torch.arange(n, device=mask.device)
    z = torch.masked_select(index[None, None, None, :], mask=mask)

    return x, y, z


class CkyDistribution(StructuredDistribution):
    def __init__(self, logits: C) -> None:
        super(CkyDistribution, self).__init__(logits=logits)

    def log_scores(self, targets: Union[C, D, P]) -> Tensor:
        return cky_scores(
            logits=self.logits, targets=targets,
            semiring=Log,
        )

    @lazy_property
    def log_partitions(self) -> Tensor:
        return cky_partitions(
            logits=self.logits._replace(data=Log.sum(self.logits.data, dim=-1)),
            semiring=Log,
        )

    @lazy_property
    def max(self) -> Tensor:
        return cky_partitions(
            logits=self.logits._replace(data=Max.sum(self.logits.data, dim=-1)),
            semiring=Max,
        )

    @lazy_property
    def entropy(self) -> Tensor:
        return cky_exceptions(
            logits1=self.logits._replace(data=Log.sum(self.logits.data, dim=-1)),
            logits2=self.logits._replace(data=Log.sum(self.logits.data, dim=-1)),
            log_prob=self.logits._replace(data=Xen.sum(
                torch.full_like(self.logits.data, fill_value=Xen.zero),
                torch.log_softmax(self.logits.data, dim=-1),
                torch.log_softmax(self.logits.data, dim=-1),
                dim=-1,
            )),
            semiring=Log,
            exception=Xen,
        )

    def kl(self, other: 'CkyDistribution') -> Tensor:
        return cky_exceptions(
            logits1=self.logits._replace(data=Log.sum(self.logits.data, dim=-1)),
            logits2=other.logits._replace(data=Log.sum(other.logits.data, dim=-1)),
            log_prob=self.logits._replace(data=Div.sum(
                torch.full_like(self.logits.data, fill_value=Div.zero),
                torch.log_softmax(self.logits.data, dim=-1),
                torch.log_softmax(other.logits.data, dim=-1),
                dim=-1,
            )),
            semiring=Log,
            exception=Div,
        )

    @lazy_property
    def argmax(self) -> C:
        argmax = super(CkyDistribution, self).argmax
        x, y, z = masked_select(argmax > 0)

        return C(
            data=torch.stack([x, y, z], dim=-1),
            token_sizes=self.logits.token_sizes * 2 - 1,
        )


class CkyDecoder(StructuredDecoder):
    def __init__(self, *, num_targets: int) -> None:
        super(CkyDecoder, self).__init__(num_targets=num_targets)

    def forward(self, logits: C) -> CkyDistribution:
        return CkyDistribution(logits=logits)


if __name__ == '__main__':
    from torch_struct import TreeCRF

    num_targets = 17
    logits1 = C(data=torch.randn((3, 5, 5, num_targets), requires_grad=True), token_sizes=torch.tensor([5, 2, 3]))
    logits2 = C(data=torch.randn((3, 5, 5, num_targets), requires_grad=True), token_sizes=torch.tensor([5, 2, 3]))

    excepted1 = TreeCRF(logits1.data, logits1.token_sizes)
    excepted2 = TreeCRF(logits2.data, logits2.token_sizes)
    print(excepted1.kl(excepted2))

    actual1 = CkyDecoder(num_targets=num_targets)(logits1)
    actual2 = CkyDecoder(num_targets=num_targets)(logits2)
    print(actual1.kl(actual2))
