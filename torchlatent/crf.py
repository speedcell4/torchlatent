from typing import Tuple, Type, Union

import torch
from torch import Tensor, nn
from torch.distributions.utils import lazy_property
from torch.nn import init
from torchrua import C, D, P

from torchlatent.abc import StructuredDecoder, StructuredDistribution
from torchlatent.semiring import Log, Max, Semiring

T = Tuple[Tensor, Tensor, Tensor]


def crf_scores(logits: Union[C, D, P], targets: Union[C, D, P], bias: T, semiring: Type[Semiring]) -> Tensor:
    bias, head_bias, last_bias = bias

    targets = _, token_sizes = targets.cat()
    head_bias = targets.head().rua(head_bias)
    last_bias = targets.last().rua(last_bias)
    bias = targets.data.roll(1).rua(bias, targets)

    logits, _ = logits.idx().cat().rua(logits, targets)
    logits = semiring.segment_prod(logits, sizes=token_sizes)

    token_sizes = torch.stack([torch.ones_like(token_sizes), token_sizes - 1], dim=-1)
    bias = semiring.segment_prod(bias, sizes=token_sizes.view(-1))[1::2]

    return semiring.mul(
        semiring.mul(head_bias, last_bias),
        semiring.mul(logits, bias),
    )


def crf_partitions(logits: Union[C, D, P], bias: T, semiring: Type[Semiring]) -> Tensor:
    bias, head_bias, last_bias = bias

    logits = logits.pack()
    last_indices = logits.idx().last()
    logits, batch_sizes, _, _ = logits

    _, *batch_sizes = sections = batch_sizes.detach().cpu().tolist()
    emission, *logits = torch.split(logits, sections, dim=0)

    charts = [semiring.mul(head_bias, emission)]
    for emission, batch_size in zip(logits, batch_sizes):
        charts.append(semiring.mul(
            semiring.bmm(charts[-1][:batch_size], bias),
            emission,
        ))

    emission = torch.cat(charts, dim=0)[last_indices]
    return semiring.sum(semiring.mul(emission, last_bias), dim=-1)


class CrfDistribution(StructuredDistribution):
    def __init__(self, logits: Union[C, D, P], bias: T) -> None:
        super(CrfDistribution, self).__init__(logits=logits)
        self.bias = bias

    def log_scores(self, targets: Union[C, D, P]) -> Tensor:
        return crf_scores(
            logits=self.logits, targets=targets,
            bias=self.bias,
            semiring=Log,
        )

    @lazy_property
    def log_partitions(self) -> Tensor:
        return crf_partitions(
            logits=self.logits,
            bias=self.bias,
            semiring=Log,
        )

    @lazy_property
    def max(self) -> Tensor:
        return crf_partitions(
            logits=self.logits,
            bias=self.bias,
            semiring=Max,
        )

    @lazy_property
    def argmax(self) -> Union[C, D, P]:
        argmax = super(CrfDistribution, self).argmax.argmax(dim=-1)
        return self.logits._replace(data=argmax)


class CrfDecoder(StructuredDecoder):
    def __init__(self, *, num_targets: int) -> None:
        super(CrfDecoder, self).__init__(num_targets=num_targets)

        self.bias = nn.Parameter(torch.empty((num_targets, num_targets)))
        self.head_bias = nn.Parameter(torch.empty((num_targets,)))
        self.last_bias = nn.Parameter(torch.empty((num_targets,)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.zeros_(self.bias)
        init.zeros_(self.head_bias)
        init.zeros_(self.last_bias)

    def forward(self, logits: Union[C, D, P]) -> CrfDistribution:
        return CrfDistribution(
            logits=logits,
            bias=(
                self.bias,
                self.head_bias,
                self.last_bias,
            ),
        )
