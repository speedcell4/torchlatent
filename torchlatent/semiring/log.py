from logging import getLogger

import torch
from torch import Tensor

from torchlatent.semiring.abc import build_build_unit, build_mv_fn, build_vm_fn, build_mm_fn, build_reduce_fn

logger = getLogger(__name__)


def logsumexp(x: Tensor, dim: int) -> Tensor:
    m, _ = x.max(dim=dim)
    mask = m == -float('inf')

    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float('inf'))


def add(lhs: Tensor, rhs: Tensor) -> Tensor:
    return logsumexp(torch.stack([lhs, rhs], dim=-1), dim=-1)


def mul(lhs: Tensor, rhs: Tensor) -> Tensor:
    return lhs + rhs


def sum(x: Tensor, dim: int) -> Tensor:
    return logsumexp(x, dim=dim)


def prod(x: Tensor, dim: int) -> Tensor:
    return x.sum(dim=dim)


zero: float = -float('inf')
one: float = 0
build_unit = build_build_unit(zero=zero, one=one)

mv = build_mv_fn(mul_fn=mul, sum_fn=sum)
vm = build_vm_fn(mul_fn=mul, sum_fn=sum)

try:

    from bym import logbmm as mm

    logger.info('using bym.logbmm')

except ImportError:

    mm = build_mm_fn(mul_fn=mul, sum_fn=sum)

    logger.info('using naive.logbmm')

reduce = build_reduce_fn(mm_fn=mm)
