from logging import getLogger

import torch
from torch import Tensor

from torchlatent.semiring.abc import build_unit_fn, build_bmv_fn, build_bvm_fn, build_bmm_fn, build_reduce_fn

logger = getLogger(__name__)


def add(lhs: Tensor, rhs: Tensor) -> Tensor:
    return torch.max(lhs, rhs)


def mul(lhs: Tensor, rhs: Tensor) -> Tensor:
    return lhs + rhs


def sum(x: Tensor, dim: int) -> Tensor:
    return x.max(dim=dim)[0]


def prod(x: Tensor, dim: int) -> Tensor:
    return x.sum(dim=dim)


zero: float = float('-inf')
one: float = 0
build_unit = build_unit_fn(zero=zero, one=one)

bmv = build_bmv_fn(mul_fn=mul, sum_fn=sum)
bvm = build_bvm_fn(mul_fn=mul, sum_fn=sum)

try:

    from bym import maxbmm as bmm

    logger.info('using bym.maxbmm')

except ImportError:

    bmm = build_bmm_fn(mul_fn=mul, sum_fn=sum)

    logger.info('using naive.maxbmm')

reduce = build_reduce_fn(mm_fn=bmm)
