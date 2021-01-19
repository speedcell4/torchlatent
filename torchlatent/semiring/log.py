from logging import getLogger

import torch
from torch import Tensor

from torchlatent.semiring.abc import build_unit_fn, build_bmm_fn, build_reduce_fn

logger = getLogger(__name__)


def logsumexp(x: Tensor, dim: int) -> Tensor:
    with torch.no_grad():
        m, _ = torch.max(x, dim=dim, keepdim=True)
        m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    z = (x - m).exp().sum(dim=dim, keepdim=True)
    mask = z == 0
    z = torch.where(mask, torch.ones_like(z), z).log()
    z = torch.where(mask, torch.full_like(z, -float('inf')), z)
    return (z + m).squeeze(dim=dim)


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
build_unit = build_unit_fn(zero=zero, one=one)

bmm = build_bmm_fn(mul_fn=mul, sum_fn=sum)
reduce = build_reduce_fn(mm_fn=bmm)
