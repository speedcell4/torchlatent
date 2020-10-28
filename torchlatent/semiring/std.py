import torch
from torch import Tensor

from torchlatent.semiring.abc import build_unit_fn, build_bmv_fn, build_bvm_fn, build_reduce_fn


def add(lhs: Tensor, rhs: Tensor) -> Tensor:
    return lhs + rhs


def mul(lhs: Tensor, rhs: Tensor) -> Tensor:
    return lhs * rhs


def sum(x: Tensor, dim: int) -> Tensor:
    return x.sum(dim=dim)


def prod(x: Tensor, dim: int) -> Tensor:
    return x.prod(dim=dim)


zero: float = 0.
one: float = 1.
build_unit = build_unit_fn(zero=zero, one=one)

bvm = build_bvm_fn(mul_fn=mul, sum_fn=sum)
bmv = build_bmv_fn(mul_fn=mul, sum_fn=sum)
bmm = torch.bmm
reduce = build_reduce_fn(mm_fn=bmm)
