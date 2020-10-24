from torch import Tensor

from torchlatent.semiring.abc import build_build_unit, build_mv_fn, build_vm_fn, build_mm_fn, build_reduce_fn


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
build_unit = build_build_unit(zero=zero, one=one)

mv = build_mv_fn(mul_fn=mul, sum_fn=sum)
vm = build_vm_fn(mul_fn=mul, sum_fn=sum)
mm = build_mm_fn(mul_fn=mul, sum_fn=sum)
reduce = build_reduce_fn(mm_fn=mm)
