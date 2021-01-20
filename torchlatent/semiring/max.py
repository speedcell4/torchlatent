import torch
from torch import Tensor

from torchlatent.semiring.abc import compile_fill_unit, compile_bmm, compile_tree_reduction


def add(lhs: Tensor, rhs: Tensor) -> Tensor:
    return torch.max(lhs, rhs)


def mul(lhs: Tensor, rhs: Tensor) -> Tensor:
    return lhs + rhs


def sum(x: Tensor, dim: int) -> Tensor:
    return x.max(dim=dim)[0]


def prod(x: Tensor, dim: int) -> Tensor:
    return x.sum(dim=dim)


bmm = compile_bmm(mul=mul, sum=sum)
fill_unit = compile_fill_unit(zero=float('-inf'), one=0.)
tree_reduce = compile_tree_reduction(bmm=bmm)
