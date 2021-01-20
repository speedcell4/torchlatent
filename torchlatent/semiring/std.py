import torch
from torch import Tensor

from torchlatent.semiring.abc import compile_fill_unit, compile_tree_reduction


def add(lhs: Tensor, rhs: Tensor) -> Tensor:
    return lhs + rhs


def mul(lhs: Tensor, rhs: Tensor) -> Tensor:
    return lhs * rhs


def sum(x: Tensor, dim: int) -> Tensor:
    return x.sum(dim=dim)


def prod(x: Tensor, dim: int) -> Tensor:
    return x.prod(dim=dim)


bmm = torch.bmm
fill_unit = compile_fill_unit(zero=0., one=1.)
tree_reduce = compile_tree_reduction(bmm=bmm)
