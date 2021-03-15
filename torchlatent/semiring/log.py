import torch
from torch import Tensor

from torchlatent.functional import logsumexp
from torchlatent.semiring.abc import compile_fill_unit, compile_bmm, compile_tree_reduction


def add(lhs: Tensor, rhs: Tensor) -> Tensor:
    return logsumexp(torch.stack([lhs, rhs], dim=-1), dim=-1)


def mul(lhs: Tensor, rhs: Tensor) -> Tensor:
    return lhs + rhs


def sum(x: Tensor, dim: int) -> Tensor:
    return logsumexp(x, dim=dim)


def prod(x: Tensor, dim: int) -> Tensor:
    return x.sum(dim=dim)


bmm = compile_bmm(mul=mul, sum=sum)
fill_unit = compile_fill_unit(zero=-float('inf'), one=0.)
tree_reduce = compile_tree_reduction(bmm=bmm)
