import torch
from torch import Tensor

from torchlatent.semiring.abc import compile_fill_unit, compile_bmm, compile_tree_reduction


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


bmm = compile_bmm(mul=mul, sum=sum)
fill_unit = compile_fill_unit(zero=-float('inf'), one=0.)
tree_reduce = compile_tree_reduction(bmm=bmm)
