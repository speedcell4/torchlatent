import torch
from torch import Tensor

from torchlatent.functional import logsumexp
from torchlatent.semiring.abc import compile_bmm, compile_tree_reduction


def convert(x: Tensor) -> Tensor:
    return torch.stack([x, torch.zeros_like(x)], dim=1)


def unconvert(x: Tensor) -> Tensor:
    return x[:, 1]


def mul(lhs: Tensor, rhs: Tensor) -> Tensor:
    return lhs + rhs


def sum(x: Tensor, dim: int) -> Tensor:
    log_z = logsumexp(x[:, 0], dim=dim, keepdim=True)
    log_prob = x[:, 0] - log_z
    prob = log_prob.exp()

    h = x[:, 1] - log_prob
    mask = torch.isinf(h) & (prob == 0)
    h = torch.where(mask, torch.zeros_like(h), h)

    h = torch.sum(h * prob, dim=dim)
    return torch.stack([log_z.squeeze(dim=dim), h], dim=1)


bmm = compile_bmm(mul=mul, sum=sum)


@torch.no_grad()
def fill_unit(x: Tensor) -> Tensor:
    eye = torch.eye(x.size(-1), device=x.device, dtype=torch.float32)
    zeros = torch.zeros_like(eye)
    return torch.stack([eye.log(), zeros], dim=0)


tree_reduce = compile_tree_reduction(bmm=bmm)

if __name__ == '__main__':
    x = convert(torch.randn((7, 3, 3)))
    y = convert(torch.randn((7, 3, 3)))
    print(bmm(x, y))
