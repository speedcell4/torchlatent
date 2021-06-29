import torch
from torch import Tensor

__all__ = [
    'logsumexp',
]


def logsumexp(tensor: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    with torch.no_grad():
        m, _ = torch.max(tensor, dim=dim, keepdim=True)
        m = m.masked_fill_(torch.isneginf(m), 0.)

    z = (tensor - m).exp_().sum(dim=dim, keepdim=True)
    mask = z == 0
    z = z.masked_fill_(mask, 1.).log_()
    z = z.masked_fill_(mask, -float('inf')).add_(m)

    if not keepdim:
        z = z.squeeze(dim=dim)
    return z
