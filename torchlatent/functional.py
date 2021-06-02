import torch
from torch import Tensor


def logsumexp(x: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    with torch.no_grad():
        m, _ = torch.max(x, dim=dim, keepdim=True)
        mask = torch.isneginf(m)
        m = m.masked_fill_(mask, 0.)

    z = (x - m).exp_().sum(dim=dim, keepdim=True)
    mask = z == 0
    z = z.masked_fill_(mask, 1.).log_()
    z = z.masked_fill_(mask, -float('inf')).add_(m)

    if not keepdim:
        z = z.squeeze(dim=dim)
    return z
