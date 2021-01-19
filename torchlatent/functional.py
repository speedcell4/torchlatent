import torch
from torch import Tensor


def logsumexp(x: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    with torch.no_grad():
        m, _ = torch.max(x, dim=dim, keepdim=True)
        m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    z = (x - m).exp().sum(dim=dim, keepdim=True)
    mask = z == 0
    z = torch.where(mask, torch.ones_like(z), z).log()
    z = torch.where(mask, torch.full_like(z, -float('inf')), z) + m

    if not keepdim:
        z = z.squeeze(dim=dim)
    return z
