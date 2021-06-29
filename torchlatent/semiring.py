import torch
from torch import Tensor

from torchlatent.functional import logsumexp

__all__ = [
    'Semiring',
    'Std', 'Log', 'Max',
]


class Semiring(object):
    zero: float
    one: float

    @classmethod
    def eye_like(cls, tensor: Tensor) -> Tensor:
        *_, n = tensor.size()
        eye = torch.full((n, n), fill_value=cls.zero, dtype=tensor.dtype, device=tensor.device)
        index = torch.arange(n, dtype=torch.long, device=tensor.device)
        eye[index, index] = cls.one
        return eye

    @classmethod
    def add(cls, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError

    @classmethod
    def mul(cls, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError

    @classmethod
    def sum(cls, tensor: Tensor, dim: int, keepdim: bool = False) -> Tensor:
        raise NotImplementedError

    @classmethod
    def prod(cls, tensor: Tensor, dim: int, keepdim: bool = False) -> Tensor:
        raise NotImplementedError

    @classmethod
    def bmm(cls, x: Tensor, y: Tensor) -> Tensor:
        return cls.sum(cls.mul(x.unsqueeze(-1), y.unsqueeze(-3)), dim=-2, keepdim=False)


class Std(Semiring):
    zero = 0.
    one = 1.

    @classmethod
    def add(cls, x: Tensor, y: Tensor) -> Tensor:
        return x + y

    @classmethod
    def mul(cls, x: Tensor, y: Tensor) -> Tensor:
        return x * y

    @classmethod
    def sum(cls, tensor: Tensor, dim: int, keepdim: bool = False) -> Tensor:
        return torch.sum(tensor, dim=dim, keepdim=keepdim)

    @classmethod
    def prod(cls, tensor: Tensor, dim: int, keepdim: bool = False) -> Tensor:
        return torch.prod(tensor, dim=dim, keepdim=keepdim)


class Log(Semiring):
    zero = -float('inf')
    one = 0.

    @classmethod
    def add(cls, x: Tensor, y: Tensor) -> Tensor:
        with torch.no_grad():
            m = torch.maximum(x, y)
            m = m.masked_fill_(torch.isneginf(m), 0.)

        z = (x - m).exp_() + (y - m).exp_()
        mask = z == 0
        z = z.masked_fill_(mask, 1.).log_()
        return z.masked_fill_(mask, -float('inf')).add_(m)

    @classmethod
    def mul(cls, x: Tensor, y: Tensor) -> Tensor:
        return x + y

    @classmethod
    def sum(cls, tensor: Tensor, dim: int, keepdim: bool = False) -> Tensor:
        return logsumexp(tensor, dim=dim, keepdim=keepdim)

    @classmethod
    def prod(cls, tensor: Tensor, dim: int, keepdim: bool = False) -> Tensor:
        return torch.sum(tensor, dim=dim, keepdim=keepdim)


class Max(Semiring):
    zero = -float('inf')
    one = 0.

    @classmethod
    def add(cls, x: Tensor, y: Tensor) -> Tensor:
        return torch.maximum(x, y)

    @classmethod
    def mul(cls, x: Tensor, y: Tensor) -> Tensor:
        return x + y

    @classmethod
    def sum(cls, tensor: Tensor, dim: int, keepdim: bool = False) -> Tensor:
        return torch.max(tensor, dim=dim, keepdim=keepdim)[0]

    @classmethod
    def prod(cls, tensor: Tensor, dim: int, keepdim: bool = False) -> Tensor:
        return torch.sum(tensor, dim=dim, keepdim=keepdim)
