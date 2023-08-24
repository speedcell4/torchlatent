import torch
from torch import Tensor
from torchrua import segment_logsumexp
from torchrua import segment_max
from torchrua import segment_prod
from torchrua import segment_sum

from torchlatent.functional import logaddexp
from torchlatent.functional import logsumexp

__all__ = [
    'Semiring', 'ExceptionSemiring',
    'Std', 'Log', 'Max', 'Xen', 'Div',

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
    def segment_sum(cls, tensor: Tensor, sizes: Tensor) -> Tensor:
        raise NotImplementedError

    @classmethod
    def segment_prod(cls, tensor: Tensor, sizes: Tensor) -> Tensor:
        raise NotImplementedError

    @classmethod
    def bmm(cls, x: Tensor, y: Tensor) -> Tensor:
        return cls.sum(cls.mul(x[..., :, :, None], y[..., None, :, :]), dim=-2, keepdim=False)


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

    @classmethod
    def segment_sum(cls, tensor: Tensor, sizes: Tensor) -> Tensor:
        return segment_sum(tensor, segment_sizes=sizes)

    @classmethod
    def segment_prod(cls, tensor: Tensor, sizes: Tensor) -> Tensor:
        return segment_prod(tensor, segment_sizes=sizes)


class Log(Semiring):
    zero = -float('inf')
    one = 0.

    @classmethod
    def add(cls, x: Tensor, y: Tensor) -> Tensor:
        return logaddexp(x, y)

    @classmethod
    def mul(cls, x: Tensor, y: Tensor) -> Tensor:
        return x + y

    @classmethod
    def sum(cls, tensor: Tensor, dim: int, keepdim: bool = False) -> Tensor:
        return logsumexp(tensor, dim=dim, keepdim=keepdim)

    @classmethod
    def prod(cls, tensor: Tensor, dim: int, keepdim: bool = False) -> Tensor:
        return torch.sum(tensor, dim=dim, keepdim=keepdim)

    @classmethod
    def segment_sum(cls, tensor: Tensor, sizes: Tensor) -> Tensor:
        return segment_logsumexp(tensor, segment_sizes=sizes)

    @classmethod
    def segment_prod(cls, tensor: Tensor, sizes: Tensor) -> Tensor:
        return segment_sum(tensor, segment_sizes=sizes)


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
        return torch.max(tensor, dim=dim, keepdim=keepdim).values

    @classmethod
    def prod(cls, tensor: Tensor, dim: int, keepdim: bool = False) -> Tensor:
        return torch.sum(tensor, dim=dim, keepdim=keepdim)

    @classmethod
    def segment_sum(cls, tensor: Tensor, sizes: Tensor) -> Tensor:
        return segment_max(tensor, segment_sizes=sizes)

    @classmethod
    def segment_prod(cls, tensor: Tensor, sizes: Tensor) -> Tensor:
        return segment_sum(tensor, segment_sizes=sizes)


class ExceptionSemiring(Semiring):
    @classmethod
    def sum(cls, tensor: Tensor, log_p: Tensor, log_q: Tensor, dim: int, keepdim: bool = False) -> Tensor:
        raise NotImplementedError

    @classmethod
    def segment_sum(cls, tensor: Tensor, log_p: Tensor, log_q: Tensor, sizes: Tensor) -> Tensor:
        raise NotImplementedError


class Xen(ExceptionSemiring):
    zero = 0.
    one = 0.

    @classmethod
    def add(cls, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError

    @classmethod
    def mul(cls, x: Tensor, y: Tensor) -> Tensor:
        return x + y

    @classmethod
    def sum(cls, tensor: Tensor, log_p: Tensor, log_q: Tensor, dim: int, keepdim: bool = False) -> Tensor:
        return torch.sum((tensor - log_q) * log_p.exp(), dim=dim, keepdim=keepdim)

    @classmethod
    def prod(cls, tensor: Tensor, dim: int, keepdim: bool = False) -> Tensor:
        return torch.sum(tensor, dim=dim, keepdim=keepdim)

    @classmethod
    def segment_sum(cls, tensor: Tensor, log_p: Tensor, log_q: Tensor, sizes: Tensor) -> Tensor:
        return segment_sum((tensor - log_q) * log_p.exp(), segment_sizes=sizes)

    @classmethod
    def segment_prod(cls, tensor: Tensor, sizes: Tensor) -> Tensor:
        return segment_sum(tensor, segment_sizes=sizes)


class Div(ExceptionSemiring):
    zero = 0.
    one = 0.

    @classmethod
    def add(cls, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError

    @classmethod
    def mul(cls, x: Tensor, y: Tensor) -> Tensor:
        return x + y

    @classmethod
    def sum(cls, tensor: Tensor, log_p: Tensor, log_q: Tensor, dim: int, keepdim: bool = False) -> Tensor:
        return torch.sum((tensor - log_q + log_p) * log_p.exp(), dim=dim, keepdim=keepdim)

    @classmethod
    def prod(cls, tensor: Tensor, dim: int, keepdim: bool = False) -> Tensor:
        return torch.sum(tensor, dim=dim, keepdim=keepdim)

    @classmethod
    def segment_sum(cls, tensor: Tensor, log_p: Tensor, log_q: Tensor, sizes: Tensor) -> Tensor:
        return segment_sum((tensor - log_q + log_p) * log_p.exp(), segment_sizes=sizes)

    @classmethod
    def segment_prod(cls, tensor: Tensor, sizes: Tensor) -> Tensor:
        return segment_sum(tensor, segment_sizes=sizes)
