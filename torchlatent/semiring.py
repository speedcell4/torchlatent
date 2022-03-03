import torch
from torch import Tensor
from torchrua.scatter import scatter_add, scatter_max, scatter_mul, scatter_logsumexp
from torchrua.reduction import reduce_sequence, ReductionIndices

from torchlatent.functional import logsumexp, logaddexp

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
    def scatter_add(cls, tensor: Tensor, index: Tensor) -> Tensor:
        raise NotImplementedError

    @classmethod
    def scatter_mul(cls, tensor: Tensor, index: Tensor) -> Tensor:
        raise NotImplementedError

    @classmethod
    def bmm(cls, x: Tensor, y: Tensor) -> Tensor:
        return cls.sum(cls.mul(x[..., :, :, None], y[..., None, :, :]), dim=-2, keepdim=False)

    @classmethod
    def reduce(cls, tensor: Tensor, indices: ReductionIndices) -> Tensor:
        return reduce_sequence(cls.bmm)(tensor=tensor, indices=indices)


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
    def scatter_add(cls, tensor: Tensor, index: Tensor) -> Tensor:
        return scatter_add(tensor=tensor, index=index)

    @classmethod
    def scatter_mul(cls, tensor: Tensor, index: Tensor) -> Tensor:
        return scatter_mul(tensor=tensor, index=index)


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
    def scatter_add(cls, tensor: Tensor, index: Tensor) -> Tensor:
        return scatter_logsumexp(tensor=tensor, index=index)

    @classmethod
    def scatter_mul(cls, tensor: Tensor, index: Tensor) -> Tensor:
        return scatter_add(tensor=tensor, index=index)


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
    def scatter_add(cls, tensor: Tensor, index: Tensor) -> Tensor:
        return scatter_max(tensor=tensor, index=index)

    @classmethod
    def scatter_mul(cls, tensor: Tensor, index: Tensor) -> Tensor:
        return scatter_add(tensor=tensor, index=index)
