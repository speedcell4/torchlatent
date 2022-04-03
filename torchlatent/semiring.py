import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device
from torchrua import cat_packed_indices, cat_padded_indices, CattedSequence
from torchrua.reduction import reduce_sequence, ReductionIndices
from torchrua.scatter import scatter_add, scatter_logsumexp

from torchlatent.functional import logsumexp, logaddexp
from torchlatent.types import Sequence

__all__ = [
    'Semiring',
    'Std', 'Log', 'Max',
]


@torch.no_grad()
def segment_indices(sequence: Sequence, batch_first: bool = True, device: Device = None):
    if isinstance(sequence, CattedSequence):
        data, token_sizes = sequence
        return segment_catted_indices(token_sizes=token_sizes, device=data.device)

    if isinstance(sequence, PackedSequence):
        data, batch_sizes, _, unsorted_indices = sequence
        return segment_packed_indices(batch_sizes=batch_sizes, unsorted_indices=unsorted_indices, device=data.device)

    if isinstance(sequence, tuple) and torch.is_tensor(sequence[0]) and torch.is_tensor(sequence[1]):
        data, token_sizes = sequence
        return segment_padded_indices(token_sizes=token_sizes, batch_first=batch_first, device=device)

    raise KeyError(f'type {type(sequence)} is not supported')


@torch.no_grad()
def segment_catted_indices(token_sizes: Tensor, device: Device = None):
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)

    batch_ptr = torch.repeat_interleave(repeats=token_sizes)
    return ..., batch_ptr, token_sizes


@torch.no_grad()
def segment_packed_indices(batch_sizes: Tensor, unsorted_indices: Tensor, device: Device = None):
    if device is None:
        if unsorted_indices is not None:
            device = unsorted_indices.device
        else:
            device = batch_sizes.device

    batch_sizes = batch_sizes.to(device=device)
    unsorted_indices = unsorted_indices.to(device=device)

    indices, token_sizes = cat_packed_indices(
        batch_sizes=batch_sizes,
        unsorted_indices=unsorted_indices,
        device=device,
    )
    batch_ptr = torch.repeat_interleave(repeats=token_sizes)
    return indices, batch_ptr, token_sizes


@torch.no_grad()
def segment_padded_indices(token_sizes: Tensor, batch_first: bool, device: Device = None):
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)

    if batch_first:
        (batch_ptr, token_ptr), _ = cat_padded_indices(
            token_sizes=token_sizes, batch_first=batch_first, device=device)
        return (batch_ptr, token_ptr), batch_ptr, token_sizes
    else:
        (token_ptr, batch_ptr), _ = cat_padded_indices(
            token_sizes=token_sizes, batch_first=batch_first, device=device)
        return (token_ptr, batch_ptr), batch_ptr, token_sizes


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

    @classmethod
    def reduce(cls, tensor: Tensor, indices: ReductionIndices) -> Tensor:
        return reduce_sequence(data=tensor, indices=indices, op=cls.bmm)


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
        return torch.segment_reduce(tensor, reduce='sum', lengths=sizes, unsafe=True)

    @classmethod
    def segment_prod(cls, tensor: Tensor, sizes: Tensor) -> Tensor:
        raise NotImplementedError


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

    @classmethod
    def segment_sum(cls, tensor: Tensor, sizes: Tensor) -> Tensor:
        m = torch.segment_reduce(tensor, reduce='max', lengths=sizes, unsafe=True).detach()
        z = (tensor - torch.repeat_interleave(m, repeats=sizes)).exp()
        return torch.segment_reduce(z, reduce='sum', lengths=sizes, unsafe=True).log() + m

    @classmethod
    def segment_prod(cls, tensor: Tensor, sizes: Tensor) -> Tensor:
        return torch.segment_reduce(tensor, reduce='sum', lengths=sizes, unsafe=True)


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
        return torch.segment_reduce(tensor, reduce='max', lengths=sizes, unsafe=True)

    @classmethod
    def segment_prod(cls, tensor: Tensor, sizes: Tensor) -> Tensor:
        return torch.segment_reduce(tensor, reduce='sum', lengths=sizes, unsafe=True)
