from abc import ABCMeta
from typing import Tuple, NamedTuple
from typing import Type

import torch
from torch import Tensor
from torch import nn
from torch.distributions.utils import lazy_property
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device
from torchrua import CattedSequence
from torchrua import major_sizes_to_ptr, accumulate_sizes
from torchrua import pad_packed_sequence, pad_catted_sequence

from torchlatent.abc import DistributionABC
from torchlatent.semiring import Semiring, Log, Max
from torchlatent.types import Sequence


class CkyIndices(NamedTuple):
    token_size: int
    cache_size: int

    src: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]
    tgt: Tuple[Tensor, Tensor]


@torch.no_grad()
def cky_indices(token_sizes: Tensor, device: Device = None):
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)
    acc_token_sizes = accumulate_sizes(sizes=token_sizes)

    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)
    x_ptr, z_ptr = major_sizes_to_ptr(sizes=token_ptr + 1)
    batch_ptr = batch_ptr[z_ptr]
    y_ptr = z_ptr - acc_token_sizes[batch_ptr]

    token_size = token_sizes.max().item()
    cache_size, = token_ptr.size()

    return CkyIndices(
        token_size=token_size, cache_size=cache_size,
        src=((y_ptr - x_ptr, z_ptr), (batch_ptr, x_ptr, y_ptr)),
        tgt=(token_sizes - 1, acc_token_sizes),
    )


def cky_partition(data: Tensor, indices: CkyIndices, semiring: Type[Semiring]) -> Tensor:
    token_size, cache_size, (src1, src2), tgt = indices

    tensor0 = torch.full((token_size, cache_size, *data.size()[3:]), fill_value=semiring.zero, requires_grad=False)
    tensor1 = torch.full((token_size, cache_size, *data.size()[3:]), fill_value=semiring.zero, requires_grad=False)
    tensor2 = torch.full((token_size, cache_size, *data.size()[3:]), fill_value=semiring.zero, requires_grad=False)

    tensor0[src1] = data[src2]
    tensor1[0, :] = tensor2[-1, :] = tensor0[0, :]

    for w in range(1, token_size):
        tensor1[w, :-w] = tensor2[-w - 1, w:] = semiring.mul(
            semiring.sum(semiring.mul(tensor1[:w, :-w], tensor2[-w:, w:]), dim=0),
            tensor0[w, w:],
        )

    return tensor1[tgt]


class CkyDistribution(DistributionABC):
    def __init__(self, scores: Tensor, indices: CkyIndices) -> None:
        super(CkyDistribution, self).__init__(validate_args=False)

        self.scores = scores
        self.indices = indices

    def log_scores(self, value: Sequence) -> Tensor:
        raise NotImplementedError

    @lazy_property
    def log_partitions(self) -> Tensor:
        return cky_partition(data=Log.sum(self.scores, dim=-1), indices=self.indices, semiring=Log)

    @lazy_property
    def max(self) -> Tensor:
        return cky_partition(data=Max.sum(self.scores, dim=-1), indices=self.indices, semiring=Max)

    @lazy_property
    def argmax(self) -> Tensor:
        mask = super(CkyDistribution, self).argmax > 0
        b, n, _, m = self.scores

        index = torch.arange(n, device=mask.device)
        x = torch.masked_select(index[None, :, None, None], mask=mask)
        y = torch.masked_select(index[None, None, :, None], mask=mask)

        index = torch.arange(m, device=mask.device)
        z = torch.masked_select(index[None, None, None, :], mask=mask)
        return torch.stack([x, y, z], dim=0)

    @lazy_property
    def marginals(self) -> Tensor:
        grad, = torch.autograd.grad(
            self.log_partitions, self.scores, torch.ones_like(self.log_partitions),
            create_graph=True, only_inputs=True, allow_unused=False,
        )
        return grad

    @lazy_property
    def entropy(self) -> Tensor:
        raise NotImplementedError


class CkyDecoderABC(nn.Module, metaclass=ABCMeta):
    def reset_parameters(self) -> None:
        raise NotImplementedError

    def extra_repr(self) -> str:
        raise NotImplementedError

    def forward_scores(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def forward(self, sequence: Sequence, indices: CkyIndices = None) -> CkyDistribution:
        if isinstance(sequence, CattedSequence):
            features, token_sizes = pad_catted_sequence(sequence, batch_first=True)
        elif isinstance(sequence, PackedSequence):
            features, token_sizes = pad_packed_sequence(sequence, batch_first=True)
        elif isinstance(sequence, tuple) and torch.tensor(sequence[0]) and torch.is_tensor(sequence[1]):
            features, token_sizes = sequence
        else:
            raise KeyError(f'type {type(sequence)} is not supported')

        if indices is None:
            indices = cky_indices(token_sizes=token_sizes, device=features.device)

        return CkyDistribution(
            scores=self.forward_scores(features=features),
            indices=indices,
        )


class CkyDecoder(CkyDecoderABC):
    def __init__(self, in_features: int, bias: bool = True) -> None:
        super(CkyDecoder, self).__init__()

        self.fc1 = nn.Linear(in_features, in_features, bias=bias)
        self.fc2 = nn.Linear(in_features, in_features, bias=bias)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def extra_repr(self) -> str:
        return ', '.join([
            f'in_features={self.fc1.in_features}',
            f'bias={self.fc1.bias is not None}',
        ])

    def forward_scores(self, features: Tensor, *args, **kwargs) -> Tensor:
        x = self.fc1(features[..., :, None, :])
        y = self.fc2(features[..., None, :, :])
        return (x[..., None, :] @ y[..., :, None])[..., 0, 0]

