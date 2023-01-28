from abc import ABCMeta
from functools import singledispatch
from typing import Tuple, NamedTuple
from typing import Type

import torch
from torch import Tensor
from torch import nn
from torch.distributions.utils import lazy_property
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchlatent.abc import DistributionABC
from torchlatent.nn.classifier import BiaffineClassifier
from torchlatent.semiring import Semiring, Log, Max
from torchlatent.types import Sequence
from torchrua import CattedSequence, transpose_sizes, pack_catted_sequence, cat_packed_indices, RuaSequential
from torchrua import major_sizes_to_ptr, accumulate_sizes
from torchrua import pad_packed_sequence, pad_catted_sequence


@singledispatch
def cky_scores_indices(sequence: Sequence, device: Device = None):
    raise KeyError(f'type {type(sequence)} is not supported')


@cky_scores_indices.register
def cky_scores_catted_indices(sequence: CattedSequence, device: Device = None):
    if device is None:
        device = sequence.data.device

    token_sizes = sequence.token_sizes.to(device=device)

    batch_ptr = torch.repeat_interleave(repeats=token_sizes)
    return ..., batch_ptr, token_sizes


@cky_scores_indices.register
def cky_scores_packed_indices(sequence: PackedSequence, device: Device = None):
    if device is None:
        device = sequence.data.device

    batch_sizes = sequence.batch_sizes.to(device=device)
    unsorted_indices = sequence.unsorted_indices.to(device=device)

    indices, token_sizes = cat_packed_indices(
        batch_sizes=batch_sizes,
        unsorted_indices=unsorted_indices,
        device=device,
    )

    batch_ptr = torch.repeat_interleave(repeats=token_sizes)
    return indices, batch_ptr, token_sizes


class CkyIndices(NamedTuple):
    token_size: int
    cache_size: int

    src: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]
    tgt: Tuple[Tensor, Tensor]


@torch.no_grad()
def cky_partition_indices(token_sizes: Tensor, device: Device = None):
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)
    acc_token_sizes = accumulate_sizes(sizes=token_sizes)

    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)
    x_ptr, z_ptr = major_sizes_to_ptr(sizes=token_ptr + 1)
    y_ptr = token_ptr[z_ptr]

    token_size = token_sizes.max().item()
    cache_size, = token_ptr.size()

    return CkyIndices(
        token_size=token_size,
        cache_size=cache_size,
        src=((y_ptr - x_ptr, x_ptr + z_ptr - y_ptr), (batch_ptr[z_ptr], x_ptr, y_ptr)),
        tgt=(token_sizes - 1, acc_token_sizes),
    )


def cky_partition(data: Tensor, indices: CkyIndices, *, semiring: Type[Semiring]) -> Tensor:
    token_size, cache_size, (src1, src2), tgt = indices

    size = (token_size, cache_size, *data.size()[3:])
    tensor0 = torch.full(size, fill_value=semiring.zero, requires_grad=False)
    tensor1 = torch.full(size, fill_value=semiring.zero, requires_grad=False)
    tensor2 = torch.full(size, fill_value=semiring.zero, requires_grad=False)

    tensor0[src1] = data[src2]
    tensor1[0, :] = tensor2[-1, :] = tensor0[0, :]

    for w in range(1, token_size):
        tensor1[w, :-w] = tensor2[-w - 1, w:] = semiring.mul(
            semiring.sum(semiring.mul(tensor1[:w, :-w], tensor2[-w:, w:]), dim=0),
            tensor0[w, :-w],
        )

    return tensor1[tgt]


class CkyDistribution(DistributionABC):
    def __init__(self, emissions: Tensor, indices: CkyIndices) -> None:
        super(CkyDistribution, self).__init__(validate_args=False)

        self.emissions = emissions
        self.indices = indices

    def log_scores(self, sequence: Sequence) -> Tensor:
        indices, batch_ptr, sizes = cky_scores_indices(sequence)
        data = sequence.data[indices]
        return Log.segment_prod(
            tensor=self.emissions[batch_ptr, data[..., 0], data[..., 1], data[..., 2]],
            sizes=sizes,
        )

    @lazy_property
    def log_partitions(self) -> Tensor:
        return cky_partition(data=Log.sum(self.emissions, dim=-1), indices=self.indices, semiring=Log)

    @lazy_property
    def max(self) -> Tensor:
        return cky_partition(data=Max.sum(self.emissions, dim=-1), indices=self.indices, semiring=Max)

    @lazy_property
    def argmax(self) -> Tensor:
        mask = super(CkyDistribution, self).argmax > 0
        b, n, _, m = mask.size()

        index = torch.arange(n, device=mask.device)
        x = torch.masked_select(index[None, :, None, None], mask=mask)
        y = torch.masked_select(index[None, None, :, None], mask=mask)

        index = torch.arange(m, device=mask.device)
        z = torch.masked_select(index[None, None, None, :], mask=mask)
        return torch.stack([x, y, z], dim=-1)

    @lazy_property
    def marginals(self) -> Tensor:
        grad, = torch.autograd.grad(
            self.log_partitions, self.emissions, torch.ones_like(self.log_partitions),
            create_graph=True, only_inputs=True, allow_unused=False,
        )
        return grad

    @lazy_property
    def entropy(self) -> Tensor:
        raise NotImplementedError


class CkyLayerABC(nn.Module, metaclass=ABCMeta):
    def reset_parameters(self) -> None:
        raise NotImplementedError

    def forward_scores(self, features: Tensor, *args, **kwargs) -> Tensor:
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
            indices = cky_partition_indices(token_sizes=token_sizes, device=features.device)

        return CkyDistribution(
            emissions=self.forward_scores(features=features),
            indices=indices,
        )

    def fit(self, sequence: Sequence, targets: Sequence, indices: CkyIndices = None) -> Tensor:
        dist = self.forward(sequence=sequence, indices=indices)
        return dist.log_partitions - dist.log_scores(sequence=targets)

    def decode(self, sequence: Sequence, indices: CkyIndices = None) -> Sequence:
        dist = self.forward(sequence=sequence, indices=indices)

        if isinstance(sequence, CattedSequence):
            return CattedSequence(data=dist.argmax, token_sizes=sequence.token_sizes * 2 - 1)

        if isinstance(sequence, PackedSequence):
            token_sizes = transpose_sizes(sizes=sequence.batch_sizes)
            token_sizes = token_sizes[sequence.unsorted_indices] * 2 - 1
            return pack_catted_sequence(CattedSequence(data=dist.argmax, token_sizes=token_sizes))

        raise KeyError(f'type {type(sequence)} is not supported')


class CkyLayer(CkyLayerABC):
    def __init__(self, num_targets: int, num_conjugates: int) -> None:
        super(CkyLayer, self).__init__()

        self.num_targets = num_targets
        self.num_conjugates = num_conjugates

    def extra_repr(self) -> str:
        return ', '.join([
            f'num_targets={self.num_targets}',
            f'num_conjugates={self.num_conjugates}',
        ])


class CkyDecoder(nn.Module):
    def __init__(self, in_features: int, hidden_features: int,
                 num_targets: int, num_conjugates: int, dropout: float) -> None:
        super(CkyDecoder, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_targets = num_targets
        self.num_conjugates = num_conjugates

        self.ffn1 = RuaSequential(
            nn.Linear(in_features, hidden_features, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ffn2 = RuaSequential(
            nn.Linear(in_features, hidden_features, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = BiaffineClassifier(
            num_conjugates=num_conjugates,
            in_features1=hidden_features,
            in_features2=hidden_features,
            out_features=num_targets,
            bias=False,
        )

        self.cky = CkyLayer(
            num_targets=num_targets,
            num_conjugates=num_conjugates,
        )

    def forward(self, sequence: Sequence) -> CkyDistribution:
        features1, _ = self.ffn1(sequence)
        features2, _ = self.ffn2(sequence)

        emissions = self.classifier(features1, features2)
        return self.cky(emissions)
