from abc import ABCMeta
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch import nn
from torch.nn import init
from torchrua import ReductionIndices, PackedSequence, CattedSequence
from torchrua import reduce_packed_indices, reduce_catted_indices

from torchlatent.crf.catting import CattedCrfDistribution
from torchlatent.crf.packing import PackedCrfDistribution

__all__ = [
    'CrfDecoderABC', 'CrfDecoder',
    'PackedCrfDistribution',
    'CattedCrfDistribution',
    'Sequence',
]

Sequence = Union[
    PackedSequence,
    CattedSequence,
]


class CrfDecoderABC(nn.Module, metaclass=ABCMeta):
    def __init__(self, num_tags: int, num_conjugates: int) -> None:
        super(CrfDecoderABC, self).__init__()

        self.num_tags = num_tags
        self.num_conjugates = num_conjugates

    def reset_parameters(self) -> None:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return ', '.join([
            f'num_tags={self.num_tags}',
            f'num_conjugates={self.num_conjugates}',
        ])

    @staticmethod
    def compile_indices(emissions: Sequence,
                        tags: Optional[Sequence] = None,
                        indices: Optional[ReductionIndices] = None, **kwargs):
        assert emissions.data.dim() == 3, f'{emissions.data.dim()} != {3}'
        if tags is not None:
            assert tags.data.dim() == 2, f'{tags.data.dim()} != {2}'

        if indices is None:
            if isinstance(emissions, PackedSequence):
                batch_sizes = emissions.batch_sizes.to(device=emissions.data.device)
                return reduce_packed_indices(batch_sizes=batch_sizes)

            if isinstance(emissions, CattedSequence):
                token_sizes = emissions.token_sizes.to(device=emissions.data.device)
                return reduce_catted_indices(token_sizes=token_sizes)

        return indices

    def obtain_parameters(self, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        return self.transitions, self.head_transitions, self.last_transitions

    def forward(self, emissions: Sequence, tags: Optional[Sequence] = None,
                indices: Optional[ReductionIndices] = None, **kwargs):
        indices = self.compile_indices(emissions=emissions, tags=tags, indices=indices)
        transitions, head_transitions, last_transitions = self.obtain_parameters(
            emissions=emissions, tags=tags, indices=indices,
        )

        if isinstance(emissions, PackedSequence):
            dist = PackedCrfDistribution(
                emissions=emissions, indices=indices,
                transitions=transitions,
                head_transitions=head_transitions,
                last_transitions=last_transitions,
            )
            return dist, tags

        if isinstance(emissions, CattedSequence):
            dist = CattedCrfDistribution(
                emissions=emissions, indices=indices,
                transitions=transitions,
                head_transitions=head_transitions,
                last_transitions=last_transitions,
            )
            return dist, tags

        raise TypeError(f'{type(emissions)} is not supported.')

    def fit(self, emissions: Sequence, tags: Sequence,
            indices: Optional[ReductionIndices] = None, **kwargs) -> Tensor:
        dist, tags = self(emissions=emissions, tags=tags, instr=indices, **kwargs)

        return dist.log_prob(tags=tags)

    def decode(self, emissions: Sequence,
               indices: Optional[ReductionIndices] = None, **kwargs) -> Sequence:
        dist, _ = self(emissions=emissions, tags=None, instr=indices, **kwargs)
        return dist.argmax

    def marginals(self, emissions: Sequence,
                  indices: Optional[ReductionIndices] = None, **kwargs) -> Tensor:
        dist, _ = self(emissions=emissions, tags=None, instr=indices, **kwargs)
        return dist.marginals


class CrfDecoder(CrfDecoderABC):
    def __init__(self, num_tags: int, num_conjugates: int = 1) -> None:
        super(CrfDecoder, self).__init__(num_tags=num_tags, num_conjugates=num_conjugates)

        self.transitions = nn.Parameter(torch.empty((1, self.num_conjugates, self.num_tags, self.num_tags)))
        self.head_transitions = nn.Parameter(torch.empty((1, self.num_conjugates, self.num_tags)))
        self.last_transitions = nn.Parameter(torch.empty((1, self.num_conjugates, self.num_tags)))

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, bound: float = 0.01) -> None:
        init.uniform_(self.transitions, -bound, +bound)
        init.uniform_(self.head_transitions, -bound, +bound)
        init.uniform_(self.last_transitions, -bound, +bound)
