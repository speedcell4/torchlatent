from abc import ABCMeta
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.nn import init
from torch.nn.utils.rnn import PackedSequence
from torchrua import TreeReduceIndices, tree_reduce_packed_indices

from torchlatent.crf.packing import PackedCrfDistribution

__all__ = {
    'CrfDecoderABC', 'CrfDecoder',
}


class CrfDecoderABC(nn.Module, metaclass=ABCMeta):
    def __init__(self, num_tags: int, num_conjugates: int):
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

    def compile_indices(self, emissions: PackedSequence, tags: Optional[PackedSequence] = None,
                        indices: Optional[TreeReduceIndices] = None, **kwargs):
        assert emissions.data.dim() == 3, f'{emissions.data.dim()} != {3}'
        if tags is not None:
            assert tags.data.dim() == 2, f'{tags.data.dim()} != {2}'

        if indices is None:
            batch_sizes = emissions.batch_sizes.to(device=emissions.data.device)
            indices = tree_reduce_packed_indices(batch_sizes=batch_sizes)

        return indices

    def obtain_parameters(self, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        return self.transitions, self.head_transitions, self.tail_transitions

    def forward(self, emissions: PackedSequence, tags: Optional[PackedSequence] = None,
                indices: Optional[TreeReduceIndices] = None, **kwargs):
        indices = self.compile_indices(emissions=emissions, tags=tags, indices=indices)
        transitions, head_transitions, tail_transitions = self.obtain_parameters(
            emissions=emissions, tags=tags, indices=indices,
        )

        dist = PackedCrfDistribution(
            emissions=emissions, indices=indices,
            transitions=transitions,
            head_transitions=head_transitions,
            tail_transitions=tail_transitions,
        )

        return dist, tags

    def fit(self, emissions: PackedSequence, tags: PackedSequence,
            indices: Optional[TreeReduceIndices] = None, **kwargs) -> Tensor:
        dist, tags = self(emissions=emissions, tags=tags, instr=indices, **kwargs)

        return dist.log_prob(tags=tags)

    def decode(self, emissions: PackedSequence,
               indices: Optional[TreeReduceIndices] = None, **kwargs) -> PackedSequence:
        dist, _ = self(emissions=emissions, tags=None, instr=indices, **kwargs)
        return dist.argmax

    def marginals(self, emissions: PackedSequence,
                  indices: Optional[TreeReduceIndices] = None, **kwargs) -> Tensor:
        dist, _ = self(emissions=emissions, tags=None, instr=indices, **kwargs)
        return dist.marginals


class CrfDecoder(CrfDecoderABC):
    def __init__(self, num_tags: int, num_conjugates: int = 1) -> None:
        super(CrfDecoder, self).__init__(num_tags=num_tags, num_conjugates=num_conjugates)

        self.transitions = nn.Parameter(
            torch.empty((1, self.num_conjugates, self.num_tags, self.num_tags)),
            requires_grad=True,
        )
        self.head_transitions = nn.Parameter(
            torch.empty((1, self.num_conjugates, self.num_tags)),
            requires_grad=True,
        )
        self.tail_transitions = nn.Parameter(
            torch.empty((1, self.num_conjugates, self.num_tags)),
            requires_grad=True,
        )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        bound = 0.01
        init.uniform_(self.transitions, -bound, +bound)
        init.uniform_(self.head_transitions, -bound, +bound)
        init.uniform_(self.tail_transitions, -bound, +bound)
