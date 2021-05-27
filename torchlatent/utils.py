from typing import Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence


def broadcast_packed_sequences(
        emissions: PackedSequence, tags: Optional[PackedSequence],
        transitions: Tensor, start_transitions: Tensor, end_transitions: Tensor):
    """
        Args:
            emissions: [t1, c1, n]
            tags: [t1, c1]
            transitions: [t2, c2, n, n]
            start_transitions: [h2, c2, n]
            end_transitions: [h2, c2, n]
    """
    assert emissions.data.dim() == 3, f'{emissions.data.size()}'
    assert transitions.dim() == 4, f'{transitions.size()}'
    assert start_transitions.dim() == 3, f'{start_transitions.size()}'
    assert end_transitions.dim() == 3, f'{end_transitions.size()}'

    _, _, n = emissions.data.size()
    h = emissions.batch_sizes[0].item()

    if tags is None:
        t, c, = torch.broadcast_shapes(
            emissions.data.size()[:2],
            transitions.size()[:2],
        )
    else:
        assert tags.data.dim() == 2, f'{tags.data.size()}'

        t, c, = torch.broadcast_shapes(
            emissions.data.size()[:2],
            tags.data.size()[:2],
            transitions.size()[:2],
        )
        tags = tags._replace(data=tags.data.expand((t, c)))

    emissions = emissions._replace(data=emissions.data.expand((t, c, n)))
    transitions = transitions.expand((t, c, n, n))
    start_transitions = start_transitions.expand((h, c, n))
    end_transitions = end_transitions.expand((h, c, n))

    return emissions, tags, transitions, start_transitions, end_transitions, (t, c, n, h)
