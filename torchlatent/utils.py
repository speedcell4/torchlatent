from typing import Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence


def broadcast_packed_sequences(
        emissions: PackedSequence, tags: Optional[PackedSequence],
        transitions: Tensor, head_transitions: Tensor, tail_transitions: Tensor):
    """
        Args:
            emissions: [t1, c1, n]
            tags: [t1, c1]
            transitions: [t2, c2, n, n]
            head_transitions: [h2, c2, n]
            tail_transitions: [h2, c2, n]
    """
    assert emissions.data.dim() == 3, f'{emissions.data.size()}'
    assert transitions.dim() == 4, f'{transitions.size()}'
    assert head_transitions.dim() == 3, f'{head_transitions.size()}'
    assert tail_transitions.dim() == 3, f'{tail_transitions.size()}'

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
    head_transitions = head_transitions.expand((h, c, n))
    tail_transitions = tail_transitions.expand((h, c, n))

    return emissions, tags, transitions, head_transitions, tail_transitions, (t, c, n, h)
