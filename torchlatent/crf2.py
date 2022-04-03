from typing import Tuple, Sequence, Type

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device
from torchrua import roll_catted_indices, CattedSequence, head_catted_indices, last_catted_indices, head_packed_indices, \
    last_packed_indices, accumulate_sizes

from torchlatent.semiring import segment_catted_indices, segment_packed_indices, Semiring


@torch.no_grad()
def broadcast_catted_shapes(sequence: CattedSequence, transitions: Tuple[Tensor, Tensor, Tensor]):
    sequence, token_sizes = sequence
    transitions, head_transitions, last_transitions = transitions

    t1, c1, *_ = sequence.size()
    h1, = token_sizes.size()

    t2, c2, _, _ = transitions.size()
    h3, c3, _ = head_transitions.size()
    h4, c4, _ = last_transitions.size()

    return torch.broadcast_shapes((t1, c1, h1), (t2, c2, 1), (1, c3, h3), (1, c4, h4))


@torch.no_grad()
def broadcast_packed_shapes(sequence: PackedSequence, transitions: Tuple[Tensor, Tensor, Tensor]):
    sequence, batch_sizes, _, _ = sequence
    transitions, head_transitions, last_transitions = transitions

    t1, c1, *_ = sequence.size()
    h1 = batch_sizes[0].item()

    t2, c2, _, _ = transitions.size()
    h3, c3, _ = head_transitions.size()
    h4, c4, _ = last_transitions.size()

    return torch.broadcast_shapes((t1, c1, h1), (t2, c2, 1), (1, c3, h3), (1, c4, h4))


@torch.no_grad()
def crf_segment_catted_indices(token_sizes: Tensor, device: Device = None):
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)

    curr, _, token_sizes = segment_catted_indices(token_sizes=token_sizes, device=device)
    prev = roll_catted_indices(token_sizes=token_sizes, shifts=1, device=device)
    head = head_catted_indices(token_sizes=token_sizes, device=device)
    last = last_catted_indices(token_sizes=token_sizes, device=device)
    return prev, curr, torch.arange(token_sizes.size()[0], device=device), head, last, token_sizes


@torch.no_grad()
def crf_segment_packed_indices(batch_sizes: Tensor, unsorted_indices: Tensor, device: Device):
    if device is None:
        if unsorted_indices is not None:
            device = unsorted_indices.device
        else:
            device = batch_sizes.device

    batch_sizes = batch_sizes.to(device=device)
    unsorted_indices = unsorted_indices.to(device=device)

    curr, _, token_sizes = segment_packed_indices(
        batch_sizes=batch_sizes, unsorted_indices=unsorted_indices, device=device,
    )
    prev = roll_catted_indices(token_sizes=token_sizes, shifts=1, device=device)
    head = head_packed_indices(token_sizes=token_sizes, unsorted_indices=unsorted_indices, device=device)
    last = last_packed_indices(token_sizes=token_sizes, unsorted_indices=unsorted_indices, device=device)
    return curr[prev], curr, unsorted_indices, head, last, token_sizes


def crf_segment_reduce(emissions: Sequence, targets: Sequence,
                       transitions: Tuple[Tensor, Tensor, Tensor], semiring: Type[Semiring]) -> Tensor:
    if isinstance(emissions, CattedSequence):
        emissions, token_sizes = emissions
        prev, curr, unsorted_indices, head, last, sizes = crf_segment_catted_indices(
            token_sizes=token_sizes, device=emissions.device,
        )
    elif isinstance(emissions, PackedSequence):
        emissions, batch_sizes, _, unsorted_indices = emissions
        prev, curr, unsorted_indices, head, last, sizes = crf_segment_packed_indices(
            batch_sizes=batch_sizes, unsorted_indices=unsorted_indices, device=emissions.device,
        )
    else:
        raise NotImplementedError

    if isinstance(targets, CattedSequence):
        t, c, h = broadcast_catted_shapes(targets, transitions=transitions)
        targets, _ = targets
    elif isinstance(targets, PackedSequence):
        t, c, h = broadcast_packed_shapes(targets, transitions=transitions)
        targets, _, _, _ = targets
    else:
        raise NotImplementedError

    c = torch.arange(c, device=emissions.device)

    transitions, head_transitions, last_transitions = transitions
    emissions = emissions.expand((t, c, -1))
    targets = targets.expand((t, c))
    transitions = transitions.expand((t, c, -1, -1))
    head_transitions = head_transitions.expand((h, c, -1))
    last_transitions = last_transitions.expand((h, c, -1))

    emissions = emissions[curr[:, None], c[None, :], targets[curr]]
    transitions = transitions[curr[:, None], c[None, :], targets[prev], targets[curr]]
    transitions[accumulate_sizes(sizes=sizes)] = semiring.one
    head_transitions = head_transitions[unsorted_indices[:, None], c[None, :], targets[head]]
    last_transitions = last_transitions[unsorted_indices[:, None], c[None, :], targets[last]]

    emissions = semiring.segment_prod(semiring.mul(emissions, transitions), sizes=sizes)
    return semiring.mul(emissions, semiring.mul(head_transitions, last_transitions))
