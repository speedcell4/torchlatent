from typing import List

import torch
from einops import rearrange
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_sequence
from torchrua import stack_packed_sequences

from tests.test_crf import assert_equal
from torchlatent import CrfDecoder


def gen_lengths(batch_size: int, total_length: int) -> List[int]:
    lengths = torch.randint(0, total_length, (batch_size,), dtype=torch.long) + 1
    return lengths.detach().cpu().tolist()


def gen_emissions(lengths: List[int], num_tags: int):
    return pack_sequence([
        torch.randn((length, num_tags), dtype=torch.float32, requires_grad=True)
        for length in lengths
    ], enforce_sorted=False)


def gen_tags(lengths: List[int], num_tags: int):
    return pack_sequence([
        torch.randint(0, num_tags, (length,), dtype=torch.long, requires_grad=False)
        for length in lengths
    ], enforce_sorted=False)


@given(
    batch_size=st.integers(1, 10),
    total_length=st.integers(1, 10),
    num_tags=st.integers(1, 10),
    num_packs=st.integers(1, 10),
)
def test_stacked_packed_sequences(batch_size, total_length, num_tags, num_packs):
    lengths = gen_lengths(batch_size, total_length)

    emissions = [
        gen_emissions(lengths, num_tags)
        for _ in range(num_packs)
    ]
    tags = [
        gen_tags(lengths, num_tags)
        for _ in range(num_packs)
    ]

    stacked_emissions = stack_packed_sequences(emissions)
    stacked_tags = stack_packed_sequences(tags)

    decoder = CrfDecoder(num_tags)
    losses = [
        decoder.fit(e, t, reduction='none')
        for e, t in zip(emissions, tags)
    ]
    loss1 = rearrange(torch.stack(losses, dim=1), 'b n ...-> (b n) ...')
    loss2 = decoder.fit(stacked_emissions, stacked_tags)

    assert_equal(loss1, loss2)
