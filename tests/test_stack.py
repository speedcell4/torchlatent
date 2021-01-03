from typing import List

import torch
from einops import rearrange
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_sequence
from torchrua import stack_packed_sequences

from tests.test_crf import assert_equal
from torchlatent.crf import CrfDecoder, StackedCrfDecoder
from torchlatent.instr import stack_instr


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


@given(
    batch_size=st.integers(1, 10),
    total_length=st.integers(1, 10),
    num_tags=st.integers(1, 10),
    num_packs=st.integers(1, 10),
)
def test_stack_instr(
        batch_size: int, total_length: int, num_tags: int, num_packs: int):
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

    _, _, _, batch_ptr, instr = decoder._validate(
        emissions=emissions[0], tags=tags[0],
        lengths=None, batch_ptr=None, instr=None,
    )
    stacked_batch_ptr, stacked_instr = stack_instr(
        batch_ptr=batch_ptr, instr=instr, n=num_packs,
    )

    losses = [
        decoder.fit(e, t, batch_ptr=batch_ptr, instr=instr)
        for e, t in zip(emissions, tags)
    ]
    loss1 = rearrange(torch.stack(losses, dim=1), 'b n ...-> (b n) ...')
    loss2 = decoder.fit(
        stacked_emissions, stacked_tags,
        batch_ptr=stacked_batch_ptr, instr=stacked_instr,
    )

    assert_equal(loss1, loss2)


def test_stacked_crf_decoder():
    layer1 = CrfDecoder(5)
    layer2 = CrfDecoder(5)
    layer3 = CrfDecoder(5)
    layer = StackedCrfDecoder(layer1, layer2, layer3)

    emissions1 = pack_sequence([
        torch.randn((5, 5), requires_grad=True),
        torch.randn((2, 5), requires_grad=True),
        torch.randn((3, 5), requires_grad=True),
    ], enforce_sorted=False)
    emissions2 = pack_sequence([
        torch.randn((5, 5), requires_grad=True),
        torch.randn((2, 5), requires_grad=True),
        torch.randn((3, 5), requires_grad=True),
    ], enforce_sorted=False)
    emissions3 = pack_sequence([
        torch.randn((5, 5), requires_grad=True),
        torch.randn((2, 5), requires_grad=True),
        torch.randn((3, 5), requires_grad=True),
    ], enforce_sorted=False)
    emissions = stack_packed_sequences([emissions1, emissions2, emissions3])

    tags1 = pack_sequence([
        torch.randint(5, (5,), dtype=torch.long),
        torch.randint(5, (2,), dtype=torch.long),
        torch.randint(5, (3,), dtype=torch.long),
    ], enforce_sorted=False)
    tags2 = pack_sequence([
        torch.randint(5, (5,), dtype=torch.long),
        torch.randint(5, (2,), dtype=torch.long),
        torch.randint(5, (3,), dtype=torch.long),
    ], enforce_sorted=False)
    tags3 = pack_sequence([
        torch.randint(5, (5,), dtype=torch.long),
        torch.randint(5, (2,), dtype=torch.long),
        torch.randint(5, (3,), dtype=torch.long),
    ], enforce_sorted=False)
    tags = stack_packed_sequences([tags1, tags2, tags3])

    loss_lhs = layer.fit(emissions=emissions, tags=tags)

    loss1 = layer1.fit(emissions=emissions1, tags=tags1)
    loss2 = layer2.fit(emissions=emissions2, tags=tags2)
    loss3 = layer3.fit(emissions=emissions3, tags=tags3)
    loss_rhs = torch.stack([loss1, loss2, loss3], dim=1).view(-1)
    assert_equal(loss_lhs, loss_rhs)

    prediction_lhs = layer.decode(emissions=emissions)

    prediction1 = layer1.decode(emissions=emissions1)
    prediction2 = layer2.decode(emissions=emissions2)
    prediction3 = layer3.decode(emissions=emissions3)
    prediction_rhs = torch.stack([prediction1.data, prediction2.data, prediction3.data], dim=1).view(-1)
    assert_equal(prediction_lhs.data, prediction_rhs)
