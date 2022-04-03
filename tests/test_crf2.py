import torch
import torchcrf
from hypothesis import given
from torch.testing import assert_close
from torchrua import cat_sequence, pad_sequence, pad_catted_indices, pad_packed_indices, pack_sequence, \
    pad_catted_sequence

from tests.strategies import devices, sizes, TOKEN_SIZE, TINY_BATCH_SIZE
from tests.utils import assert_grad_close, assert_equal
from torchlatent.crf2 import CrfDecoder


@given(
    device=devices(),
    token_sizes=sizes(TINY_BATCH_SIZE, TOKEN_SIZE),
    num_tags=sizes(TOKEN_SIZE),
)
def test_crf_catted_fit(device, token_sizes, num_tags):
    actual_decoder = CrfDecoder(num_tags)
    excepted_decoder = torchcrf.CRF(num_tags, batch_first=False)

    actual_decoder.transitions.data = excepted_decoder.transitions[None, None, :, :]
    actual_decoder.head_transitions.data = excepted_decoder.start_transitions[None, None, :]
    actual_decoder.last_transitions.data = excepted_decoder.end_transitions[None, None, :]

    emissions = [
        torch.randn((token_size, num_tags), requires_grad=True, device=device)
        for token_size in token_sizes
    ]
    targets = [
        torch.randint(0, num_tags, (token_size,), device=device)
        for token_size in token_sizes
    ]

    catted_emissions = cat_sequence([x[:, None] for x in emissions])
    catted_targets = cat_sequence([x[:, None] for x in targets])

    padded_emissions, _ = pad_sequence(emissions, batch_first=False)
    padded_targets, _ = pad_sequence(targets, batch_first=False)

    size, ptr = pad_catted_indices(catted_emissions.token_sizes, batch_first=False)
    mask = torch.zeros(size, dtype=torch.bool, device=device)
    mask[ptr] = True

    actual = actual_decoder.fit(emissions=catted_emissions, targets=catted_targets)[:, 0]
    excepted = excepted_decoder.forward(
        emissions=padded_emissions, tags=padded_targets.long(),
        mask=mask.byte(), reduction='none',
    ).neg()

    assert_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual, expected=excepted, inputs=emissions)


@given(
    device=devices(),
    token_sizes=sizes(TINY_BATCH_SIZE, TOKEN_SIZE),
    num_tags=sizes(TOKEN_SIZE),
)
def test_crf_catted_decode(device, token_sizes, num_tags):
    actual_decoder = CrfDecoder(num_tags)
    excepted_decoder = torchcrf.CRF(num_tags, batch_first=False)

    actual_decoder.transitions.data = excepted_decoder.transitions[None, None, :, :]
    actual_decoder.head_transitions.data = excepted_decoder.start_transitions[None, None, :]
    actual_decoder.last_transitions.data = excepted_decoder.end_transitions[None, None, :]

    emissions = [
        torch.randn((token_size, num_tags), requires_grad=True, device=device)
        for token_size in token_sizes
    ]

    catted_emissions = cat_sequence([x[:, None] for x in emissions])

    padded_emissions, _ = pad_sequence(emissions, batch_first=False)

    size, ptr = pad_catted_indices(catted_emissions.token_sizes, batch_first=False)
    mask = torch.zeros(size, dtype=torch.bool, device=device)
    mask[ptr] = True

    actual, actual_token_sizes = actual_decoder.decode(emissions=catted_emissions)

    excepted = excepted_decoder.decode(emissions=padded_emissions, mask=mask.byte())
    excepted, excepted_token_sizes = cat_sequence([torch.tensor(x, device=device) for x in excepted])

    assert_equal(actual=actual[:, 0], expected=excepted)
    assert_equal(actual=actual_token_sizes, expected=excepted_token_sizes)


@given(
    device=devices(),
    token_sizes=sizes(TINY_BATCH_SIZE, TOKEN_SIZE),
    num_tags=sizes(TOKEN_SIZE),
)
def test_crf_packed_fit(device, token_sizes, num_tags):
    actual_decoder = CrfDecoder(num_tags)
    excepted_decoder = torchcrf.CRF(num_tags, batch_first=False)

    actual_decoder.transitions.data = excepted_decoder.transitions[None, None, :, :]
    actual_decoder.head_transitions.data = excepted_decoder.start_transitions[None, None, :]
    actual_decoder.last_transitions.data = excepted_decoder.end_transitions[None, None, :]

    emissions = [
        torch.randn((token_size, num_tags), requires_grad=True, device=device)
        for token_size in token_sizes
    ]
    targets = [
        torch.randint(0, num_tags, (token_size,), device=device)
        for token_size in token_sizes
    ]

    packed_emissions = pack_sequence([x[:, None] for x in emissions])
    packed_targets = pack_sequence([x[:, None] for x in targets])

    padded_emissions, _ = pad_sequence(emissions, batch_first=False)
    padded_targets, _ = pad_sequence(targets, batch_first=False)

    size, ptr, _ = pad_packed_indices(
        batch_sizes=packed_emissions.batch_sizes,
        sorted_indices=packed_emissions.sorted_indices,
        unsorted_indices=packed_emissions.unsorted_indices,
        batch_first=False,
    )
    mask = torch.zeros(size, dtype=torch.bool, device=device)
    mask[ptr] = True

    actual = actual_decoder.fit(emissions=packed_emissions, targets=packed_targets)[:, 0]
    excepted = excepted_decoder.forward(
        emissions=padded_emissions, tags=padded_targets.long(),
        mask=mask.byte(), reduction='none',
    ).neg()

    assert_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual, expected=excepted, inputs=emissions)
