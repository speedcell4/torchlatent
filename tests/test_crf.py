import torch
from hypothesis import given
from torchrua import pack_sequence, cat_sequence, pack_catted_sequence

from tests.strategies import devices, sizes, BATCH_SIZE, TOKEN_SIZE, NUM_CONJUGATES, NUM_TAGS
from tests.utils import assert_close, assert_grad_close, assert_packed_sequence_equal
from third.crf import CrfDecoder as ThirdPartyCrfDecoder
from torchlatent.crf import CrfDecoder


@given(
    device=devices(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    num_conjugate=sizes(NUM_CONJUGATES),
    num_tags=sizes(NUM_TAGS),
)
def test_crf_packed_fit(device, token_sizes, num_conjugate, num_tags):
    emissions = pack_sequence([
        torch.randn((token_size, num_conjugate, num_tags), device=device, requires_grad=True)
        for token_size in token_sizes
    ], device=device)

    tags = pack_sequence([
        torch.randint(0, num_tags, (token_size, num_conjugate), device=device)
        for token_size in token_sizes
    ], device=device)

    actual_decoder = CrfDecoder(num_tags=num_tags, num_conjugates=num_conjugate).to(device=device)
    expected_decoder = ThirdPartyCrfDecoder(num_tags=num_tags, num_conjugates=num_conjugate).to(device=device)
    expected_decoder.reset_parameters_with_(decoder=actual_decoder)

    actual = actual_decoder.fit(emissions=emissions, tags=tags)
    expected = expected_decoder.fit(emissions=emissions, tags=tags)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=(emissions.data,))


@given(
    device=devices(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    num_conjugate=sizes(NUM_CONJUGATES),
    num_tags=sizes(NUM_TAGS),
)
def test_crf_packed_decode(device, token_sizes, num_conjugate, num_tags):
    emissions = pack_sequence([
        torch.randn((token_size, num_conjugate, num_tags), device=device, requires_grad=True)
        for token_size in token_sizes
    ], device=device)

    actual_decoder = CrfDecoder(num_tags=num_tags, num_conjugates=num_conjugate).to(device=device)
    expected_decoder = ThirdPartyCrfDecoder(num_tags=num_tags, num_conjugates=num_conjugate).to(device=device)
    expected_decoder.reset_parameters_with_(decoder=actual_decoder)

    expected = expected_decoder.decode(emissions=emissions)
    actual = actual_decoder.decode(emissions=emissions)

    assert_packed_sequence_equal(actual=actual, expected=expected)


@given(
    device=devices(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    num_conjugate=sizes(NUM_CONJUGATES),
    num_tags=sizes(NUM_TAGS),
)
def test_crf_catted_fit(device, token_sizes, num_conjugate, num_tags):
    emissions = [
        torch.randn((token_size, num_conjugate, num_tags), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    tags = [
        torch.randint(0, num_tags, (token_size, num_conjugate), device=device)
        for token_size in token_sizes
    ]

    catted_emissions = cat_sequence(emissions, device=device)
    packed_emissions = pack_sequence(emissions, device=device)

    catted_tags = cat_sequence(tags, device=device)
    packed_tags = pack_sequence(tags, device=device)

    actual_decoder = CrfDecoder(num_tags=num_tags, num_conjugates=num_conjugate).to(device=device)
    expected_decoder = ThirdPartyCrfDecoder(num_tags=num_tags, num_conjugates=num_conjugate).to(device=device)
    expected_decoder.reset_parameters_with_(decoder=actual_decoder)

    actual = actual_decoder.fit(emissions=catted_emissions, tags=catted_tags)
    expected = expected_decoder.fit(emissions=packed_emissions, tags=packed_tags)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=tuple(emissions))


@given(
    device=devices(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    num_conjugate=sizes(NUM_CONJUGATES),
    num_tags=sizes(NUM_TAGS),
)
def test_crf_catted_decode(device, token_sizes, num_conjugate, num_tags):
    emissions = [
        torch.randn((token_size, num_conjugate, num_tags), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    catted_emissions = cat_sequence(emissions, device=device)
    packed_emissions = pack_sequence(emissions, device=device)

    actual_decoder = CrfDecoder(num_tags=num_tags, num_conjugates=num_conjugate).to(device=device)
    expected_decoder = ThirdPartyCrfDecoder(num_tags=num_tags, num_conjugates=num_conjugate).to(device=device)
    expected_decoder.reset_parameters_with_(decoder=actual_decoder)

    expected = expected_decoder.decode(emissions=packed_emissions)
    actual = actual_decoder.decode(emissions=catted_emissions)
    actual = pack_catted_sequence(*actual, device=device)

    assert_packed_sequence_equal(actual=actual, expected=expected)
