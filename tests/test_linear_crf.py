import torch
from hypothesis import given
from torch.testing import assert_close
from torchcrf import CRF
from torchnyan import BATCH_SIZE
from torchnyan import TOKEN_SIZE
from torchnyan import assert_grad_close
from torchnyan import device
from torchnyan import sizes
from torchrua import cat_sequence
from torchrua import pack_sequence
from torchrua import pad_sequence

from torchlatent.linear_crf import crf_partitions
from torchlatent.semiring import Log


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    num_targets=sizes(TOKEN_SIZE),
)
def test_crf_catted_partitions(token_sizes, num_targets):
    inputs = [
        torch.randn((token_size, num_targets), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    excepted_emissions, token_sizes = pad_sequence(inputs, batch_first=True)
    index = torch.arange(token_sizes.max().detach().cpu().item(), device=token_sizes.device)
    mask = index[None, :] < token_sizes[:, None]

    excepted_crf = CRF(num_tags=num_targets, batch_first=False)
    excepted = excepted_crf._compute_normalizer(excepted_emissions.transpose(0, 1), mask.t())

    actual = crf_partitions(
        emissions=cat_sequence(inputs),
        transitions=(excepted_crf.transitions, excepted_crf.start_transitions, excepted_crf.end_transitions),
        semiring=Log,
    )

    assert_close(actual=actual, expected=excepted, rtol=1e-4, atol=1e-4)
    assert_grad_close(actual=actual, expected=excepted, inputs=inputs, rtol=1e-4, atol=1e-4)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    num_targets=sizes(TOKEN_SIZE),
)
def test_crf_packed_partitions(token_sizes, num_targets):
    inputs = [
        torch.randn((token_size, num_targets), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    excepted_emissions, token_sizes = pad_sequence(inputs, batch_first=True)
    index = torch.arange(token_sizes.max().detach().cpu().item(), device=token_sizes.device)
    mask = index[None, :] < token_sizes[:, None]

    excepted_crf = CRF(num_tags=num_targets, batch_first=False)
    excepted = excepted_crf._compute_normalizer(excepted_emissions.transpose(0, 1), mask.t())

    actual = crf_partitions(
        emissions=pack_sequence(inputs),
        transitions=(excepted_crf.transitions, excepted_crf.start_transitions, excepted_crf.end_transitions),
        semiring=Log,
    )

    assert_close(actual=actual, expected=excepted, rtol=1e-4, atol=1e-4)
    assert_grad_close(actual=actual, expected=excepted, inputs=inputs, rtol=1e-4, atol=1e-4)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    num_targets=sizes(TOKEN_SIZE),
)
def test_crf_padded_partitions(token_sizes, num_targets):
    inputs = [
        torch.randn((token_size, num_targets), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    excepted_emissions, token_sizes = pad_sequence(inputs, batch_first=True)
    index = torch.arange(token_sizes.max().detach().cpu().item(), device=token_sizes.device)
    mask = index[None, :] < token_sizes[:, None]

    excepted_crf = CRF(num_tags=num_targets, batch_first=False)
    excepted = excepted_crf._compute_normalizer(excepted_emissions.transpose(0, 1), mask.t())

    actual = crf_partitions(
        emissions=pad_sequence(inputs, batch_first=True),
        transitions=(excepted_crf.transitions, excepted_crf.start_transitions, excepted_crf.end_transitions),
        semiring=Log,
    )

    assert_close(actual=actual, expected=excepted, rtol=1e-4, atol=1e-4)
    assert_grad_close(actual=actual, expected=excepted, inputs=inputs, rtol=1e-4, atol=1e-4)
