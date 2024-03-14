import torch
from hypothesis import given, settings
from torch_struct import TreeCRF
from torchnyan import BATCH_SIZE, TINY_TOKEN_SIZE, assert_close, assert_grad_close, device, sizes
from torchrua import C

from torchlatent.cky import CkyDecoder, masked_select


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TINY_TOKEN_SIZE),
    num_targets=sizes(TINY_TOKEN_SIZE),
)
def test_cky_scores(token_sizes, num_targets):
    logits = torch.randn(
        (len(token_sizes), max(token_sizes), max(token_sizes), num_targets),
        device=device, requires_grad=True,
    )
    token_sizes = torch.tensor(token_sizes, device=device)

    expected = TreeCRF(logits, lengths=token_sizes)
    actual = CkyDecoder(num_targets=num_targets)(logits=C(logits, token_sizes))

    expected = expected.log_prob(expected.argmax)
    actual = actual.log_probs(actual.argmax)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=(logits,))


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TINY_TOKEN_SIZE),
    num_targets=sizes(TINY_TOKEN_SIZE),
)
def test_cky_partitions(token_sizes, num_targets):
    logits = torch.randn(
        (len(token_sizes), max(token_sizes), max(token_sizes), num_targets),
        device=device, requires_grad=True,
    )
    token_sizes = torch.tensor(token_sizes, device=device)

    expected = TreeCRF(logits, lengths=token_sizes)
    actual = CkyDecoder(num_targets=num_targets)(logits=C(logits, token_sizes))

    expected = expected.partition
    actual = actual.log_partitions

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=(logits,))


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TINY_TOKEN_SIZE),
    num_targets=sizes(TINY_TOKEN_SIZE),
)
def test_cky_argmax(token_sizes, num_targets):
    logits = torch.randn(
        (len(token_sizes), max(token_sizes), max(token_sizes), num_targets),
        device=device, requires_grad=True,
    )
    token_sizes = torch.tensor(token_sizes, device=device)

    expected = TreeCRF(logits, lengths=token_sizes)
    actual = CkyDecoder(num_targets=num_targets)(logits=C(logits, token_sizes))

    expected = C(data=torch.stack(masked_select(expected.argmax.bool()), dim=-1), token_sizes=token_sizes * 2 - 1)
    actual = actual.argmax

    for actual, expected in zip(actual.tolist(), expected.tolist()):
        assert set(map(tuple, actual)) == set(map(tuple, expected))


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TINY_TOKEN_SIZE),
    num_targets=sizes(TINY_TOKEN_SIZE),
)
def test_cky_entropy(token_sizes, num_targets):
    logits = torch.randn(
        (len(token_sizes), max(token_sizes), max(token_sizes), num_targets),
        device=device, requires_grad=True,
    )
    token_sizes = torch.tensor(token_sizes, device=device)

    expected = TreeCRF(logits, lengths=token_sizes)
    actual = CkyDecoder(num_targets=num_targets)(logits=C(logits, token_sizes))

    expected = expected.entropy
    actual = actual.entropy

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=(logits,), rtol=1e-4, atol=1e-4)


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TINY_TOKEN_SIZE),
    num_targets=sizes(TINY_TOKEN_SIZE),
)
def test_cky_kl(token_sizes, num_targets):
    logits1 = torch.randn(
        (len(token_sizes), max(token_sizes), max(token_sizes), num_targets),
        device=device, requires_grad=True,
    )
    logits2 = torch.randn(
        (len(token_sizes), max(token_sizes), max(token_sizes), num_targets),
        device=device, requires_grad=True,
    )
    token_sizes = torch.tensor(token_sizes, device=device)

    expected1 = TreeCRF(logits1, lengths=token_sizes)
    expected2 = TreeCRF(logits2, lengths=token_sizes)
    actual1 = CkyDecoder(num_targets=num_targets)(logits=C(logits1, token_sizes))
    actual2 = CkyDecoder(num_targets=num_targets)(logits=C(logits2, token_sizes))

    expected = expected1.kl(expected2)
    actual = actual1.kl(actual2)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=(logits1, logits2), rtol=1e-4, atol=1e-4)
