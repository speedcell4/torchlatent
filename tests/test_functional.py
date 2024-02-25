import torch
from hypothesis import given, settings, strategies as st
from torchnyan.assertion import assert_close, assert_grad_close
from torchnyan.strategy import TINY_BATCH_SIZE, TINY_TOKEN_SIZE, device, sizes

from torchlatent.functional import logaddexp, logsumexp


@settings(deadline=None)
@given(
    token_sizes=sizes(TINY_BATCH_SIZE, TINY_TOKEN_SIZE)
)
def test_logaddexp(token_sizes):
    x = torch.randn(token_sizes, device=device, requires_grad=True)
    y = torch.randn(token_sizes, device=device, requires_grad=True)

    actual = logaddexp(x, y)
    expected = torch.logaddexp(x, y)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=(x, y))


@settings(deadline=None)
@given(
    data=st.data(),
    token_sizes=sizes(TINY_BATCH_SIZE, TINY_TOKEN_SIZE)
)
def test_logsumexp(data, token_sizes):
    tensor = torch.randn(token_sizes, device=device, requires_grad=True)
    dim = data.draw(st.integers(min_value=-len(token_sizes), max_value=len(token_sizes) - 1))

    actual = logsumexp(tensor, dim=dim)
    expected = torch.logsumexp(tensor, dim=dim)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=(tensor,))
