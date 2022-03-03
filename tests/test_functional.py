import torch
from hypothesis import given, strategies as st

from tests.strategies import devices, sizes, TINY_TOKEN_SIZE, TINY_BATCH_SIZE
from tests.utils import assert_close, assert_grad_close
from torchlatent.functional import logaddexp, logsumexp


@given(
    device=devices(),
    token_sizes=sizes(TINY_BATCH_SIZE, TINY_TOKEN_SIZE)
)
def test_logaddexp(device, token_sizes):
    x = torch.randn(token_sizes, device=device, requires_grad=True)
    y = torch.randn(token_sizes, device=device, requires_grad=True)

    actual = logaddexp(x, y)
    expected = torch.logaddexp(x, y)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=(x, y))


@given(
    data=st.data(),
    device=devices(),
    token_sizes=sizes(TINY_BATCH_SIZE, TINY_TOKEN_SIZE)
)
def test_logsumexp(data, device, token_sizes):
    tensor = torch.randn(token_sizes, device=device, requires_grad=True)
    dim = data.draw(st.integers(min_value=-len(token_sizes), max_value=len(token_sizes) - 1))

    actual = logsumexp(tensor, dim=dim)
    expected = torch.logsumexp(tensor, dim=dim)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=(tensor,))
