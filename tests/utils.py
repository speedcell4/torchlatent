from typing import Tuple

import torch
from torch import Tensor
from torch.testing import assert_close, assert_equal

__all__ = [
    'assert_close',
    'assert_equal',
    'assert_grad_close',
]


def assert_grad_close(actual: Tensor, expected: Tensor, inputs: Tuple[Tensor, ...]) -> None:
    grad = torch.randn_like(actual)

    actual_grads = torch.autograd.grad(actual, inputs, grad)
    expected_grads = torch.autograd.grad(expected, inputs, grad)

    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        if actual_grad is None:
            assert expected_grad is None
        else:
            assert_close(actual=actual_grad, expected=expected_grad)
