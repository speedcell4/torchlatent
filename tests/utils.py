from typing import Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.testing import assert_close, assert_equal

__all__ = [
    'assert_close',
    'assert_equal',
    'assert_grad_close',
    'assert_packed_close',
    'assert_packed_equal',
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


def assert_packed_equal(actual: PackedSequence, expected: PackedSequence) -> None:
    assert_equal(actual=actual.data, expected=expected.data)
    assert_equal(actual=actual.batch_sizes, expected=expected.batch_sizes)

    if actual.sorted_indices is None:
        assert expected.sorted_indices is None
    else:
        assert_equal(actual=actual.sorted_indices, expected=expected.sorted_indices)

    if actual.unsorted_indices is None:
        assert expected.unsorted_indices is None
    else:
        assert_equal(actual=actual.unsorted_indices, expected=expected.unsorted_indices)


def assert_packed_close(actual: PackedSequence, expected: PackedSequence) -> None:
    assert_close(actual=actual.data, expected=expected.data)
    assert_equal(actual=actual.batch_sizes, expected=expected.batch_sizes)

    if actual.sorted_indices is None:
        assert expected.sorted_indices is None
    else:
        assert_equal(actual=actual.sorted_indices, expected=expected.sorted_indices)

    if actual.unsorted_indices is None:
        assert expected.unsorted_indices is None
    else:
        assert_equal(actual=actual.unsorted_indices, expected=expected.unsorted_indices)
