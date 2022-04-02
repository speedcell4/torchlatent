import torch
from hypothesis import given
from torch.testing import assert_close
from torch_struct import TreeCRF

from tests.strategies import sizes, BATCH_SIZE, TOKEN_SIZE, devices, TINY_BATCH_SIZE
from tests.utils import assert_grad_close
from torchlatent.cky import CkyDistribution, cky_indices


@given(
    device=devices(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    num_tags=sizes(TOKEN_SIZE),
)
def test_cky_log_partitions(device, token_sizes, num_tags):
    scores = torch.randn(
        (len(token_sizes), max(token_sizes), max(token_sizes), num_tags),
        requires_grad=True, device=device,
    )
    token_sizes = torch.tensor(token_sizes, device=device)

    excepted = TreeCRF(log_potentials=scores, lengths=token_sizes)
    actual = CkyDistribution(scores=scores, indices=cky_indices(token_sizes=token_sizes, device=device))

    assert_close(actual=actual.log_partitions, expected=excepted.partition)
    assert_grad_close(actual=actual.log_partitions, expected=excepted.partition, inputs=scores, rtol=1e-5, atol=1e-5)
