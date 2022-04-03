import torch
from hypothesis import given, strategies as st
from torch.testing import assert_close
from torch_struct import TreeCRF
from torchrua import pack_sequence

from tests.strategies import sizes, BATCH_SIZE, TOKEN_SIZE, EMBEDDING_DIM, devices, TINY_BATCH_SIZE
from tests.utils import assert_grad_close
from torchlatent.cky import CkyDistribution, cky_indices, CkyDecoder


@given(
    device=devices(),
    token_sizes=sizes(TINY_BATCH_SIZE, TOKEN_SIZE),
    embedding_dim=sizes(EMBEDDING_DIM),
    num_tags=sizes(TOKEN_SIZE),
    bias=st.booleans(),
)
def test_cky_log_scores(device, token_sizes, embedding_dim, num_tags, bias):
    sequence = pack_sequence([
        torch.randn((token_size, embedding_dim), requires_grad=True, device=device)
        for token_size in token_sizes
    ])

    decoder = CkyDecoder(in_features=embedding_dim, out_features=num_tags, bias=bias)
    cky = decoder.forward(sequence=sequence)

    assert_close(actual=cky.max, expected=cky.log_scores(decoder.decode(sequence=sequence)))


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
    assert_grad_close(actual=actual.log_partitions, expected=excepted.partition, inputs=scores)
