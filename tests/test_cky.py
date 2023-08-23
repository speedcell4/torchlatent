import torch
from hypothesis import given
from hypothesis import strategies as st
from torch_struct import TreeCRF

from torchlatent.cky import CkyDecoder
from torchlatent.cky import CkyDistribution
from torchlatent.cky import cky_partitions_indices
from torchnyan.assertion import assert_close
from torchnyan.assertion import assert_grad_close
from torchnyan.strategy import BATCH_SIZE
from torchnyan.strategy import FEATURE_DIM
from torchnyan.strategy import TINY_BATCH_SIZE
from torchnyan.strategy import TOKEN_SIZE
from torchnyan.strategy import device
from torchnyan.strategy import sizes
from torchrua import cat_sequence


@given(
    token_sizes=sizes(TINY_BATCH_SIZE, TOKEN_SIZE),
    embedding_dim=sizes(FEATURE_DIM),
    num_tags=sizes(TOKEN_SIZE),
    dropout=st.floats(0, 1),
)
def test_cky_catted_max(token_sizes, embedding_dim, num_tags, dropout):
    sequence = cat_sequence([
        torch.randn((token_size, embedding_dim), requires_grad=True, device=device)
        for token_size in token_sizes
    ])

    targets = cat_sequence([
        torch.empty((token_size * 2 - 1,), dtype=torch.long, device=device)
        for token_size in token_sizes
    ])

    decoder = CkyDecoder(
        in_features=embedding_dim, hidden_features=embedding_dim,
        num_targets=num_tags, dropout=dropout,
    ).to(device=device)
    dist = decoder(sequence)

    assert_close(actual=dist.max, expected=dist.log_scores(targets=targets._replace(data=dist.argmax)))


# @given(
#     token_sizes=sizes(TINY_BATCH_SIZE, TOKEN_SIZE),
#     embedding_dim=sizes(FEATURE_DIM),
#     num_tags=sizes(TOKEN_SIZE),
#     bias=st.booleans(),
# )
# def test_cky_packed_max(token_sizes, embedding_dim, num_tags, bias):
#     sequence = pack_sequence([
#         torch.randn((token_size, embedding_dim), requires_grad=True, device=device)
#         for token_size in token_sizes
#     ])
#
#     decoder = CkyLayer(in_features=embedding_dim, out_features=num_tags, bias=bias).to(device=device)
#     cky = decoder.forward(sequence=sequence)
#
#     assert_close(actual=cky.max, expected=cky.log_scores(decoder.decode(sequence=sequence)))


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    num_tags=sizes(TOKEN_SIZE),
)
def test_cky_log_partitions(token_sizes, num_tags):
    scores = torch.randn(
        (len(token_sizes), max(token_sizes), max(token_sizes), num_tags),
        requires_grad=True, device=device,
    )
    token_sizes = torch.tensor(token_sizes, device=device)

    expected = TreeCRF(log_potentials=scores, lengths=token_sizes)
    actual = CkyDistribution(emissions=scores, indices=cky_partitions_indices(token_sizes=token_sizes, device=device))

    assert_close(actual=actual.log_partitions, expected=expected.partition)
    assert_grad_close(actual=actual.log_partitions, expected=expected.partition, inputs=scores, rtol=1e-5, atol=1e-5)
