import torch
import torchcrf
from hypothesis import given
from torch.testing import assert_close
from torchrua import cat_sequence, pad_sequence, pad_catted_indices

from tests.strategies import devices, sizes, TOKEN_SIZE, TINY_BATCH_SIZE
from tests.utils import assert_grad_close
from torchlatent.crf2 import CrfDecoder


@given(
    device=devices(),
    token_sizes=sizes(TINY_BATCH_SIZE, TOKEN_SIZE),
    num_tags=sizes(TOKEN_SIZE),
)
def test_crf_catted_fit(device, token_sizes, num_tags):
    decoder1 = CrfDecoder(num_tags)
    decoder2 = torchcrf.CRF(num_tags, batch_first=False)

    decoder1.transitions.data = decoder2.transitions[None, None, :, :]
    decoder1.head_transitions.data = decoder2.start_transitions[None, None, :]
    decoder1.last_transitions.data = decoder2.end_transitions[None, None, :]

    emissions = [
        torch.randn((token_size, num_tags), requires_grad=True, device=device)
        for token_size in token_sizes
    ]
    targets = [
        torch.randint(0, num_tags, (token_size,), device=device)
        for token_size in token_sizes
    ]

    catted_emissions = cat_sequence([x[:, None] for x in emissions])
    catted_targets = cat_sequence([x[:, None] for x in targets])

    padded_emissions, _ = pad_sequence(emissions, batch_first=False)
    padded_targets, _ = pad_sequence(targets, batch_first=False)

    size, ptr = pad_catted_indices(catted_emissions.token_sizes, batch_first=False)
    mask = torch.zeros(size, dtype=torch.bool, device=device)
    mask[ptr] = True

    actual = decoder1.fit(catted_emissions, catted_targets)[:, 0]
    excepted = decoder2.forward(padded_emissions, padded_targets, mask=mask, reduction='none').neg()

    assert_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual, expected=excepted, inputs=emissions)
