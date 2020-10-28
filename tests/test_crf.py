import torch
from hypothesis import given, strategies as st
from hypothesis import settings
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence
from torchcrf import CRF
from torchrua import lengths_to_mask

from torchlatent.crf import CrfDecoder

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def assert_equal(x: Tensor, y: Tensor) -> None:
    assert x.size() == y.size(), f'{x.size()} != {y.size()}'
    assert torch.allclose(x, y, rtol=1e-5, atol=1e-5), f'{x.view(-1)} != {y.view(-1)}'


@settings(deadline=None)
@given(
    batch_size=st.integers(1, 12),
    total_length=st.integers(1, 12),
    num_tags=st.integers(1, 12),
)
def test_crf_decoder_fit(batch_size, total_length, num_tags):
    our_decoder = CrfDecoder(num_tags=num_tags).to(device=device)
    their_decoder = CRF(num_tags, batch_first=True).to(device=device)
    their_decoder.transitions.data[:] = our_decoder.transitions.data[:]
    their_decoder.start_transitions.data[:] = our_decoder.start_transitions.data[:]
    their_decoder.end_transitions.data[:] = our_decoder.end_transitions.data[:]

    emissions = torch.randn((batch_size, total_length, num_tags), device=device)
    lengths = torch.randint(0, total_length, (batch_size,), device=device) + 1
    lengths[torch.randint(0, batch_size, ()).item()] = total_length

    our_emissions = pack_padded_sequence(emissions, lengths=lengths, batch_first=True, enforce_sorted=False)
    our_emissions.data.requires_grad_(True)
    their_emissions = emissions.clone().requires_grad_(True)

    tags = torch.randint(0, num_tags, (batch_size, total_length), device=device)
    our_tags = pack_padded_sequence(tags, lengths=lengths, batch_first=True, enforce_sorted=False)
    their_tags = tags.clone()
    mask = lengths_to_mask(lengths=lengths, filling_mask=True, batch_first=True, device=device)

    our_log_prob = our_decoder.fit(emissions=our_emissions, tags=our_tags)
    their_log_prob = their_decoder(emissions=their_emissions, tags=their_tags, mask=mask, reduction='none')
    assert_equal(our_log_prob, their_log_prob)

    our_emissions.data.grad = None
    their_emissions.grad = None
    our_log_prob.sum().backward()
    their_log_prob.sum().backward()
    assert_equal(our_decoder.transitions, their_decoder.transitions)
    assert_equal(our_decoder.start_transitions, their_decoder.start_transitions)
    assert_equal(our_decoder.end_transitions, their_decoder.end_transitions)

    our_emissions_grad = our_emissions.data.grad
    their_emissions_grad = pack_padded_sequence(
        their_emissions.grad, lengths=lengths, batch_first=True, enforce_sorted=False,
    ).data

    assert_equal(our_emissions_grad, their_emissions_grad)
