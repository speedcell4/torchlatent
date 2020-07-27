import torch
from hypothesis import given, strategies as st
from torchcrf import CRF

from torchlatent.crf import CrfDecoder
from torchlatent.functional import build_mask


@given(
    batch_size=st.integers(1, 12),
    sentence_length=st.integers(2, 12),  # TODO: check sentence_length = 1 case
    num_tags=st.integers(1, 12),
)
def test_crf_decoder_correctness(batch_size, sentence_length, num_tags):
    decoder1 = CrfDecoder(num_tags=num_tags, batch_first=True)
    decoder2 = CRF(num_tags, batch_first=True)
    decoder2.transitions.data[:] = decoder1.transition.data[:]
    decoder2.start_transitions.data[:] = decoder1.start_transition.data[:]
    decoder2.end_transitions.data[:] = decoder1.end_transition.data[:]

    emission = torch.randn((batch_size, sentence_length, num_tags))
    emission1 = emission.clone().requires_grad_(True)
    emission2 = emission.clone().requires_grad_(True)

    lengths = torch.randint(0, sentence_length, (batch_size,)) + 1
    mask = build_mask(lengths, padding_mask=False, batch_first=True, max_length=sentence_length)
    target = torch.randint(0, num_tags, (batch_size, sentence_length))

    log_prob1 = decoder1.fit(log_potentials=emission1, target=target, lengths=lengths)
    log_prob2 = decoder2(emissions=emission2, tags=target, mask=mask, reduction='none')
    assert torch.allclose(log_prob1, log_prob2, rtol=1e-5, atol=1e-5)

    emission1.grad = None
    emission2.grad = None
    log_prob1.backward(torch.ones_like(log_prob1))
    log_prob2.backward(torch.ones_like(log_prob2))
    assert torch.allclose(emission1.grad, emission2.grad, rtol=1e-5, atol=1e-5)
