import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_sequence

from torchlatent import CrfDecoder
from torchlatent.crf_scan import CrfDecoderScan


@given(
    batch_size=st.integers(1, 5),
    num_conj=st.integers(1, 5),
    num_tags=st.integers(1, 5),
)
def test_marginal(batch_size, num_conj, num_tags):
    lengths = torch.randint(1, 12, (batch_size,))

    emissions = pack_sequence([
        torch.randn((length, num_conj, num_tags), requires_grad=True)
        for length in lengths
    ], enforce_sorted=False)

    crf1 = CrfDecoder(num_tags=num_tags, num_conjugates=num_conj)
    crf2 = CrfDecoderScan(num_tags=num_tags, num_conjugates=num_conj)

    with torch.no_grad():
        crf2.transitions.data[:] = crf1.transitions.data[:]
        crf2.start_transitions.data[:] = crf1.start_transitions.data[:]
        crf2.end_transitions.data[:] = crf1.end_transitions.data[:]

    tgt = crf1.marginals(emissions=emissions)
    prd = crf2.marginals(emissions=emissions)

    assert torch.allclose(tgt, prd, rtol=1e-5, atol=1e-5)

    grad_tgt, = torch.autograd.grad(
        tgt, emissions.data, torch.ones_like(tgt),
        create_graph=True, allow_unused=False, only_inputs=True,
    )
    grad_prd, = torch.autograd.grad(
        prd, emissions.data, torch.ones_like(prd),
        create_graph=True, allow_unused=False, only_inputs=True,
    )

    assert torch.allclose(grad_tgt, grad_prd, rtol=1e-5, atol=1e-5)


@given(
    batch_size=st.integers(1, 5),
    num_conj=st.integers(1, 5),
    num_tags=st.integers(1, 5),
)
def test_fit(batch_size, num_conj, num_tags):
    lengths = torch.randint(1, 12, (batch_size,))

    emissions = pack_sequence([
        torch.randn((length, num_conj, num_tags), requires_grad=True)
        for length in lengths
    ], enforce_sorted=False)

    tags = pack_sequence([
        torch.randint(0, num_tags, (length, num_conj))
        for length in lengths
    ], enforce_sorted=False)

    crf1 = CrfDecoder(num_tags=num_tags, num_conjugates=num_conj)
    crf2 = CrfDecoderScan(num_tags=num_tags, num_conjugates=num_conj)

    with torch.no_grad():
        crf2.transitions.data[:] = crf1.transitions.data[:]
        crf2.start_transitions.data[:] = crf1.start_transitions.data[:]
        crf2.end_transitions.data[:] = crf1.end_transitions.data[:]

    tgt = crf1.fit(emissions=emissions, tags=tags)
    prd = crf2.fit(emissions=emissions, tags=tags)

    assert torch.allclose(tgt, prd, rtol=1e-5, atol=1e-5)

    grad_tgt, = torch.autograd.grad(
        tgt, emissions.data, torch.ones_like(tgt),
        create_graph=True, allow_unused=False, only_inputs=True,
    )
    grad_prd, = torch.autograd.grad(
        prd, emissions.data, torch.ones_like(prd),
        create_graph=True, allow_unused=False, only_inputs=True,
    )

    assert torch.allclose(grad_tgt, grad_prd, rtol=1e-5, atol=1e-5)


@given(
    batch_size=st.integers(1, 5),
    num_conj=st.integers(1, 5),
    num_tags=st.integers(1, 5),
)
def test_decode(batch_size, num_conj, num_tags):
    lengths = torch.randint(1, 12, (batch_size,))

    emissions = pack_sequence([
        torch.randn((length, num_conj, num_tags), requires_grad=True)
        for length in lengths
    ], enforce_sorted=False)

    crf1 = CrfDecoder(num_tags=num_tags, num_conjugates=num_conj)
    crf2 = CrfDecoderScan(num_tags=num_tags, num_conjugates=num_conj)

    with torch.no_grad():
        crf2.transitions.data[:] = crf1.transitions.data[:]
        crf2.start_transitions.data[:] = crf1.start_transitions.data[:]
        crf2.end_transitions.data[:] = crf1.end_transitions.data[:]

    tgt = crf1.decode(emissions=emissions)
    prd = crf2.decode(emissions=emissions)

    assert torch.equal(tgt.data, prd.data)
    assert torch.equal(tgt.batch_sizes, prd.batch_sizes)
    assert torch.equal(tgt.sorted_indices, prd.sorted_indices)
    assert torch.equal(tgt.unsorted_indices, prd.unsorted_indices)
