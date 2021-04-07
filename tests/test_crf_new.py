import torch
from hypothesis import given, strategies as st
from torchcrf import CRF
from torchrua import pad_packed_sequence, lengths_to_mask

from tests.strategies import length_lists, num_tags_integers, conjugated_emissions_packs, conjugated_tags_packs
from torchlatent.crf import compute_log_scores, compute_log_partitions
from torchlatent.instr import build_crf_batched_instr
from torchlatent.semiring import log


@given(
    data=st.data(),
    lengths=length_lists(),
    num_tags=num_tags_integers(),
    num_conjugates=num_tags_integers(),
)
def test_compute_log_scores_given_emissions(data, lengths, num_tags, num_conjugates):
    crf = CRF(num_tags=num_tags)

    emissions = data.draw(
        conjugated_emissions_packs(
            lengths=lengths, num_tags=num_tags, num_conjugates=num_conjugates))
    tags = data.draw(
        conjugated_tags_packs(
            lengths=lengths, num_tags=num_tags, num_conjugates=num_conjugates))

    out = compute_log_scores(
        emissions=emissions,
        tags=tags,
        transitions=crf.transitions[None, None, ...],
        start_transitions=crf.start_transitions[None, None, ...],
        end_transitions=crf.end_transitions[None, None, ...],
    )

    padded_emissions, lengths = pad_packed_sequence(pack=emissions, batch_first=False)
    mask = lengths_to_mask(lengths=lengths, batch_first=False)
    padded_tags, _ = pad_packed_sequence(pack=tags, batch_first=False)

    tgt = torch.stack([
        crf._compute_score(padded_emissions[..., index, :], padded_tags[..., index], mask)
        for index in range(num_conjugates)
    ], dim=1)

    assert torch.allclose(out, tgt, rtol=1e-3, atol=1e-3)

    out_grad, = torch.autograd.grad(
        out, emissions.data, torch.ones_like(out),
        retain_graph=False, create_graph=False, only_inputs=True,
    )
    tgt_grad, = torch.autograd.grad(
        tgt, emissions.data, torch.ones_like(tgt),
        retain_graph=False, create_graph=False, only_inputs=True,
    )

    assert torch.allclose(out_grad, tgt_grad, rtol=1e-3, atol=1e-3)


@given(
    data=st.data(),
    lengths=length_lists(),
    num_tags=num_tags_integers(),
    num_conjugates=num_tags_integers(),
)
def test_compute_log_scores_given_crfs(data, lengths, num_tags, num_conjugates):
    crfs = [CRF(num_tags=num_tags) for _ in range(num_conjugates)]

    emissions = data.draw(
        conjugated_emissions_packs(
            lengths=lengths, num_tags=num_tags, num_conjugates=num_conjugates))
    tags = data.draw(
        conjugated_tags_packs(
            lengths=lengths, num_tags=num_tags, num_conjugates=num_conjugates))

    out = compute_log_scores(
        emissions=emissions,
        tags=tags,
        transitions=torch.stack([crf.transitions[None, ...] for crf in crfs], dim=1),
        start_transitions=torch.stack([crf.start_transitions[None, ...] for crf in crfs], dim=1),
        end_transitions=torch.stack([crf.end_transitions[None, ...] for crf in crfs], dim=1),
    )

    padded_emissions, lengths = pad_packed_sequence(pack=emissions, batch_first=False)
    mask = lengths_to_mask(lengths=lengths, batch_first=False)
    padded_tags, _ = pad_packed_sequence(pack=tags, batch_first=False)

    tgt = torch.stack([
        crfs[index]._compute_score(padded_emissions[..., index, :], padded_tags[..., index], mask)
        for index in range(num_conjugates)
    ], dim=1)

    assert torch.allclose(out, tgt, rtol=1e-3, atol=1e-3)

    out_grad, = torch.autograd.grad(
        out, emissions.data, torch.ones_like(out),
        retain_graph=False, create_graph=False, only_inputs=True,
    )
    tgt_grad, = torch.autograd.grad(
        tgt, emissions.data, torch.ones_like(tgt),
        retain_graph=False, create_graph=False, only_inputs=True,
    )

    assert torch.allclose(out_grad, tgt_grad, rtol=1e-3, atol=1e-3)


@given(
    data=st.data(),
    lengths=length_lists(),
    num_tags=num_tags_integers(),
    num_conjugates=num_tags_integers(),
)
def test_compute_log_partitions_given_emissions(data, lengths, num_tags, num_conjugates):
    crf = CRF(num_tags=num_tags)

    emissions = data.draw(
        conjugated_emissions_packs(
            lengths=lengths, num_tags=num_tags, num_conjugates=num_conjugates))

    instr = build_crf_batched_instr(
        lengths=torch.tensor(lengths, dtype=torch.long),
        sorted_indices=emissions.sorted_indices,
    )

    out = compute_log_partitions(
        emissions=emissions,
        instr=instr,
        transitions=crf.transitions[None, None, ...],
        start_transitions=crf.start_transitions[None, None, ...],
        end_transitions=crf.end_transitions[None, None, ...],
        unit=log.fill_unit(crf.transitions),
    )

    padded_emissions, lengths = pad_packed_sequence(pack=emissions, batch_first=False)
    mask = lengths_to_mask(lengths=lengths, batch_first=False)

    tgt = torch.stack([
        crf._compute_normalizer(padded_emissions[..., index, :], mask)
        for index in range(num_conjugates)
    ], dim=1)

    assert torch.allclose(out, tgt, rtol=1e-3, atol=1e-3)

    out_grad, = torch.autograd.grad(
        out, emissions.data, torch.ones_like(out),
        retain_graph=False, create_graph=False, only_inputs=True,
    )
    tgt_grad, = torch.autograd.grad(
        tgt, emissions.data, torch.ones_like(tgt),
        retain_graph=False, create_graph=False, only_inputs=True,
    )

    assert torch.allclose(out_grad, tgt_grad, rtol=1e-3, atol=1e-3)


@given(
    data=st.data(),
    lengths=length_lists(),
    num_tags=num_tags_integers(),
    num_conjugates=num_tags_integers(),
)
def test_compute_log_partitions_given_crfs(data, lengths, num_tags, num_conjugates):
    crfs = [CRF(num_tags=num_tags) for _ in range(num_conjugates)]

    emissions = data.draw(
        conjugated_emissions_packs(
            lengths=lengths, num_tags=num_tags, num_conjugates=num_conjugates))

    instr = build_crf_batched_instr(
        lengths=torch.tensor(lengths, dtype=torch.long),
        sorted_indices=emissions.sorted_indices,
    )

    out = compute_log_partitions(
        emissions=emissions,
        instr=instr,
        transitions=torch.stack([crf.transitions[None, ...] for crf in crfs], dim=1),
        start_transitions=torch.stack([crf.start_transitions[None, ...] for crf in crfs], dim=1),
        end_transitions=torch.stack([crf.end_transitions[None, ...] for crf in crfs], dim=1),
        unit=log.fill_unit(crfs[0].transitions),
    )

    padded_emissions, lengths = pad_packed_sequence(pack=emissions, batch_first=False)
    mask = lengths_to_mask(lengths=lengths, batch_first=False)

    tgt = torch.stack([
        crfs[index]._compute_normalizer(padded_emissions[..., index, :], mask)
        for index in range(num_conjugates)
    ], dim=1)

    assert torch.allclose(out, tgt, rtol=1e-3, atol=1e-3)

    out_grad, = torch.autograd.grad(
        out, emissions.data, torch.ones_like(out),
        retain_graph=False, create_graph=False, only_inputs=True,
    )
    tgt_grad, = torch.autograd.grad(
        tgt, emissions.data, torch.ones_like(tgt),
        retain_graph=False, create_graph=False, only_inputs=True,
    )

    assert torch.allclose(out_grad, tgt_grad, rtol=1e-3, atol=1e-3)
