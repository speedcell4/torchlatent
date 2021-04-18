import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_sequence
from torchcrf import CRF
from torchrua import pad_packed_sequence, lengths_to_mask

from tests.strategies import length_lists, num_tags_integers, conjugated_emissions_packs, conjugated_tags_packs
from torchlatent import CrfDecoder, ConjugatedCrfDecoder
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


@given(
    data=st.data(),
    lengths=length_lists(),
    num_tags=num_tags_integers(),
    num_conjugates=num_tags_integers(),
)
def test_crf_decoder_given_emissions(data, lengths, num_tags, num_conjugates):
    crf_decoder = CrfDecoder(num_tags=num_tags, num_conjugates=1)
    tgt_crf = CRF(num_tags=num_tags)

    with torch.no_grad():
        crf_decoder.transitions.data = tgt_crf.transitions[None, None, :, :]
        crf_decoder.start_transitions.data = tgt_crf.start_transitions[None, None, :]
        crf_decoder.end_transitions.data = tgt_crf.end_transitions[None, None, :]

    emissions = data.draw(
        conjugated_emissions_packs(
            lengths=lengths, num_tags=num_tags, num_conjugates=num_conjugates))
    tags = data.draw(
        conjugated_tags_packs(
            lengths=lengths, num_tags=num_tags, num_conjugates=num_conjugates))

    padded_emissions, lengths = pad_packed_sequence(emissions, batch_first=False)
    mask = lengths_to_mask(lengths=lengths, batch_first=False)
    padded_tags, _ = pad_packed_sequence(tags, batch_first=False)

    instr = build_crf_batched_instr(lengths=lengths)

    our = crf_decoder.fit(emissions=emissions, tags=tags, instr=instr, reduction='none')

    tgt = torch.stack([
        tgt_crf.forward(
            emissions=padded_emissions[..., index, :],
            tags=padded_tags[..., index],
            mask=mask, reduction='none',
        )
        for index in range(num_conjugates)
    ], dim=1)

    assert torch.allclose(our, tgt, rtol=1e-3, atol=1e-3)

    out_grad, = torch.autograd.grad(
        our, emissions.data, torch.ones_like(our),
        retain_graph=False, create_graph=False, only_inputs=True,
    )
    tgt_grad, = torch.autograd.grad(
        tgt, emissions.data, torch.ones_like(tgt),
        retain_graph=False, create_graph=False, only_inputs=True,
    )

    assert torch.allclose(out_grad, tgt_grad, rtol=1e-3, atol=1e-3)

    out_pred = crf_decoder.decode(emissions=emissions, instr=instr)

    tgt_pred = [
        pack_sequence([
            torch.tensor(x, dtype=torch.long)
            for x in tgt_crf.decode(padded_emissions[..., index, :], mask=mask)
        ], enforce_sorted=False)
        for index in range(num_conjugates)
    ]
    tgt_pred = tgt_pred[0]._replace(data=torch.stack([
        t.data for t in tgt_pred
    ], dim=1))

    assert torch.equal(out_pred.data, tgt_pred.data)
    assert torch.equal(out_pred.batch_sizes, tgt_pred.batch_sizes)
    assert torch.equal(out_pred.sorted_indices, tgt_pred.sorted_indices)
    assert torch.equal(out_pred.unsorted_indices, tgt_pred.unsorted_indices)


@given(
    data=st.data(),
    lengths=length_lists(),
    num_tags=num_tags_integers(),
    num_conjugates=num_tags_integers(),
)
def test_crf_decoder_given_crfs(data, lengths, num_tags, num_conjugates):
    crf_decoder = [CrfDecoder(num_tags=num_tags) for _ in range(num_conjugates)]
    tgt_crf = [CRF(num_tags=num_tags) for _ in range(num_conjugates)]

    with torch.no_grad():
        for crf, tgt in zip(crf_decoder, tgt_crf):
            crf.transitions.data = tgt.transitions[None, None, :, :]
            crf.start_transitions.data = tgt.start_transitions[None, None, :]
            crf.end_transitions.data = tgt.end_transitions[None, None, :]

    crf_decoder = ConjugatedCrfDecoder(*crf_decoder)

    emissions = data.draw(
        conjugated_emissions_packs(
            lengths=lengths, num_tags=num_tags, num_conjugates=num_conjugates))
    tags = data.draw(
        conjugated_tags_packs(
            lengths=lengths, num_tags=num_tags, num_conjugates=num_conjugates))

    padded_emissions, lengths = pad_packed_sequence(emissions, batch_first=False)
    mask = lengths_to_mask(lengths=lengths, batch_first=False)
    padded_tags, _ = pad_packed_sequence(tags, batch_first=False)

    instr = build_crf_batched_instr(lengths=lengths)

    our = crf_decoder.fit(emissions=emissions, tags=tags, instr=instr, reduction='none')

    tgt = torch.stack([
        tgt_crf[index].forward(
            emissions=padded_emissions[..., index, :],
            tags=padded_tags[..., index],
            mask=mask, reduction='none',
        )
        for index in range(num_conjugates)
    ], dim=1)

    assert torch.allclose(our, tgt, rtol=1e-3, atol=1e-3)

    out_grad, = torch.autograd.grad(
        our, emissions.data, torch.ones_like(our),
        retain_graph=False, create_graph=False, only_inputs=True,
    )
    tgt_grad, = torch.autograd.grad(
        tgt, emissions.data, torch.ones_like(tgt),
        retain_graph=False, create_graph=False, only_inputs=True,
    )

    assert torch.allclose(out_grad, tgt_grad, rtol=1e-3, atol=1e-3)

    out_pred = crf_decoder.decode(emissions=emissions, instr=instr)

    tgt_pred = [
        pack_sequence([
            torch.tensor(x, dtype=torch.long)
            for x in tgt_crf[index].decode(padded_emissions[..., index, :], mask=mask)
        ], enforce_sorted=False)
        for index in range(num_conjugates)
    ]
    tgt_pred = tgt_pred[0]._replace(data=torch.stack([
        t.data for t in tgt_pred
    ], dim=1))

    assert torch.equal(out_pred.data, tgt_pred.data)
    assert torch.equal(out_pred.batch_sizes, tgt_pred.batch_sizes)
    assert torch.equal(out_pred.sorted_indices, tgt_pred.sorted_indices)
    assert torch.equal(out_pred.unsorted_indices, tgt_pred.unsorted_indices)


@given(
    data=st.data(),
    lengths=length_lists(),
    num_tags=num_tags_integers(),
)
def test_compute_log_scores_give_time_wise_transitions(data, lengths, num_tags):
    emissions_list = []
    tags_list = []
    transitions_list = []
    start_transitions_list = []
    end_transitions_list = []

    log_scores_list = []
    grad_list = []

    for length in lengths:
        emissions = pack_sequence([torch.randn((length, 1, num_tags), requires_grad=True)])
        tags = pack_sequence([torch.randint(0, num_tags, (length, 1))])
        transitions = torch.randn((length, 1, num_tags, num_tags), requires_grad=True)
        start_transitions = torch.randn((length, 1, num_tags), requires_grad=True)
        end_transitions = torch.randn((length, 1, num_tags), requires_grad=True)

        log_scores = compute_log_scores(
            emissions=emissions, tags=tags,
            transitions=transitions,
            start_transitions=start_transitions,
            end_transitions=end_transitions,
        )
        grad, = torch.autograd.grad(
            log_scores, emissions.data, torch.ones_like(log_scores),
        )

        emissions_list.append(emissions)
        tags_list.append(tags)
        transitions_list.append(transitions)
        start_transitions_list.append(start_transitions)
        end_transitions_list.append(end_transitions)
        log_scores_list.append(log_scores)
        grad_list.append(grad)

    out = torch.cat(log_scores_list, dim=0)
    out_grad = pack_sequence(grad_list, enforce_sorted=False).data

    emissions = pack_sequence([
        emission.data for emission in emissions_list], enforce_sorted=False)
    tags = pack_sequence([
        tag.data for tag in tags_list], enforce_sorted=False)
    transitions = pack_sequence([
        transition.data for transition in transitions_list], enforce_sorted=False)
    start_transitions = pack_sequence([
        start_transition.data for start_transition in start_transitions_list], enforce_sorted=False)
    end_transitions = pack_sequence([
        end_transition.data for end_transition in end_transitions_list], enforce_sorted=False)

    tgt = compute_log_scores(
        emissions=emissions, tags=tags,
        transitions=transitions.data,
        start_transitions=start_transitions.data,
        end_transitions=end_transitions.data,
    )
    tgt_grad, = torch.autograd.grad(
        tgt, emissions.data, torch.ones_like(tgt),
    )

    assert torch.allclose(out, tgt, rtol=1e-3, atol=1e-3)
    assert torch.allclose(out_grad, tgt_grad, rtol=1e-3, atol=1e-3)


@given(
    data=st.data(),
    lengths=length_lists(),
    num_tags=num_tags_integers(),
)
def test_compute_log_partitions_give_time_wise_transitions(data, lengths, num_tags):
    emissions_list = []
    transitions_list = []
    start_transitions_list = []
    end_transitions_list = []

    log_partitions_list = []
    grad_list = []

    for length in lengths:
        emissions = pack_sequence([torch.randn((length, 1, num_tags), requires_grad=True)])
        transitions = torch.randn((length, 1, num_tags, num_tags), requires_grad=True)
        start_transitions = torch.randn((length, 1, num_tags), requires_grad=True)
        end_transitions = torch.randn((length, 1, num_tags), requires_grad=True)
        instr = build_crf_batched_instr([length], None, device=torch.device('cpu'))

        log_partitions = compute_log_partitions(
            emissions=emissions, instr=instr,
            transitions=transitions,
            start_transitions=start_transitions,
            end_transitions=end_transitions,
            unit=log.fill_unit(transitions),
        )
        grad, = torch.autograd.grad(
            log_partitions, emissions.data, torch.ones_like(log_partitions),
        )

        emissions_list.append(emissions)
        transitions_list.append(transitions)
        start_transitions_list.append(start_transitions)
        end_transitions_list.append(end_transitions)
        log_partitions_list.append(log_partitions)
        grad_list.append(grad)

    out = torch.cat(log_partitions_list, dim=0)
    out_grad = pack_sequence(grad_list, enforce_sorted=False).data

    emissions = pack_sequence([
        emission.data for emission in emissions_list], enforce_sorted=False)
    transitions = pack_sequence([
        transition.data for transition in transitions_list], enforce_sorted=False)
    start_transitions = pack_sequence([
        start_transition.data for start_transition in start_transitions_list], enforce_sorted=False)
    end_transitions = pack_sequence([
        end_transition.data for end_transition in end_transitions_list], enforce_sorted=False)

    instr = build_crf_batched_instr(torch.tensor(lengths), None, device=torch.device('cpu'))
    tgt = compute_log_partitions(
        emissions=emissions, instr=instr,
        transitions=transitions.data,
        start_transitions=start_transitions.data,
        end_transitions=end_transitions.data,
        unit=log.fill_unit(transitions.data),
    )
    tgt_grad, = torch.autograd.grad(
        tgt, emissions.data, torch.ones_like(tgt),
    )

    assert torch.allclose(out, tgt, rtol=1e-3, atol=1e-3)
    assert torch.allclose(out_grad, tgt_grad, rtol=1e-3, atol=1e-3)


@given(
    data=st.data(),
    lengths=length_lists(),
    num_tags=num_tags_integers(),
)
def test_crf_give_time_wise_transitions(data, lengths, num_tags):
    emissions_list = []
    tags_list = []
    transitions_list = []
    start_transitions_list = []
    end_transitions_list = []

    loss_list = []
    grad_list = []
    pred_list = []

    for length in lengths:
        emissions = pack_sequence([torch.randn((length, 1, num_tags), requires_grad=True)])
        tags = pack_sequence([torch.randint(0, num_tags, (length, 1))])
        transitions = torch.randn((length, 1, num_tags, num_tags), requires_grad=True)
        start_transitions = torch.randn((length, 1, num_tags), requires_grad=True)
        end_transitions = torch.randn((length, 1, num_tags), requires_grad=True)

        crf = CrfDecoder(num_tags=num_tags)
        with torch.no_grad():
            crf.transitions.data = transitions.data
            crf.start_transitions.data = start_transitions.data
            crf.end_transitions.data = end_transitions.data
        loss = crf.fit(emissions=emissions, tags=tags)
        pred = crf.decode(emissions=emissions).data

        grad, = torch.autograd.grad(
            loss, emissions.data, torch.ones_like(loss),
        )

        emissions_list.append(emissions)
        tags_list.append(tags)
        transitions_list.append(transitions)
        start_transitions_list.append(start_transitions)
        end_transitions_list.append(end_transitions)
        loss_list.append(loss)
        grad_list.append(grad)
        pred_list.append(pred)

    out_loss = torch.cat(loss_list, dim=0)
    out_grad = pack_sequence(grad_list, enforce_sorted=False).data
    out_pred = pack_sequence(pred_list, enforce_sorted=False).data

    emissions = pack_sequence([
        emission.data for emission in emissions_list], enforce_sorted=False)
    tags = pack_sequence([
        tag.data for tag in tags_list], enforce_sorted=False)
    transitions = pack_sequence([
        transition.data for transition in transitions_list], enforce_sorted=False)
    start_transitions = pack_sequence([
        start_transition.data for start_transition in start_transitions_list], enforce_sorted=False)
    end_transitions = pack_sequence([
        end_transition.data for end_transition in end_transitions_list], enforce_sorted=False)

    crf = CrfDecoder(num_tags=num_tags)
    with torch.no_grad():
        crf.transitions.data = transitions.data
        crf.start_transitions.data = start_transitions.data
        crf.end_transitions.data = end_transitions.data

    tgt_loss = crf.fit(emissions=emissions, tags=tags)
    tgt_grad, = torch.autograd.grad(
        tgt_loss, emissions.data, torch.ones_like(tgt_loss),
    )
    tgt_pred = crf.decode(emissions=emissions).data

    assert torch.allclose(out_loss, tgt_loss, rtol=1e-3, atol=1e-3)
    assert torch.allclose(out_grad, tgt_grad, rtol=1e-3, atol=1e-3)
    assert torch.equal(out_pred, tgt_pred)
