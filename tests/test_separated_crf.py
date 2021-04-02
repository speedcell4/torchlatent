import torch
from hypothesis import given, strategies as st

from tests.strategies import length_lists, num_tags_integers, emission_packs, tag_packs
from torchlatent import CrfDecoder
from torchlatent.separated_crf import LsmCrfDecoder, SumCrfDecoder


@given(
    data=st.data(),
    lengths=length_lists(),
    num_tags=num_tags_integers(),
    num_syn_tags=num_tags_integers(),
    num_sem_tags=num_tags_integers(),
)
def test_separated_crf(data, lengths, num_tags, num_syn_tags, num_sem_tags):
    emissions = data.draw(emission_packs(lengths=lengths, num_tags=num_tags))
    tags = data.draw(tag_packs(lengths=lengths, num_tags=num_tags))

    syn_mapping = torch.randint(0, num_syn_tags, (num_tags,))
    sem_mapping = torch.randint(0, num_sem_tags, (num_tags,))
    crf1 = CrfDecoder(num_tags)
    crf2 = LsmCrfDecoder(syn_mapping, sem_mapping)
    crf3 = SumCrfDecoder(syn_mapping, sem_mapping)

    _ = crf1.fit(emissions, tags).sum().neg()
    _ = crf2.fit(emissions, tags).sum().neg()
    _ = crf3.fit(emissions, tags).sum().neg()
