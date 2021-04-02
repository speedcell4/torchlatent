import torch
from hypothesis import given, strategies as st

from tests.strategies import length_lists, num_tags_integers, emission_packs, tag_packs
from torchlatent import CrfDecoder
from torchlatent.factorized_crf import SumCrfDecoder, LseCrfDecoder


@given(
    data=st.data(),
    lengths=length_lists(),
    num_tags=num_tags_integers(),
    num_syn_tags=num_tags_integers(),
    num_sem_tags=num_tags_integers(),
)
def test_factorized_crf(data, lengths, num_tags, num_syn_tags, num_sem_tags):
    emissions = data.draw(emission_packs(lengths=lengths, num_tags=num_tags))
    tags = data.draw(tag_packs(lengths=lengths, num_tags=num_tags))

    syn_mapping = torch.randint(0, num_syn_tags, (num_tags,))
    sem_mapping = torch.randint(0, num_sem_tags, (num_tags,))
    crf = CrfDecoder(num_tags=num_tags)
    lse_crf = LseCrfDecoder(syn_mapping=syn_mapping, sem_mapping=sem_mapping)
    sum_crf = SumCrfDecoder(syn_mapping=syn_mapping, sem_mapping=sem_mapping)

    _ = crf.fit(emissions, tags).sum().neg()
    _ = lse_crf.fit(emissions, tags).sum().neg()
    _ = sum_crf.fit(emissions, tags).sum().neg()
