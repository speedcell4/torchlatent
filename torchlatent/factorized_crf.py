from logging import getLogger
from typing import Tuple, Callable, List

import torch
from torch import Tensor

from torchlatent import CrfDecoderABC, CrfDecoder
from torchlatent.functional import logsumexp

logger = getLogger(__name__)


def factorize_syn_sem(tag: str, out_token: str, sep: str) -> Tuple[str, str]:
    if tag == out_token:
        syn, sem = f'{out_token}_syn', f'{out_token}_sem'
    else:
        syn, sem = tag.split(sep)
    return syn, sem


def build_syn_sem_mapping(tags: List[str], out_token: str, sep: str) -> Tuple[Tensor, Tensor]:
    syn_vocab, sem_vocab = {}, {}
    for tag in tags:
        syn, sem = factorize_syn_sem(tag, out_token, sep)
        if syn not in syn_vocab:
            syn_vocab[syn] = len(syn_vocab)
        if sem not in sem_vocab:
            sem_vocab[sem] = len(sem_vocab)

    logger.info(f'syn({len(syn_vocab)}) => {list(syn_vocab.keys())}')
    logger.info(f'sem({len(sem_vocab)}) => {list(sem_vocab.keys())}')

    syn_tags, sem_tags = [], []
    for tag in tags:
        syn, sem = factorize_syn_sem(tag, out_token, sep)
        syn_tags.append(syn_vocab[syn])
        sem_tags.append(sem_vocab[sem])

    syn_mapping = torch.tensor(syn_tags, dtype=torch.long)
    sem_mapping = torch.tensor(sem_tags, dtype=torch.long)
    return syn_mapping, sem_mapping


class FactorizedCrfDecoderABC(CrfDecoderABC):
    aggregate_fn: Callable

    @classmethod
    def from_tags(cls, tags: List[str], out_token: str, sep: str) -> 'FactorizedCrfDecoderABC':
        syn_mapping, sem_mapping = build_syn_sem_mapping(tags=tags, out_token=out_token, sep=sep)
        return FactorizedCrfDecoderABC(syn_mapping=syn_mapping, sem_mapping=sem_mapping)

    def __init__(self, syn_mapping: Tensor, sem_mapping: Tensor) -> None:
        super(FactorizedCrfDecoderABC, self).__init__(num_tags=syn_mapping.size(), num_conjugates=1)

        num_syn = syn_mapping.max().item() + 1
        num_sem = sem_mapping.max().item() + 1

        self.syn_crf = CrfDecoder(num_tags=num_syn)
        self.sem_crf = CrfDecoder(num_tags=num_sem)

        self.register_buffer('syn', syn_mapping)
        self.register_buffer('sem', sem_mapping)

    def reset_parameters(self, bound: float = 0.01) -> None:
        self.syn_crf.reset_parameters(bound=bound)
        self.sem_crf.reset_parameters(bound=bound)

    def obtain_parameters(self, *args, **kwargs):
        syn_transitions, syn_start_transitions, syn_end_transitions = \
            self.syn_crf.obtain_parameters(*args, **kwargs)
        syn_transitions = syn_transitions[..., self.syn[:, None], self.syn[None, :]]
        syn_start_transitions = syn_start_transitions[..., self.syn]
        syn_end_transitions = syn_end_transitions[..., self.syn]

        sem_transitions, sem_start_transitions, sem_end_transitions = \
            self.sem_crf.obtain_parameters(*args, **kwargs)
        sem_transitions = sem_transitions[..., self.sem[:, None], self.sem[None, :]]
        sem_start_transitions = sem_start_transitions[..., self.sem]
        sem_end_transitions = sem_end_transitions[..., self.sem]

        transitions = torch.stack([syn_transitions, sem_transitions], dim=-1)
        start_transitions = torch.stack([syn_start_transitions, sem_start_transitions], dim=-1)
        end_transitions = torch.stack([syn_end_transitions, sem_end_transitions], dim=-1)

        transitions = self.__class__.aggregate_fn(transitions, dim=-1)
        start_transitions = self.__class__.aggregate_fn(start_transitions, dim=-1)
        end_transitions = self.__class__.aggregate_fn(end_transitions, dim=-1)

        return transitions, start_transitions, end_transitions


class SumCrfDecoder(FactorizedCrfDecoderABC):
    aggregate_fn = torch.sum


class LseCrfDecoder(FactorizedCrfDecoderABC):
    aggregate_fn = logsumexp
