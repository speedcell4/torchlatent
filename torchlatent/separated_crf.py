from collections import Counter
from logging import getLogger
from typing import Tuple, Callable

import torch
from torch import Tensor
from torch import nn
from torch.nn import init
from torchglyph.vocab import Vocab

from torchlatent import CrfDecoderABC
from torchlatent.functional import logsumexp

logger = getLogger(__name__)


def build_syn_sem_mapping(vocab: Vocab, out_token: str, sep: str) -> Tuple[Tensor, Tensor]:
    syn, sem = Counter(), Counter()
    for tag in vocab.itos:
        if tag == out_token:
            syn.update(out_token)
            sem.update(None)
        else:
            x, y = tag.split(sep)
            syn.update(x)
            sem.update(y)

    syn = Vocab(counter=syn, unk_token=None, pad_token=None)
    sem = Vocab(counter=sem, unk_token=None, pad_token=None)
    logger.info(f'syn.vocab({len(syn)}) => {syn.itos}')
    logger.info(f'sem.vocab({len(sem)}) => {sem.itos}')

    tag_syn, tag_sem = [], []
    for tag in vocab.itos:
        if tag == out_token:
            x, y = out_token, None
        else:
            x, y = tag.split(sep)
        tag_syn.append(syn.stoi[x])
        tag_sem.append(sem.stoi[x])

    syn = torch.tensor(tag_syn, dtype=torch.long)
    sem = torch.tensor(tag_sem, dtype=torch.long)
    return syn, sem


class SepCrfDecoderABC(CrfDecoderABC):
    aggregate_fn: Callable

    @classmethod
    def from_vocab(cls, vocab: Vocab, out_token: str, sep: str) -> 'SepCrfDecoderABC':
        syn_mapping, sem_mapping = build_syn_sem_mapping(vocab=vocab, out_token=out_token, sep=sep)
        return SepCrfDecoderABC(syn_mapping=syn_mapping, sem_mapping=sem_mapping)

    def __init__(self, syn_mapping: Tensor, sem_mapping: Tensor) -> None:
        super(SepCrfDecoderABC, self).__init__(num_packs=None)

        num_syn = syn_mapping.max().item() + 1
        num_sem = sem_mapping.max().item() + 1

        self.syn_transitions = nn.Parameter(torch.empty((num_syn, num_syn), requires_grad=True))
        self.syn_start_transitions = nn.Parameter(torch.empty((num_syn,), requires_grad=True))
        self.syn_end_transitions = nn.Parameter(torch.empty((num_syn,), requires_grad=True))

        self.sem_transitions = nn.Parameter(torch.empty((num_sem, num_sem), requires_grad=True))
        self.sem_start_transitions = nn.Parameter(torch.empty((num_sem,), requires_grad=True))
        self.sem_end_transitions = nn.Parameter(torch.empty((num_sem,), requires_grad=True))

        self.register_buffer('syn', syn_mapping)
        self.register_buffer('sem', sem_mapping)

        self.reset_parameters()

    def reset_parameters(self, bound: float = 0.01) -> None:
        init.uniform_(self.syn_transitions, -bound, +bound)
        init.uniform_(self.syn_start_transitions, -bound, +bound)
        init.uniform_(self.syn_end_transitions, -bound, +bound)

        init.uniform_(self.sem_transitions, -bound, +bound)
        init.uniform_(self.sem_start_transitions, -bound, +bound)
        init.uniform_(self.sem_end_transitions, -bound, +bound)

    def _obtain_parameters(self, *args, **kwargs):
        syn_transitions = self.syn_transitions[self.syn[:, None], self.syn[None, :]]
        syn_start_transitions = self.syn_start_transitions[self.syn]
        syn_end_transitions = self.syn_end_transitions[self.syn]

        sem_transitions = self.sem_transitions[self.sem[:, None], self.sem[None, :]]
        sem_start_transitions = self.sem_start_transitions[self.sem]
        sem_end_transitions = self.sem_end_transitions[self.sem]

        transitions = torch.stack([syn_transitions, sem_transitions], dim=-1)
        start_transitions = torch.stack([syn_start_transitions, sem_start_transitions], dim=-1)
        end_transitions = torch.stack([syn_end_transitions, sem_end_transitions], dim=-1)

        transitions = self.__class__.aggregate_fn(transitions, dim=-1)
        start_transitions = self.__class__.aggregate_fn(start_transitions, dim=-1)
        end_transitions = self.__class__.aggregate_fn(end_transitions, dim=-1)

        return transitions, start_transitions, end_transitions


class SumCrfDecoder(SepCrfDecoderABC):
    aggregate_fn = torch.sum


class LsmCrfDecoder(SepCrfDecoderABC):
    aggregate_fn = logsumexp
