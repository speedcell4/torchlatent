import torch
from torch.nn.utils.rnn import pack_sequence

from torchlatent import CrfDecoder
from torchlatent.separated_crf import LsmCrfDecoder, SumCrfDecoder


def test_separated_crf():
    num_tags = 70
    syn_tensor = torch.randint(0, 25, (num_tags,))
    sem_tensor = torch.randint(0, 25, (num_tags,))
    crf1 = CrfDecoder(num_tags)
    crf2 = LsmCrfDecoder(syn_tensor, sem_tensor)
    crf3 = SumCrfDecoder(syn_tensor, sem_tensor)

    emissions = pack_sequence([
        torch.randn((5, num_tags), requires_grad=True),
        torch.randn((2, num_tags), requires_grad=True),
        torch.randn((3, num_tags), requires_grad=True),
    ], enforce_sorted=False)

    tags = pack_sequence([
        torch.randint(0, num_tags, (5,)),
        torch.randint(0, num_tags, (2,)),
        torch.randint(0, num_tags, (3,)),
    ], enforce_sorted=False)

    print(crf1.fit(emissions, tags).sum().neg())
    print(crf2.fit(emissions, tags).sum().neg())
    print(crf3.fit(emissions, tags).sum().neg())
