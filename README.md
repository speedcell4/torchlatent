# TorchLatent

![Unit Tests](https://github.com/speedcell4/torchlatent/workflows/Unit%20Tests/badge.svg)
![Upload Python Package](https://github.com/speedcell4/torchlatent/workflows/Upload%20Python%20Package/badge.svg)

## Requirements

- Python 3.7
- PyTorch 1.6.0

## Installation

`python3 -m pip torchlatent`

## Quickstart

```python
import torch
from torch.nn.utils.rnn import pack_sequence

from torchlatent.crf import CrfDecoder

num_tags = 7
num_conjugates = 1

decoder = CrfDecoder(num_tags=num_tags, num_conjugates=num_conjugates)

emissions = pack_sequence([
    torch.randn((5, num_conjugates, num_tags)),
    torch.randn((2, num_conjugates, num_tags)),
    torch.randn((3, num_conjugates, num_tags)),
], enforce_sorted=False)
emissions.data.requires_grad_(True)

tags = pack_sequence([
    torch.randint(0, num_tags, (5, num_conjugates)),
    torch.randint(0, num_tags, (2, num_conjugates)),
    torch.randint(0, num_tags, (3, num_conjugates)),
], enforce_sorted=False)

print(decoder.fit(emissions, tags))
# tensor([[-10.7137],
#         [ -6.3496],
#         [ -7.9656]], grad_fn=<SubBackward0>)

print(decoder.decode(emissions))
# PackedSequence(data=tensor([[0],
#         [4],
#         [6],
#         [0],
#         [4],
#         [2],
#         [1],
#         [1],
#         [2],
#         [5]]), batch_sizes=tensor([3, 3, 2, 1, 1]), sorted_indices=tensor([0, 2, 1]), unsorted_indices=tensor([0, 2, 1]))
```

## Latent Structures and Utilities

- [x] Conditional Random Fields (CRF)
- [ ] Non-Projective Dependency Tree (Matrix-tree Theorem)
- [ ] Probabilistic Context-free Grammars (PCFG)
- [ ] Dependency Model with Valence (DMV)