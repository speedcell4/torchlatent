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

decoder = CrfDecoder(num_tags=num_tags)

emissions = pack_sequence([
    torch.randn((5, num_tags)),
    torch.randn((2, num_tags)),
    torch.randn((3, num_tags)),
], enforce_sorted=False)
emissions.data.requires_grad_(True)

tags = pack_sequence([
    torch.randint(0, num_tags, (5,)),
    torch.randint(0, num_tags, (2,)),
    torch.randint(0, num_tags, (3,)),
], enforce_sorted=False)

print(decoder.fit(emissions, tags, reduction='sum'))
print(decoder.decode(emissions))

# tensor(-24.1321, grad_fn=<SumBackward0>)
# PackedSequence(data=tensor([1, 3, 5, 6, 0, 2, 5, 2, 1, 1]), batch_sizes=tensor([3, 3, 2, 1, 1]), sorted_indices=tensor([0, 2, 1]), unsorted_indices=tensor([0, 2, 1]))
```

## Latent Structures and Utilities

- [x] Conditional Random Fields (CRF)
- [ ] Non-Projective Dependency Tree (Matrix-tree Theorem)
- [ ] Probabilistic Context-free Grammars (PCFG)
- [ ] Dependency Model with Valence (DMV)

## Thanks

This library is greatly inspired by [torch-struct](https://github.com/harvardnlp/pytorch-struct).
