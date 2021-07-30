# TorchLatent

![Unit Tests](https://github.com/speedcell4/torchlatent/workflows/Unit%20Tests/badge.svg)
![Upload Python Package](https://github.com/speedcell4/torchlatent/workflows/Upload%20Python%20Package/badge.svg)
[![Downloads](https://pepy.tech/badge/torchrua)](https://pepy.tech/project/torchrua)

## Requirements

- Python 3.7
- PyTorch 1.6.0

## Installation

`python3 -m pip torchlatent`

## Usage

```python
import torch
from torch.nn.utils.rnn import pack_sequence

from torchlatent.crf import CrfDecoder

num_tags = 3
num_conjugates = 1

decoder = CrfDecoder(num_tags=num_tags, num_conjugates=num_conjugates)

emissions = pack_sequence([
    torch.randn((5, num_conjugates, num_tags), requires_grad=True),
    torch.randn((2, num_conjugates, num_tags), requires_grad=True),
    torch.randn((3, num_conjugates, num_tags), requires_grad=True),
], enforce_sorted=False)

tags = pack_sequence([
    torch.randint(0, num_tags, (5, num_conjugates)),
    torch.randint(0, num_tags, (2, num_conjugates)),
    torch.randint(0, num_tags, (3, num_conjugates)),
], enforce_sorted=False)

print(decoder.fit(emissions=emissions, tags=tags))
# tensor([[-6.7424],
#         [-5.1288],
#         [-2.7283]], grad_fn=<SubBackward0>)

print(decoder.decode(emissions=emissions))
# PackedSequence(data=tensor([[2],
#         [0],
#         [1],
#         [0],
#         [2],
#         [0],
#         [2],
#         [0],
#         [1],
#         [2]]), 
#         batch_sizes=tensor([3, 3, 2, 1, 1]),
#         sorted_indices=tensor([0, 2, 1]),
#         unsorted_indices=tensor([0, 2, 1]))

print(decoder.marginals(emissions=emissions))
# tensor([[[0.1040, 0.1001, 0.7958]],
#
#         [[0.5736, 0.0784, 0.3479]],
#
#         [[0.0932, 0.8797, 0.0271]],
#
#         [[0.6558, 0.0472, 0.2971]],
#
#         [[0.2740, 0.1109, 0.6152]],
#
#         [[0.4811, 0.2163, 0.3026]],
#
#         [[0.2321, 0.3478, 0.4201]],
#
#         [[0.4987, 0.1986, 0.3027]],
#
#         [[0.2029, 0.5888, 0.2083]],
#
#         [[0.2802, 0.2358, 0.4840]]], grad_fn=<AddBackward0>)
```

## Latent Structures

- [ ] Conditional Random Fields (CRF)
    - [x] Conjugated
    - [ ] Dynamic Transition Matrix
    - [ ] Second-order
    - [ ] Variant-order
- [ ] Tree CRF
- [ ] Non-Projective Dependency Tree (Matrix-tree Theorem)
- [ ] Probabilistic Context-free Grammars (PCFG)
- [ ] Dependency Model with Valence (DMV)