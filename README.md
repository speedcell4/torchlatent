# TorchLatent

![Unit Tests](https://github.com/speedcell4/torchlatent/workflows/Unit%20Tests/badge.svg)
![Upload Python Package](https://github.com/speedcell4/torchlatent/workflows/Upload%20Python%20Package/badge.svg)

## Requirements

- Python 3.7
- PyTorch 1.4.0 

## Installation

`python3 -m pip install git+https://github.com/speedcell4/torchlatent.git`

## Quickstart

Simply provide your batch-level `log_potentials` along with its `lengths`, and the corresponding target label sequence `target`, then call `CrfDecoder.fit` to obtain the log-likelihood or `CrfDecoder.decode` to search the most likely label sequence.

```python
import torch

from torchlatent.crf import CrfDecoder

batch_size = 7
sentence_length = 11
num_tags = 13

layer = CrfDecoder(num_tags=num_tags, batch_first=True)

log_potentials = torch.randn((batch_size, sentence_length, num_tags))
lengths = torch.randint(0, sentence_length, (batch_size,)) + 1
target = torch.randint(0, num_tags, (batch_size, sentence_length))

print(layer.fit(log_potentials=log_potentials, target=target, lengths=lengths))
print(layer.decode(log_potentials=log_potentials, lengths=lengths))
```

However, handling the sentences in mini-batch requires cumbersomely compiling their indices on-the-fly. Thus, preparing their indices `instr` and `seq_ptr` through calling `build_crf_batch_instr` and `build_seq_ptr` ahead and converting your `log_potentials` and `target` to `PackedSequence` form can bring you further accelerations.

```python
from torch.nn.utils.rnn import pack_padded_sequence
from torchlatent.crf import build_crf_batch_instr, build_seq_ptr

log_potentials = pack_padded_sequence(
    log_potentials, lengths=lengths,
    batch_first=True, enforce_sorted=False,
)
instr = build_crf_batch_instr(lengths=lengths, device=log_potentials.data.device)
seq_ptr = build_seq_ptr(lengths=lengths, device=log_potentials.data.device)
target = pack_padded_sequence(
    target, lengths=lengths,
    batch_first=True, enforce_sorted=False,
)

print(layer.fit(log_potentials=log_potentials, target=target, seq_ptr=seq_ptr, instr=instr))
print(layer.decode(log_potentials=log_potentials, seq_ptr=seq_ptr, instr=instr))
```

## Latent Structures and Utilities

- [x] Conditional Random Fields (CRF)
- [ ] Non-Projective Dependency Tree (Matrix-tree Theorem)
- [ ] Probabilistic Context-free Grammars (PCFG)
- [ ] Dependency Model with Valence (DMV)

## Thanks

This library is greatly inspired by [torch-struct](https://github.com/harvardnlp/pytorch-struct).
