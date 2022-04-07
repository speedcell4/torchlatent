from typing import Union

from torch.nn.utils.rnn import PackedSequence

from torchrua import CattedSequence, PaddedSequence

Sequence = Union[CattedSequence, PackedSequence, PaddedSequence]
