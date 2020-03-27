from typing import Union

import torch
from torch.nn.utils.rnn import PackedSequence

from torchlatent.utilities import build_ins

try:
    from torchglyph.pipe import Pipe
    from torchglyph.proc import GetRange, ToTensor, PackPtrSeq, ToDevice, Proc


    class BuildIns(Proc):
        def __call__(self, indices: PackedSequence, *args, **kwargs):
            return build_ins(pack=indices)


    class PackedCrfPtrPipe(Pipe):
        def __init__(self, device: Union[int, torch.device], dtype: torch.dtype = torch.long) -> None:
            super(PackedCrfPtrPipe, self).__init__()
            self.with_(
                pre=GetRange(reverse=False),
                post=ToTensor(dtype=dtype),
                batch=PackPtrSeq(enforce_sorted=False) + BuildIns() + ToDevice(device=device),
            )
except ImportError:
    pass
