import warnings
from typing import Union

import torch

try:
    from torchglyph.pipe import Pipe
    from torchglyph.proc import GetRange, ToTensor, PackPtrSeq, ToDevice, Proc
    from torchlatent.proc import BuildCrfInstr, CollateCrfInstr


    class CrfInstrPipe(Pipe):
        def __init__(self, device: Union[int, torch.device]) -> None:
            super(CrfInstrPipe, self).__init__()
            self.with_(
                pre=GetRange(reverse=False) + BuildCrfInstr(),
                post=None,
                batch=CollateCrfInstr() + ToDevice(device=device),
            )

except ImportError:
    warnings.warn(f'torchglyph is required')
