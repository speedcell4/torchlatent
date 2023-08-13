from torch import Tensor
from torch.types import Device


def get_device(*tensors: Tensor, device: Device = None) -> Device:
    raise NotImplementedError


def broadcast_devices(*tensors: Tensor, device: Device = None) -> Device:
    raise NotImplementedError
