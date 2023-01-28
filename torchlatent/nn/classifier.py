import torch
from torch import Tensor
from torch import nn
from torch.nn import init


class Classifier(nn.Module):
    def __init__(self, bias: bool = False, *, in_features: int, out_features: int, num_conjugates: int) -> None:
        super(Classifier, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_conjugates = num_conjugates

        self.weight = nn.Parameter(torch.empty((num_conjugates, out_features, in_features)))
        self.bias = nn.Parameter(torch.empty((num_conjugates, out_features,))) if bias else None

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        bound = (6.0 / self.in_features) ** 0.5
        init.uniform_(self.weight, a=-bound, b=+bound)

        if self.bias is not None:
            init.zeros_(self.bias)

    def extra_repr(self) -> str:
        return ', '.join([
            f'in_features={self.in_features}',
            f'out_features={self.out_features}',
            f'num_conjugates={self.num_conjugates}',
            f'bias={self.bias is not None}',
        ])

    def forward(self, tensor: Tensor) -> Tensor:
        tensor = torch.einsum('...cx,cyx->...cy', tensor, self.weight)
        if self.bias is not None:
            tensor = tensor + self.bias

        return tensor
