import torch
from torch import Tensor
from torch import nn
from torch.nn import init


class Classifier(nn.Module):
    def __init__(self, bias: bool = False, *, num_conjugates: int,
                 in_features: int, out_features: int) -> None:
        super(Classifier, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_conjugates = num_conjugates

        self.weight = nn.Parameter(torch.empty((num_conjugates, out_features, in_features)))
        self.bias = nn.Parameter(torch.empty((num_conjugates, out_features,))) if bias else 0

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        init.zeros_(self.weight)

        if torch.is_tensor(self.bias):
            init.zeros_(self.bias)

    def extra_repr(self) -> str:
        return ', '.join([
            f'in_features={self.in_features}',
            f'out_features={self.out_features}',
            f'num_conjugates={self.num_conjugates}',
            f'bias={torch.is_tensor(self.bias)}',
        ])

    def forward(self, tensor: Tensor) -> Tensor:
        return torch.einsum('cox,...cx->...co', self.weight, tensor) + self.bias


class BiaffineClassifier(nn.Module):
    def __init__(self, bias: bool = False, *, num_conjugates: int,
                 in_features1: int, in_features2: int, out_features: int) -> None:
        super(BiaffineClassifier, self).__init__()

        self.in_features1 = in_features1
        self.in_features2 = in_features2
        self.out_features = out_features
        self.num_conjugates = num_conjugates

        self.weight0 = nn.Parameter(torch.empty((num_conjugates, out_features, in_features1, in_features2)))
        self.weight1 = nn.Parameter(torch.empty((num_conjugates, out_features, in_features1)))
        self.weight2 = nn.Parameter(torch.empty((num_conjugates, out_features, in_features2)))
        self.bias = nn.Parameter(torch.empty((num_conjugates, out_features))) if bias else 0

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        init.zeros_(self.weight0)
        init.zeros_(self.weight1)
        init.zeros_(self.weight2)

        if torch.is_tensor(self.bias):
            init.zeros_(self.bias)

    def extra_repr(self) -> str:
        return ', '.join([
            f'in_features1={self.in_features1}',
            f'in_features2={self.in_features2}',
            f'out_features={self.out_features}',
            f'num_conjugates={self.num_conjugates}',
            f'bias={torch.is_tensor(self.bias)}',
        ])

    def forward(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
        tensor0 = torch.einsum('coxy,...cx,...cy->...co', self.weight0, tensor1, tensor2)
        tensor1 = torch.einsum('cox,...cx->...co', self.weight1, tensor1)
        tensor2 = torch.einsum('coy,...cy->...co', self.weight2, tensor2)

        return tensor0 + tensor1 + tensor2 + self.bias
