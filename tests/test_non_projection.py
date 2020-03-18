import torch

from torchlatent.non_projection import NonProjectionDistribution


def test_non_projection_corner_case():
    potential = torch.tensor([
        [[0, 0, 0, 0],
         [1, 0, 1, 2],
         [2, 3, 0, 4],
         [3, 5, 6, 0],
         ],
        [[0, 0, 0, 0],
         [4, 0, 1, 0],
         [5, 2, 0, 0],
         [0, 0, 0, 0]
         ],
    ], dtype=torch.float32).log()
    length = torch.tensor([4, 3], dtype=torch.long)

    dist = NonProjectionDistribution(potential[:, None, :, :], length)
    lhs = dist.log_partitions.exp()
    rhs = torch.tensor([153, 13], dtype=torch.float32)
    assert torch.allclose(lhs, rhs, atol=1e-5)