import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torchrua import reversed_indices

from torchlatent.semiring import log


def scan_scores(semiring):
    def _scan_scores(emissions: PackedSequence, indices: Tensor,
                     transitions: Tensor, start_transitions: Tensor) -> Tensor:
        batch_size = emissions.batch_sizes[0].item()

        data = torch.empty_like(emissions.data, requires_grad=False)
        data[indices[:batch_size]] = start_transitions[:, :, None, :]

        start, end = 0, batch_size
        for batch_size in emissions.batch_sizes.detach().cpu().tolist()[1:]:
            last_start, last_end, start, end = start, start + batch_size, end, end + batch_size
            data[indices[start:end]] = semiring.bmm(
                semiring.mul(
                    data[indices[last_start:last_end]],
                    emissions.data[indices[last_start:last_end]],
                ),
                transitions[:batch_size],
            )

        return data[..., 0, :]

    return _scan_scores


scan_log_scores = scan_scores(log)


def compute_marginals(semiring):
    scan_scores_fn = scan_scores(semiring)

    def _compute_marginals(emissions: PackedSequence, transitions: Tensor,
                           start_transitions: Tensor, end_transitions: Tensor) -> Tensor:
        alpha = scan_scores_fn(
            emissions._replace(data=emissions.data[:, :, None, :]),
            torch.arange(emissions.data.size(0)),
            transitions,
            start_transitions,
        )

        beta = scan_scores_fn(
            emissions._replace(data=emissions.data[:, :, None, :]),
            reversed_indices(emissions),
            transitions.transpose(-2, -1),
            end_transitions,
        )

        return semiring.prod(torch.stack([
            alpha, beta, emissions.data
        ], dim=-1), dim=-1)

    return _compute_marginals


compute_log_marginals = compute_marginals(log)
