import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torchrua import reversed_indices, select_last

from torchlatent.semiring import log, max


def scan_scores(semiring):
    def _scan_scores(emissions: PackedSequence, indices: Tensor,
                     transitions: Tensor, start_transitions: Tensor) -> Tensor:
        """

        Args:
            emissions: [t1, c1, n]
            indices: [t1]
            transitions: [t2, c2, n, n]
            start_transitions: [t2, c2, n]

        Returns:
            [t, c, n]
        """

        batch_size = emissions.batch_sizes[0].item()

        tc_size = torch.broadcast_shapes(
            emissions.data.size()[:2],
            transitions.size()[:2],
            start_transitions.size()[:2],
        )

        data = torch.empty(
            (*tc_size, *emissions.data.size()[2:]),
            dtype=emissions.data.dtype, device=emissions.data.device, requires_grad=False)
        data[indices[:batch_size]] = start_transitions[:, :, None, :]

        start, end = 0, batch_size
        for batch_size in emissions.batch_sizes.detach().cpu().tolist()[1:]:
            last_start, last_end, start, end = start, start + batch_size, end, end + batch_size
            ans = semiring.bmm(
                semiring.mul(
                    data[indices[last_start:last_end]],
                    emissions.data[indices[last_start:last_end]],
                ),
                transitions[:batch_size],
            )
            data[indices[start:end]] = ans

        return data[..., 0, :]

    return _scan_scores


scan_log_scores = scan_scores(log)
scan_max_scores = scan_scores(max)


def compute_partitions(semiring, scan_semi_scores):
    def compute_log_partitions(
            emissions: PackedSequence, transitions: Tensor,
            start_transitions: Tensor, end_transitions: Tensor) -> Tensor:
        scores = scan_semi_scores(
            emissions._replace(data=emissions.data[:, :, None, :]),
            torch.arange(emissions.data.size(0)),
            transitions,
            start_transitions,
        )
        scores = select_last(emissions._replace(data=scores), unsort=True)
        return semiring.bmm(scores[:, :, None, :], end_transitions[..., None])[..., 0, 0]

    return compute_log_partitions


scan_log_partitions = compute_partitions(semiring=log, scan_semi_scores=scan_log_scores)
scan_max_partitions = compute_partitions(semiring=max, scan_semi_scores=scan_max_scores)


def compute_marginals(semiring, scan_semi_scores):
    def _compute_marginals(emissions: PackedSequence, transitions: Tensor,
                           start_transitions: Tensor, end_transitions: Tensor) -> Tensor:
        alpha = scan_semi_scores(
            emissions._replace(data=emissions.data[:, :, None, :]),
            torch.arange(emissions.data.size(0)),
            transitions,
            start_transitions,
        )

        beta = scan_semi_scores(
            emissions._replace(data=emissions.data[:, :, None, :]),
            reversed_indices(emissions),
            transitions.transpose(-2, -1),
            end_transitions,
        )

        return semiring.prod(torch.stack([
            alpha, beta, emissions.data
        ], dim=-1), dim=-1)

    return _compute_marginals


compute_log_marginals = compute_marginals(log, scan_log_scores)
