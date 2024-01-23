from typing import Callable

from torch import Tensor
import torch


Aggregator = Callable[[Tensor, Tensor], Tensor]


def mean_aggregator(pos_acts: Tensor, neg_acts: Tensor) -> Tensor:
    """
    The default aggregator, which computes the mean of the difference between the
    positive and negative activations.
    """
    return (pos_acts - neg_acts).mean(dim=0)


@torch.no_grad()
def pca_aggregator(pos_acts: Tensor, neg_acts: Tensor) -> Tensor:
    """
    An aggregator that uses PCA to calculate a steering vector. This will always
    have norm of 1.
    """
    deltas = pos_acts - neg_acts
    neg_deltas = -1 * deltas
    vec = _uncentered_pca(torch.cat([deltas, neg_deltas]), k=1)[:, 0]
    # PCA might find the negative of the correct vector, so we need to check
    # that the vec aligns with most of the deltas, and flip it if not.
    sign = torch.sign(torch.mean(deltas @ vec))
    return sign * vec


def _uncentered_pca(data: Tensor, k: int = 1) -> Tensor:
    # No need to center the data since we flip the data around the origin in pca_aggregator
    u, _s, _v = torch.svd(torch.t(data))
    return u[:, :k]
