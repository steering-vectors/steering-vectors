from typing import Callable

from sklearn.linear_model import LinearRegression, LogisticRegression
from torch import Tensor
import torch
import torch.nn.functional as F


Aggregator = Callable[[Tensor, Tensor], Tensor]


@torch.no_grad()
def linear_aggregator(pos_acts: Tensor, neg_acts: Tensor) -> Tensor:
    """
    An aggregator that uses linear regression to calculate a steering vector.
    """
    return _get_normalized_regression_coef(
        pos_acts, neg_acts, LinearRegression(fit_intercept=False)
    )


@torch.no_grad()
def logistic_aggregator(pos_acts: Tensor, neg_acts: Tensor) -> Tensor:
    """
    An aggregator that uses logistic regression to calculate a steering vector.
    """
    return _get_normalized_regression_coef(
        pos_acts, neg_acts, LogisticRegression(fit_intercept=False)
    )


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


def _get_normalized_regression_coef(
    pos_acts: Tensor,
    neg_acts: Tensor,
    regression: LinearRegression | LogisticRegression,
) -> Tensor:
    mean_acts = torch.stack([pos_acts, neg_acts]).mean(dim=0)

    pos_acts_centered = pos_acts - mean_acts
    neg_acts_centered = neg_acts - mean_acts

    reg = regression.fit(
        torch.cat([pos_acts_centered, neg_acts_centered])
        .cpu()
        .to(torch.float32)
        .numpy(),
        torch.cat([torch.ones(pos_acts.shape[0]), -1 * torch.ones(neg_acts.shape[0])])
        .cpu()
        .to(torch.float32)
        .numpy(),
    )

    coef = torch.tensor([reg.coef_]).to(pos_acts.device).to(pos_acts.dtype)
    normalized_coef = F.normalize(coef, dim=-1)

    return normalized_coef.view(-1).clone()


def _uncentered_pca(data: Tensor, k: int = 1) -> Tensor:
    # No need to center the data since we flip the data around the origin in pca_aggregator
    u, _s, _v = torch.svd(torch.t(data))
    return u[:, :k]
