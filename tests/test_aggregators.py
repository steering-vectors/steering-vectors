import pytest
import torch
from torch.nn.functional import cosine_similarity

from steering_vectors.aggregators import (
    logistic_aggregator,
    mean_aggregator,
    pca_aggregator,
)


def test_logistic_aggregator() -> None:
    delta = torch.randn(100)
    pos = torch.randn(64, 100)
    noise = torch.randn(64, 100) * 0.2
    neg = pos - delta + noise

    vec = logistic_aggregator()(pos, neg)
    assert vec.shape == (100,)
    assert vec.norm() == pytest.approx(1)
    assert cosine_similarity(vec, delta, dim=0) > 0.99


def test_logistic_aggregator_with_custom_sklearn_params() -> None:
    delta = torch.randn(100)
    pos = torch.randn(64, 100)
    noise = torch.randn(64, 100) * 0.2
    neg = pos - delta + noise
    aggregator = logistic_aggregator(sklearn_kwargs={"penalty": None, "max_iter": 50})

    vec = aggregator(pos, neg)
    assert vec.shape == (100,)
    assert vec.norm() == pytest.approx(1)
    assert cosine_similarity(vec, delta, dim=0) > 0.99


def test_pca_aggregator_with_single_difference() -> None:
    delta = torch.randn(100)
    pos = torch.randn(100)
    neg = pos - delta

    vec = pca_aggregator()(pos.unsqueeze(0), neg.unsqueeze(0))
    assert vec.shape == (100,)
    assert vec.norm() == pytest.approx(1)
    assert cosine_similarity(vec, delta, dim=0) == pytest.approx(1)


def test_pca_aggregator_with_synth_data() -> None:
    delta = torch.randn(100)
    pos = torch.randn(50, 100)
    noise = torch.randn(50, 100) * 0.2
    neg = pos - delta.unsqueeze(0) + noise

    vec = pca_aggregator()(pos, neg)
    assert vec.shape == (100,)
    assert vec.norm() == pytest.approx(1)
    assert cosine_similarity(vec, delta, dim=0) > 0.99


def test_pca_aggregator_and_mean_aggregator_give_similar_results() -> None:
    delta = torch.randn(100)
    pos = torch.randn(50, 100)
    noise = torch.randn(50, 100)
    neg = pos - delta.unsqueeze(0) + noise

    pca_vec = pca_aggregator()(pos, neg)
    mean_vec = mean_aggregator()(pos, neg)
    assert pca_vec.shape == (100,)
    assert mean_vec.shape == (100,)
    assert cosine_similarity(mean_vec, pca_vec, dim=0) > 0.99
