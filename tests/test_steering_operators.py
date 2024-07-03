import pytest
import torch

from steering_vectors.steering_operators import (
    ablation_operator,
    ablation_then_addition_operator,
    addition_operator,
)


def test_ablation_operator_removes_the_projection_of_the_steering_vec() -> None:
    act = torch.randn(30)
    steering_vec = torch.randn(30)
    operator = ablation_operator()
    delta = operator(act, steering_vec)
    steered_act = act + delta

    assert steered_act @ steering_vec == pytest.approx(0, abs=1e-6)


def test_addition_operator_acts_as_identity() -> None:
    act = torch.randn(30)
    steering_vec = torch.randn(30)
    operator = addition_operator()
    delta = operator(act, steering_vec)
    assert torch.allclose(delta, steering_vec)


def test_ablation_then_addition_operator_applies_both_ablation_and_addition() -> None:
    act = torch.randn(30)
    steering_vec = 5 * torch.randn(30)
    operator = ablation_then_addition_operator()
    delta = operator(act, steering_vec)
    steered_act = act + delta

    steering_vec_dir = steering_vec / torch.norm(steering_vec)

    # the component on the resulting activation on the steering direction should match the steering vector
    assert steered_act @ steering_vec_dir == pytest.approx(
        steering_vec.norm(), abs=1e-5
    )
