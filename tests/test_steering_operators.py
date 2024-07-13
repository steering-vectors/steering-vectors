import pytest
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizer

from steering_vectors.steering_operators import (
    ablation_operator,
    ablation_then_addition_operator,
    addition_operator,
)
from steering_vectors.steering_vector import SteeringVector


def test_ablation_operator_removes_the_projection_of_the_steering_vec() -> None:
    act = torch.randn(1, 3, 30)
    steering_vec = torch.randn(30)
    operator = ablation_operator()
    delta = operator(act, steering_vec.reshape(1, 1, -1))
    steered_act = act + delta

    assert steered_act @ steering_vec == pytest.approx(0, abs=1e-4)


def test_addition_operator_acts_as_identity() -> None:
    act = torch.randn(1, 3, 30)
    steering_vec = torch.randn(30)
    operator = addition_operator()
    delta = operator(act, steering_vec.reshape(1, 1, -1))
    assert torch.allclose(delta, steering_vec)


def test_ablation_then_addition_operator_applies_both_ablation_and_addition() -> None:
    act = torch.randn(1, 3, 30)
    steering_vec = 5 * torch.randn(30)
    operator = ablation_then_addition_operator()
    delta = operator(act, steering_vec.reshape(1, 1, -1))
    steered_act = act + delta

    steering_vec_dir = steering_vec / torch.norm(steering_vec)

    # the component on the resulting activation on the steering direction should match the steering vector
    assert (steered_act @ steering_vec_dir).sum() == pytest.approx(
        3 * steering_vec.norm(), abs=1e-4
    )


@torch.no_grad()
def test_ablation_operator_works_with_patch_activations(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizer,
) -> None:
    inputs = tokenizer("Hello, world", return_tensors="pt")
    original_hidden_states = model(**inputs, output_hidden_states=True).hidden_states
    vec = torch.randn(768)
    steering_vector = SteeringVector(
        layer_activations={1: vec},
        layer_type="decoder_block",
    )
    steering_vector.patch_activations(model, operator=ablation_operator())
    patched_hidden_states = model(**inputs, output_hidden_states=True).hidden_states

    # layer 1 is where the patch occurs
    assert not torch.equal(original_hidden_states[2], patched_hidden_states[2])
    assert original_hidden_states[2].shape == patched_hidden_states[2].shape
    # the projection of the steering vector on the hidden state should be zero
    assert (patched_hidden_states[2] @ vec).sum() == pytest.approx(0, abs=1e-3)


@torch.no_grad()
def test_ablation_then_addition_operator_works_with_patch_activations(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizer,
) -> None:
    inputs = tokenizer("Hello, world", return_tensors="pt")
    original_hidden_states = model(**inputs, output_hidden_states=True).hidden_states
    vec = torch.randn(768)
    steering_vector = SteeringVector(
        layer_activations={1: vec},
        layer_type="decoder_block",
    )
    steering_vector.patch_activations(model, operator=ablation_then_addition_operator())
    patched_hidden_states = model(**inputs, output_hidden_states=True).hidden_states

    # layer 1 is where the patch occurs
    assert not torch.equal(original_hidden_states[2], patched_hidden_states[2])
    assert original_hidden_states[2].shape == patched_hidden_states[2].shape
    # the projection of the steering vector on the hidden state be 3 times the norm of the steering vector squared
    # since the ablation operator first removes the projection of the steering vector before adding the steering vector
    assert (patched_hidden_states[2] @ vec).sum() == pytest.approx(
        3 * vec.norm() ** 2, abs=1e-3
    )


@torch.no_grad()
def test_addition_operator_matches_default_behavior_with_steering_vector_apply(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizer,
) -> None:
    inputs = tokenizer("Hello, world", return_tensors="pt")
    vec = torch.randn(768)
    steering_vector = SteeringVector(
        layer_activations={1: vec},
        layer_type="decoder_block",
    )
    with steering_vector.apply(model, operator=ablation_then_addition_operator()):
        default_hidden_states = model(**inputs, output_hidden_states=True).hidden_states
    with steering_vector.apply(model, operator=ablation_then_addition_operator()):
        add_op_hidden_states = model(**inputs, output_hidden_states=True).hidden_states

    # layer 1 is where the patch occurs
    assert torch.equal(default_hidden_states[2], add_op_hidden_states[2])
