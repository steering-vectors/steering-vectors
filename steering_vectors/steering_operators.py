import torch

from steering_vectors.steering_vector import PatchDeltaOperator


def addition_operator() -> PatchDeltaOperator:
    """
    Simply add the steering vector to the activation. This is the default
    """

    def _addition_operator(
        _original_activation: torch.Tensor, steering_vector: torch.Tensor
    ) -> torch.Tensor:
        return steering_vector

    return _addition_operator


def ablation_operator() -> PatchDeltaOperator:
    """
    Erase the projection of the steering vector entirely from activation.
    NOTE: This will ignore the steering vector multiplier param, and will always
    erase the steering vector entirely from the activation.
    """

    def _ablation_operator(
        original_activation: torch.Tensor, steering_vector: torch.Tensor
    ) -> torch.Tensor:
        norm_vec = steering_vector / torch.norm(steering_vector)
        return -1 * norm_vec * (original_activation @ norm_vec.squeeze()).unsqueeze(-1)

    return _ablation_operator


def ablation_then_addition_operator() -> PatchDeltaOperator:
    """
    This acts as a combination of the ablation and addition operators. It first removes the
    projection of the steering vector from the activation, and then adds the steering vector
    """

    _ablation_operator = ablation_operator()
    _addition_operator = addition_operator()

    def _ablation_then_addition_operator(
        original_activation: torch.Tensor, steering_vector: torch.Tensor
    ) -> torch.Tensor:
        ablation_delta = _ablation_operator(original_activation, steering_vector)
        addition_delta = _addition_operator(original_activation, steering_vector)
        return ablation_delta + addition_delta

    return _ablation_then_addition_operator
