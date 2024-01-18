from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Generator, Optional

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from .layer_matching import (
    LayerType,
    ModelLayerConfig,
    collect_matching_layers,
    guess_and_enhance_layer_config,
)
from .torch_utils import get_module, untuple_tensor

PatchOperator = Callable[[Tensor, Tensor], Tensor]


@dataclass
class SteeringPatchHandle:
    """
    A handle that can be used to remove a steering patch from a model after
    running `steering_vector.patch_activations()`.
    """

    model_hooks: list[RemovableHandle]

    def remove(self) -> None:
        """Remove the steering patch from the model"""
        for hook in self.model_hooks:
            hook.remove()


@dataclass
class SteeringVector:
    """A steering vector that can be applied to a model."""

    layer_activations: dict[int, Tensor]
    layer_type: LayerType = "decoder_block"

    def patch_activations(
        self,
        model: nn.Module,
        layer_config: Optional[ModelLayerConfig] = None,
        operator: Optional[PatchOperator] = None,
        multiplier: float = 1.0,
        min_token_index: int = 0,
    ) -> SteeringPatchHandle:
        """
        Patch the activations of the given model with this steering vector.
        This will modify the model in-place, and return a handle that can be used to undo the patching.
        This method does the same thing as `apply`, but requires manually undoing the patching to
        restore the model to its original state. For most cases, `apply` is easier to use.

        Args:
            model: The model to patch
            layer_config: A dictionary mapping layer types to layer matching functions.
                If not provided, this will be inferred automatically.
            operator: A function that takes the original activation and the steering vector
                and returns a modified vector that is added to the original activation.
            multiplier: A multiplier to scale the patch activations. Default is 1.0.
            min_token_index: The minimum token index to apply the patch to. Default is 0.
        Example:
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
            >>> steering_vector = SteeringVector(...)
            >>> handle = steering_vector.patch_activations(model)
            >>> model.forward(...)
            >>> handle.remove()
        """
        layer_config = guess_and_enhance_layer_config(
            model, layer_config, self.layer_type
        )
        hooks: list[RemovableHandle] = []
        if self.layer_type not in layer_config:
            raise ValueError(
                f"layer_type {self.layer_type} not provided in layer config"
            )
        matcher = layer_config[self.layer_type]
        matching_layers = collect_matching_layers(model, matcher)
        for layer_num, target_activation in self.layer_activations.items():
            layer_name = matching_layers[layer_num]

            target_activation = multiplier * self.layer_activations[layer_num]

            module = get_module(model, layer_name)
            handle = module.register_forward_hook(
                # create the hook via function call since python only creates new scopes on functions
                _create_additive_hook(
                    target_activation.reshape(1, 1, -1), min_token_index, operator
                )
            )
            hooks.append(handle)
        return SteeringPatchHandle(hooks)

    @contextmanager
    def apply(
        self,
        model: nn.Module,
        layer_config: Optional[ModelLayerConfig] = None,
        operator: Optional[PatchOperator] = None,
        multiplier: float = 1.0,
        min_token_index: int = 0,
    ) -> Generator[None, None, None]:
        """
        Apply this steering vector to the given model.

        Args:
            model: The model to patch
            layer_config: A dictionary mapping layer types to layer matching functions.
                If not provided, this will be inferred automatically.
            operator: A function that takes the original activation and the steering vector
                and returns a modified vector that is added to the original activation.
            multiplier: A multiplier to scale the patch activations. Default is 1.0.
            min_token_index: The minimum token index to apply the patch to. Default is 0.
        Example:
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
            >>> steering_vector = SteeringVector(...)
            >>> with steering_vector.apply(model):
            >>>     model.forward(...)
        """
        try:
            handle = self.patch_activations(
                model=model,
                layer_config=layer_config,
                operator=operator,
                multiplier=multiplier,
                min_token_index=min_token_index,
            )
            yield
        finally:
            handle.remove()


def _create_additive_hook(
    target_activation: Tensor,
    min_token_index: int,
    operator: PatchOperator | None,
) -> Any:
    """Create a hook function that adds the given target_activation to the model output"""

    def hook_fn(_m: Any, _inputs: Any, outputs: Any) -> Any:
        original_tensor = untuple_tensor(outputs)
        act = target_activation.to(original_tensor.device)
        delta = act
        if operator is not None:
            delta = operator(original_tensor, act)
        mask = torch.ones(original_tensor.shape[1])
        mask[:min_token_index] = 0
        mask = mask.reshape(1, -1, 1)
        mask = mask.to(original_tensor.device)
        original_tensor[None] = original_tensor + (mask * delta)
        return outputs

    return hook_fn
