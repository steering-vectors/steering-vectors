from collections import defaultdict
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any, cast

from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from .layer_matching import (
    LayerType,
    ModelLayerConfig,
    collect_matching_layers,
    guess_and_enhance_layer_config,
)
from .torch_utils import get_module, untuple_tensor


@contextmanager
def record_activations(
    model: nn.Module,
    layer_type: LayerType = "decoder_block",
    layer_config: ModelLayerConfig | None = None,
    clone_activations: bool = True,
    layer_nums: Sequence[int] | None = None,
) -> Generator[dict[int, list[Tensor]], None, None]:
    """
    Record the model activations at each layer of type `layer_type`.
    This function will record every forward pass through the model
    at all layers of the given layer_type.

    Args:
        model: The model to record activations from
        layer_type: The type of layer to record activations from
        layer_config: A dictionary mapping layer types to layer matching functions.
            If not provided, this will be inferred automatically.
        clone_activations: If True, clone the activations before recording them. Default True.
        layer_nums: A list of layer numbers to record activations from. If None, record
            activations from all matching layers
    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
        >>> with record_activations(model, layer_type="decoder_block") as recorded_activations:
        >>>     model.forward(...)
        >>> # recorded_activations is a dictionary mapping layer numbers to lists of activations
    """
    recorded_activations: dict[int, list[Tensor]] = defaultdict(list)
    layer_config = guess_and_enhance_layer_config(model, layer_config)
    if layer_type not in layer_config:
        raise ValueError(f"layer_type {layer_type} not provided in layer config")
    matcher = layer_config[layer_type]
    matching_layers = collect_matching_layers(model, matcher)
    hooks: list[RemovableHandle] = []
    for layer_num, layer_name in enumerate(matching_layers):
        if layer_nums is not None and layer_num not in layer_nums:
            continue
        module = get_module(model, layer_name)
        hook_fn = _create_read_hook(
            layer_num, recorded_activations, clone_activations=clone_activations
        )
        hooks.append(module.register_forward_hook(hook_fn))
    try:
        yield recorded_activations
    finally:
        for hook in hooks:
            hook.remove()


def _create_read_hook(
    layer_num: int, records: dict[int, list[Tensor]], clone_activations: bool
) -> Any:
    """Create a hook function that records the model activation at layer_num"""

    def hook_fn(_m: Any, _inputs: Any, outputs: Any) -> Any:
        activation = untuple_tensor(outputs)
        if not isinstance(cast(Any, activation), Tensor):
            raise ValueError(
                f"Expected a Tensor reading model activations, got {type(activation)}"
            )
        if clone_activations:
            activation = activation.clone().detach()
        records[layer_num].append(activation)
        return outputs

    return hook_fn
