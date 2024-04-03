from typing import Any

import pytest
import torch
from transformers import (
    GPT2LMHeadModel,
)

from steering_vectors.torch_utils import (
    clear_all_forward_hooks,
    get_module,
    untuple_tensor,
)


def test_clear_all_forward_hooks(model: GPT2LMHeadModel) -> None:
    input = torch.tensor([[123, 456, 789]])
    module = get_module(model, "transformer.h.3")

    def hook_fn(_m: Any, _inputs: Any, outputs: Any) -> Any:
        activation = untuple_tensor(outputs)
        # just overwrite the activations with nonsense
        activation.fill_(1.7)
        return outputs

    base_output = model(input).logits
    module.register_forward_hook(hook_fn)
    hooked_output = model(input).logits

    assert not torch.allclose(base_output, hooked_output)
    clear_all_forward_hooks(model)

    cleared_output = model(input).logits
    assert torch.allclose(base_output, cleared_output)


def test_get_module_errors_if_module_not_found(model: GPT2LMHeadModel) -> None:
    with pytest.raises(LookupError):
        get_module(model, "transformer.h.100")
