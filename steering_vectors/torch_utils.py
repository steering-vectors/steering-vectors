import torch
from torch import nn


def untuple_tensor(x: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return x[0] if isinstance(x, tuple) else x


def get_module(model: nn.Module, name: str) -> nn.Module:
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def clear_all_forward_hooks(model: nn.Module) -> None:
    """Clear all forward hooks from the given model"""
    model._forward_hooks.clear()
    for _name, submodule in model.named_modules():
        submodule._forward_hooks.clear()
