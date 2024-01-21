from collections import defaultdict
from typing import NamedTuple, Optional

import torch
from torch import Tensor, nn
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from .layer_matching import LayerType, ModelLayerConfig, guess_and_enhance_layer_config
from .record_activations import record_activations
from .steering_vector import SteeringVector


class SteeringVectorTrainingSample(NamedTuple):
    positive_prompt: str
    negative_prompt: str


@torch.no_grad()
def train_steering_vector(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    training_samples: list[SteeringVectorTrainingSample] | list[tuple[str, str]],
    layers: Optional[list[int]] = None,
    layer_type: LayerType = "decoder_block",
    layer_config: Optional[ModelLayerConfig] = None,
    move_to_cpu: bool = False,
    read_token_index: int = -1,
    show_progress: bool = False,
    # TODO: add more options to control training
) -> SteeringVector:
    """
    Train a steering vector for the given model.

    Args:
        model: The model to train the steering vector for
        tokenizer: The tokenizer to use
        training_samples: A list of training samples, where each sample is a tuple of
            (positive_prompt, negative_prompt). The steering vector approximate the
            difference between the positive prompt and negative prompt activations.
        layers: A list of layer numbers to train the steering vector on. If None, train
            on all layers.
        layer_type: The type of layer to train the steering vector on. Default is
            "decoder_block".
        layer_config: A dictionary mapping layer types to layer matching functions.
            If not provided, this will be inferred automatically.
        move_to_cpu: If True, move the activations to the CPU before training. Default False.
        read_token_index: The index of the token to read the activations from. Default -1, meaning final token.
    """
    layer_config = guess_and_enhance_layer_config(model, layer_config, layer_type)
    pos_activations: dict[int, list[Tensor]] = defaultdict(list)
    neg_activations: dict[int, list[Tensor]] = defaultdict(list)
    # TODO: batching
    for pos_prompt, neg_prompt in tqdm(
        training_samples, disable=not show_progress, desc="Training steering vector"
    ):
        pos_acts = _extract_activations(
            model,
            tokenizer,
            pos_prompt,
            layer_type=layer_type,
            layer_config=layer_config,
            layers=layers,
            read_token_index=read_token_index,
        )
        neg_acts = _extract_activations(
            model,
            tokenizer,
            neg_prompt,
            layer_type=layer_type,
            layer_config=layer_config,
            layers=layers,
            read_token_index=read_token_index,
        )
        for layer_num, pos_act in pos_acts.items():
            if move_to_cpu:
                pos_act = pos_act.cpu()
            pos_activations[layer_num].append(pos_act)
        for layer_num, neg_act in neg_acts.items():
            if move_to_cpu:
                neg_act = neg_act.cpu()
            neg_activations[layer_num].append(neg_act)
    layer_activations = {}
    for layer_num in pos_activations.keys():
        layer_pos_acts = pos_activations[layer_num]
        layer_neg_acts = neg_activations[layer_num]
        # TODO: allow controlling how to combine activations, not just mean
        direction_vec = (
            torch.stack(layer_pos_acts) - torch.stack(layer_neg_acts)
        ).mean(dim=0)
        layer_activations[layer_num] = direction_vec
    return SteeringVector(layer_activations, layer_type)


def _extract_activations(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    layer_type: LayerType,
    layer_config: ModelLayerConfig,
    layers: list[int] | None,
    read_token_index: int,
) -> dict[int, Tensor]:
    input = tokenizer(prompt, return_tensors="pt").to(model.device)
    results = {}
    with record_activations(
        model, layer_type, layer_config, layer_nums=layers
    ) as record:
        model(**input)
    for layer_num, activation in record.items():
        results[layer_num] = activation[-1][0, read_token_index].detach()
    return results
