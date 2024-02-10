from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor, nn
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from steering_vectors.aggregators import Aggregator, mean_aggregator

from .layer_matching import LayerType, ModelLayerConfig, guess_and_enhance_layer_config
from .record_activations import record_activations
from .steering_vector import SteeringVector


@dataclass
class SteeringVectorTrainingSample:
    positive_prompt: str
    negative_prompt: str
    read_positive_token_index: Optional[int] = None
    read_negative_token_index: Optional[int] = None


@torch.no_grad()
def train_steering_vector(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    training_samples: list[SteeringVectorTrainingSample] | list[tuple[str, str]],
    layers: Optional[list[int]] = None,
    layer_type: LayerType = "decoder_block",
    layer_config: Optional[ModelLayerConfig] = None,
    move_to_cpu: bool = False,
    read_token_index: int | Callable[[str], int] = -1,
    show_progress: bool = False,
    aggregator: Aggregator = mean_aggregator,
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
        show_progress: If True, show a progress bar. Default False.
        aggregator: A function that takes the positive and negative activations for a
            layer and returns a single vector. Default is mean_aggregator.
    """
    layer_config = guess_and_enhance_layer_config(model, layer_config, layer_type)
    pos_activations: dict[int, list[Tensor]] = defaultdict(list)
    neg_activations: dict[int, list[Tensor]] = defaultdict(list)

    if isinstance(training_samples[0], tuple):
        sv_training_samples: list[SteeringVectorTrainingSample] = [
            SteeringVectorTrainingSample(sample[0], sample[1], None, None)  # type: ignore
            for sample in training_samples
        ]
    else:
        sv_training_samples = training_samples  # type: ignore[assignment]

    # TODO: batching
    for training_sample in tqdm(
        sv_training_samples,
        disable=not show_progress,
        desc="Training steering vector",
    ):
        pos_index = _get_token_index(
            training_sample.read_positive_token_index,
            read_token_index,
            training_sample.positive_prompt,
        )
        neg_index = _get_token_index(
            training_sample.read_negative_token_index,
            read_token_index,
            training_sample.negative_prompt,
        )
        pos_acts = _extract_activations(
            model,
            tokenizer,
            training_sample.positive_prompt,
            layer_type=layer_type,
            layer_config=layer_config,
            layers=layers,
            read_token_index=pos_index,
        )
        neg_acts = _extract_activations(
            model,
            tokenizer,
            training_sample.negative_prompt,
            layer_type=layer_type,
            layer_config=layer_config,
            layers=layers,
            read_token_index=neg_index,
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
        direction_vec = aggregator(
            torch.stack(layer_pos_acts), torch.stack(layer_neg_acts)
        )
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


def _get_token_index(
    custom_idx: int | None, default_idx: int | Callable[[str], int], prompt: str
) -> int:
    if custom_idx is None:
        if isinstance(default_idx, int):
            return default_idx
        else:
            return default_idx(prompt)
    else:
        return custom_idx
