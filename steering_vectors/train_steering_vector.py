from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import PreTrainedTokenizerBase

from steering_vectors.aggregators import Aggregator, mean_aggregator
from steering_vectors.token_utils import adjust_read_indices_for_padding, fix_pad_token
from steering_vectors.utils import batchify

from .layer_matching import LayerType, ModelLayerConfig, guess_and_enhance_layer_config
from .record_activations import record_activations
from .steering_vector import SteeringVector


@dataclass
class SteeringVectorTrainingSample:
    positive_str: str
    negative_str: str
    read_positive_token_index: int | None = None
    read_negative_token_index: int | None = None


@torch.no_grad()
def extract_activations(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    training_samples: Sequence[SteeringVectorTrainingSample | tuple[str, str]],
    layers: list[int] | None = None,
    layer_type: LayerType = "decoder_block",
    layer_config: ModelLayerConfig | None = None,
    move_to_cpu: bool = False,
    read_token_index: int | Callable[[str], int] = -1,
    show_progress: bool = False,
    batch_size: int = 1,
    tqdm_desc: str = "Extracting activations",
) -> tuple[dict[int, list[Tensor]], dict[int, list[Tensor]]]:
    """
    Extract activations from the model for the given training samples.

    Args:
        model: The model to extract activations from
        tokenizer: The tokenizer to use
        training_samples: A list of training samples, where each sample is a tuple of
            (positive_str, negative_str). The steering vector approximate the
            difference between the positive prompt and negative prompt activations.
        layers: A list of layer numbers to extract activations from. If None, extract
            from all layers.
        layer_type: The type of layer to extract activations from. Default is
            "decoder_block".
        layer_config: A dictionary mapping layer types to layer matching functions.
            If not provided, this will be inferred automatically.
        move_to_cpu: If True, move the activations to the CPU before training. Default False.
        read_token_index: The index of the token to read the activations from. Default -1, meaning final token.
        show_progress: If True, show a progress bar. Default False.
        batch_size: The batch size to use. Default 1.
        tqdm_desc: The description to use for the progress bar. Default "Extracting activations".

    Returns:
        A tuple of two dictionaries. The first dictionary maps layer numbers to lists of
        positive activations, and the second dictionary maps layer numbers to lists of
        negative activations.
    """
    fix_pad_token(tokenizer)
    layer_config = guess_and_enhance_layer_config(model, layer_config, layer_type)
    pos_acts_by_layer: dict[int, list[Tensor]] = defaultdict(list)
    neg_acts_by_layer: dict[int, list[Tensor]] = defaultdict(list)

    for raw_batch in batchify(
        training_samples,
        batch_size=batch_size,
        show_progress=show_progress,
        tqdm_desc=tqdm_desc,
    ):
        batch = _formalize_batch(raw_batch)
        pos_indices = []
        neg_indices = []
        pos_prompts = []
        neg_prompts = []
        for training_sample in batch:
            pos_prompts.append(training_sample.positive_str)
            pos_indices.append(
                _get_token_index(
                    training_sample.read_positive_token_index,
                    read_token_index,
                    training_sample.positive_str,
                )
            )
            neg_prompts.append(training_sample.negative_str)
            neg_indices.append(
                _get_token_index(
                    training_sample.read_negative_token_index,
                    read_token_index,
                    training_sample.negative_str,
                )
            )
        pos_acts = _extract_activations(
            model,
            tokenizer,
            pos_prompts,
            layer_type=layer_type,
            layer_config=layer_config,
            layers=layers,
            read_token_indices=pos_indices,
        )
        neg_acts = _extract_activations(
            model,
            tokenizer,
            neg_prompts,
            layer_type=layer_type,
            layer_config=layer_config,
            layers=layers,
            read_token_indices=neg_indices,
        )
        for layer_num, pos_act in pos_acts.items():
            if move_to_cpu:
                pos_act = pos_act.cpu()
            pos_acts_by_layer[layer_num].append(pos_act)
        for layer_num, neg_act in neg_acts.items():
            if move_to_cpu:
                neg_act = neg_act.cpu()
            neg_acts_by_layer[layer_num].append(neg_act)

    return pos_acts_by_layer, neg_acts_by_layer


def aggregate_activations(
    pos_acts_by_layer: dict[int, list[Tensor]],
    neg_acts_by_layer: dict[int, list[Tensor]],
    aggregator: Aggregator = mean_aggregator(),
) -> dict[int, Tensor]:
    """
    Apply the aggregator to the positive and negative activations for each layer.

    Args:
        pos_acts_by_layer: A dictionary mapping layer numbers to lists of positive
            activations.
        neg_acts_by_layer: A dictionary mapping layer numbers to lists of negative
            activations.
        aggregator: A function that takes the positive and negative activations for a
            layer and returns a single vector.

    Returns:
        A dictionary mapping layer numbers to the aggregated activations.
    """
    layer_activations = {}
    for layer_num in pos_acts_by_layer.keys():
        layer_pos_acts = pos_acts_by_layer[layer_num]
        layer_neg_acts = neg_acts_by_layer[layer_num]
        direction_vec = aggregator(
            torch.concat(layer_pos_acts), torch.concat(layer_neg_acts)
        )
        layer_activations[layer_num] = direction_vec
    return layer_activations


@torch.no_grad()
def train_steering_vector(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    training_samples: Sequence[SteeringVectorTrainingSample | tuple[str, str]],
    layers: list[int] | None = None,
    layer_type: LayerType = "decoder_block",
    layer_config: ModelLayerConfig | None = None,
    move_to_cpu: bool = False,
    read_token_index: int | Callable[[str], int] = -1,
    show_progress: bool = False,
    aggregator: Aggregator = mean_aggregator(),
    batch_size: int = 1,
    tqdm_desc: str = "Training steering vector",
) -> SteeringVector:
    """
    Train a steering vector for the given model.

    Args:
        model: The model to train the steering vector for
        tokenizer: The tokenizer to use
        training_samples: A list of training samples, where each sample is a tuple of
            (positive_str, negative_str). The steering vector approximate the
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
    pos_acts, neg_acts = extract_activations(
        model,
        tokenizer,
        training_samples,
        layers=layers,
        layer_type=layer_type,
        layer_config=layer_config,
        move_to_cpu=move_to_cpu,
        read_token_index=read_token_index,
        show_progress=show_progress,
        batch_size=batch_size,
        tqdm_desc=tqdm_desc,
    )
    layer_activations = aggregate_activations(pos_acts, neg_acts, aggregator)
    return SteeringVector(layer_activations, layer_type)


def _formalize_batch(
    batch: Sequence[SteeringVectorTrainingSample | tuple[str, str]],
) -> list[SteeringVectorTrainingSample]:
    return [_formalize_sample(sample) for sample in batch]


def _formalize_sample(
    sample: SteeringVectorTrainingSample | tuple[str, str],
) -> SteeringVectorTrainingSample:
    if isinstance(sample, tuple):
        return SteeringVectorTrainingSample(sample[0], sample[1])
    else:
        return sample


def _extract_activations(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    layer_type: LayerType,
    layer_config: ModelLayerConfig,
    layers: list[int] | None,
    read_token_indices: Sequence[int],
) -> dict[int, Tensor]:
    input = tokenizer(prompts, return_tensors="pt", padding=True)
    adjusted_read_indices = adjust_read_indices_for_padding(
        torch.tensor(read_token_indices), input["attention_mask"]
    )
    batch_indices = torch.arange(len(prompts))
    results = {}
    with record_activations(
        model, layer_type, layer_config, layer_nums=layers
    ) as record:
        model(**input.to(model.device))
    for layer_num, activation in record.items():
        results[layer_num] = activation[-1][
            batch_indices.to(activation[-1].device),
            adjusted_read_indices.to(activation[-1].device),
        ].detach()
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
