import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Literal

from torch import nn

LayerMatcher = str | Callable[[nn.Module, int], str]


def collect_matching_layers(model: nn.Module, layer_matcher: LayerMatcher) -> list[str]:
    """
    Find all layers in the model that match the layer_matcher, in order by layer_num.
    layer_matcher can be a string formatted like "transformer.h.{num}.mlp" or a callable
    If layer_matcher is a callable, it should take in a model and layer_num and return
    a string representing the layer name corresponding to that layer number.
    If layer_matcher is a string, it's considered a template and MUST contain a "{num}" portion
    """
    matcher_callable = _layer_matcher_to_callable(layer_matcher)
    all_layer_names = dict(model.named_modules()).keys()
    matching_layers = []
    for layer_num in range(len(all_layer_names)):
        layer_name = matcher_callable(model, layer_num)
        if layer_name in all_layer_names:
            matching_layers.append(layer_name)
        else:
            break
    return matching_layers


def get_num_matching_layers(model: nn.Module, layer_matcher: LayerMatcher) -> int:
    """Returns the number of layers in the model that match the layer_matcher"""
    return len(collect_matching_layers(model, layer_matcher))


def get_layer_name(
    model: nn.Module, layer_matcher: LayerMatcher, layer_num: int
) -> str:
    matcher_callable = _layer_matcher_to_callable(layer_matcher)
    layer_num = fix_neg_layer_num(model, layer_matcher, layer_num)
    return matcher_callable(model, layer_num)


def fix_neg_layer_num(
    model: nn.Module, layer_matcher: LayerMatcher, layer_num: int
) -> int:
    """Helper to handle negative layer nums. If layer_num is negative, return len(layers) + layer_num"""
    if layer_num >= 0:
        return layer_num
    matching_layers = collect_matching_layers(model, layer_matcher)
    return len(matching_layers) + layer_num


def get_layer_by_name(model: nn.Module, layer_name: str) -> nn.Module:
    return dict(model.named_modules())[layer_name]


def _layer_matcher_to_callable(
    layer_matcher: LayerMatcher,
) -> Callable[[nn.Module, int], str]:
    if isinstance(layer_matcher, str):
        if "{num}" not in layer_matcher:
            raise ValueError(
                "layer_matcher must be a callable or a string containing {num}"
            )
        return lambda _model, layer_num: layer_matcher.format(num=layer_num)
    return layer_matcher


LAYER_GUESS_RE = r"^([^\d]+)\.([\d]+)(.*)$"


def guess_decoder_block_matcher(model: nn.Module) -> str | None:
    """
    Guess the hidden layer matcher for a given model. This is a best guess and may not always be correct.
    """
    return _guess_block_matcher_from_layers(dict(model.named_modules()).keys())


def guess_mlp_matcher(model: nn.Module) -> str | None:
    """
    Guess the mlp layer matcher for a given model. This is a best guess and may not always be correct.
    """
    return _guess_block_matcher_from_layers(
        dict(model.named_modules()).keys(), filter=lambda guess: "mlp" in guess
    )


def guess_self_attn_matcher(model: nn.Module) -> str | None:
    """
    Guess the self attention layer matcher for a given model. This is a best guess and may not always be correct.
    """
    return _guess_block_matcher_from_layers(
        dict(model.named_modules()).keys(),
        filter=lambda guess: "attn" in guess or "attention" in guess,
    )


def guess_input_layernorm_matcher(model: nn.Module) -> str | None:
    """
    Guess the input layernorm layer matcher for a given model. This is a best guess and may not always be correct.
    """
    return _guess_block_matcher_from_layers(
        dict(model.named_modules()).keys(),
        filter=lambda guess: "ln_1" in guess or "input_layernorm" in guess,
    )


def guess_post_attention_layernorm_matcher(model: nn.Module) -> str | None:
    """
    Guess the post-attention layernorm layer matcher for a given model. This is a best guess and may not always be correct.
    """
    return _guess_block_matcher_from_layers(
        dict(model.named_modules()).keys(),
        filter=lambda guess: "ln_2" in guess or "post_attention_layernorm" in guess,
    )


# broken into a separate function for easier testing
def _guess_block_matcher_from_layers(
    layers: Iterable[str], filter: Callable[[str], bool] | None = None
) -> str | None:
    counts_by_guess: dict[str, int] = defaultdict(int)

    for layer in layers:
        if re.match(LAYER_GUESS_RE, layer):
            guess = re.sub(LAYER_GUESS_RE, r"\1.{num}\3", layer)
            if filter is None or filter(guess):
                counts_by_guess[guess] += 1
    if len(counts_by_guess) == 0:
        return None

    # score is higher for guesses that match more often, are and shorter in length
    guess_scores = [
        (guess, count + 1 / len(guess)) for guess, count in counts_by_guess.items()
    ]
    return max(guess_scores, key=lambda x: x[1])[0]


LayerType = Literal[
    "decoder_block", "self_attn", "mlp", "input_layernorm", "post_attention_layernorm"
]

ModelLayerConfig = dict[LayerType, LayerMatcher]


_LAYER_TYPE_TO_GUESSER: dict[LayerType, Callable[[nn.Module], str | None]] = {
    "decoder_block": guess_decoder_block_matcher,
    "self_attn": guess_self_attn_matcher,
    "mlp": guess_mlp_matcher,
    "input_layernorm": guess_input_layernorm_matcher,
    "post_attention_layernorm": guess_post_attention_layernorm_matcher,
}


def enhance_model_config_matchers(
    model: nn.Module, config: ModelLayerConfig, layer_type: LayerType | None = None
) -> ModelLayerConfig:
    """Returns a new layer config, attempting to fill-in missing layer matchers"""
    enhanced_config: ModelLayerConfig = {**config}
    types_to_guess: Iterable[LayerType] = (
        [layer_type] if layer_type is not None else _LAYER_TYPE_TO_GUESSER.keys()
    )
    for guess_layer_type in types_to_guess:
        if guess_layer_type not in config and (
            layer_matcher := _LAYER_TYPE_TO_GUESSER[guess_layer_type](model)
        ):
            enhanced_config[guess_layer_type] = layer_matcher
    return enhanced_config


def guess_and_enhance_layer_config(
    model: nn.Module,
    layer_config: ModelLayerConfig | None = None,
    layer_type: LayerType | None = None,
) -> ModelLayerConfig:
    """
    Try to guess any missing parts of the layer config, after checking against predefined configs.
    If layer_type is provided, only guess the layer_matcher for that layer type.
    """
    layer_config = enhance_model_config_matchers(
        model, layer_config or {}, layer_type=layer_type
    )
    return layer_config
