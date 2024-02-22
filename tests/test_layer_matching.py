from transformers import (
    GemmaForCausalLM,
    GPT2LMHeadModel,
    LlamaForCausalLM,
    MistralForCausalLM,
)

from steering_vectors.layer_matching import (
    ModelLayerConfig,
    _guess_block_matcher_from_layers,
    enhance_model_config_matchers,
    guess_decoder_block_matcher,
    guess_input_layernorm_matcher,
    guess_mlp_matcher,
    guess_post_attention_layernorm_matcher,
    guess_self_attn_matcher,
)

LlamaLayerConfig: ModelLayerConfig = {
    "decoder_block": "model.layers.{num}",
    "self_attn": "model.layers.{num}.self_attn",
    "mlp": "model.layers.{num}.mlp",
    "input_layernorm": "model.layers.{num}.input_layernorm",
    "post_attention_layernorm": "model.layers.{num}.post_attention_layernorm",
}

Gpt2LayerConfig: ModelLayerConfig = {
    "decoder_block": "transformer.h.{num}",
    "self_attn": "transformer.h.{num}.attn",
    "mlp": "transformer.h.{num}.mlp",
    "input_layernorm": "transformer.h.{num}.ln_1",
    "post_attention_layernorm": "transformer.h.{num}.ln_2",
}


def test_guess_block_matcher_from_layers() -> None:
    layers = [
        "x.e",
        "x.y.0",
        "x.y.0.attn",
        "x.y.1",
        "x.y.1.attn",
        "x.y.2",
        "x.y.2.attn",
        "x.lm_head",
    ]
    assert _guess_block_matcher_from_layers(layers) == "x.y.{num}"


def test_guess_matchers_for_llama(
    empty_llama_model: LlamaForCausalLM,
) -> None:
    assert guess_decoder_block_matcher(empty_llama_model) == "model.layers.{num}"
    assert guess_self_attn_matcher(empty_llama_model) == "model.layers.{num}.self_attn"
    assert guess_mlp_matcher(empty_llama_model) == "model.layers.{num}.mlp"
    assert (
        guess_input_layernorm_matcher(empty_llama_model)
        == "model.layers.{num}.input_layernorm"
    )
    assert (
        guess_post_attention_layernorm_matcher(empty_llama_model)
        == "model.layers.{num}.post_attention_layernorm"
    )


def test_guess_matchers_for_gpt2(model: GPT2LMHeadModel) -> None:
    assert guess_decoder_block_matcher(model) == "transformer.h.{num}"
    assert guess_self_attn_matcher(model) == "transformer.h.{num}.attn"
    assert guess_mlp_matcher(model) == "transformer.h.{num}.mlp"
    assert guess_input_layernorm_matcher(model) == "transformer.h.{num}.ln_1"
    assert guess_post_attention_layernorm_matcher(model) == "transformer.h.{num}.ln_2"


def test_guess_matchers_for_gemma(empty_gemma_model: GemmaForCausalLM) -> None:
    assert guess_decoder_block_matcher(empty_gemma_model) == "model.layers.{num}"
    assert guess_self_attn_matcher(empty_gemma_model) == "model.layers.{num}.self_attn"
    assert guess_mlp_matcher(empty_gemma_model) == "model.layers.{num}.mlp"
    assert (
        guess_input_layernorm_matcher(empty_gemma_model)
        == "model.layers.{num}.input_layernorm"
    )
    assert (
        guess_post_attention_layernorm_matcher(empty_gemma_model)
        == "model.layers.{num}.post_attention_layernorm"
    )


def test_guess_matchers_for_mistral(empty_mistral_model: MistralForCausalLM) -> None:
    assert guess_decoder_block_matcher(empty_mistral_model) == "model.layers.{num}"
    assert (
        guess_self_attn_matcher(empty_mistral_model) == "model.layers.{num}.self_attn"
    )
    assert guess_mlp_matcher(empty_mistral_model) == "model.layers.{num}.mlp"
    assert (
        guess_input_layernorm_matcher(empty_mistral_model)
        == "model.layers.{num}.input_layernorm"
    )
    assert (
        guess_post_attention_layernorm_matcher(empty_mistral_model)
        == "model.layers.{num}.post_attention_layernorm"
    )


def test_enhance_model_config_matchers_guesses_fields_if_not_provided_for_gpt2(
    model: GPT2LMHeadModel,
) -> None:
    enhanced_config = enhance_model_config_matchers(model, {})
    # it should correctly guess every field, resulting in the correct Gpt2LayerConfig
    assert enhanced_config == Gpt2LayerConfig


def test_enhance_model_config_matchers_guesses_fields_if_not_provided_for_llama(
    empty_llama_model: LlamaForCausalLM,
) -> None:
    enhanced_config = enhance_model_config_matchers(empty_llama_model, {})
    # it should correctly guess every field, resulting in the correct LlamaLayerConfig
    assert enhanced_config == LlamaLayerConfig


def test_enhance_model_config_matchers_leaves_provided_fields_as_is(
    model: GPT2LMHeadModel,
) -> None:
    enhanced_config = enhance_model_config_matchers(
        model, {"decoder_block": "my.{num}.matcher"}
    )
    assert enhanced_config["decoder_block"] == "my.{num}.matcher"
