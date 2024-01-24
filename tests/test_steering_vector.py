import torch
from transformers import GPT2LMHeadModel, LlamaForCausalLM, PreTrainedTokenizer

from steering_vectors.steering_vector import SteeringVector
from tests._original_caa.llama_wrapper import LlamaWrapper  # type: ignore


@torch.no_grad()
def test_SteeringVector_patch_activations(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizer,
) -> None:
    inputs = tokenizer("Hello, world", return_tensors="pt")
    original_hidden_states = model(**inputs, output_hidden_states=True).hidden_states
    patch = torch.randn(768)
    steering_vector = SteeringVector(
        layer_activations={1: patch},
        layer_type="decoder_block",
    )
    steering_vector.patch_activations(model)
    patched_hidden_states = model(**inputs, output_hidden_states=True).hidden_states

    # The first hidden state is the input embeddings, which are not patched
    assert torch.equal(original_hidden_states[0], patched_hidden_states[0])
    # next is the first decoder block, which is not patched
    assert torch.equal(original_hidden_states[1], patched_hidden_states[1])
    # next is the layer 1, where the patch occurs
    assert not torch.equal(original_hidden_states[2], patched_hidden_states[2])

    expected_hidden_state = original_hidden_states[2] + patch
    assert torch.equal(expected_hidden_state, patched_hidden_states[2])


@torch.no_grad()
def test_SteeringVector_apply(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizer,
) -> None:
    inputs = tokenizer("Hello, world", return_tensors="pt")
    original_hidden_states = model(**inputs, output_hidden_states=True).hidden_states
    patch = torch.randn(768)
    steering_vector = SteeringVector(
        layer_activations={1: patch},
        layer_type="decoder_block",
    )
    with steering_vector.apply(model):
        patched_hidden_states = model(**inputs, output_hidden_states=True).hidden_states

    # The first hidden state is the input embeddings, which are not patched
    assert torch.equal(original_hidden_states[0], patched_hidden_states[0])
    # next is the first decoder block, which is not patched
    assert torch.equal(original_hidden_states[1], patched_hidden_states[1])
    # next is the layer 1, where the patch occurs
    assert not torch.equal(original_hidden_states[2], patched_hidden_states[2])

    expected_hidden_state = original_hidden_states[2] + patch
    assert torch.equal(expected_hidden_state, patched_hidden_states[2])


@torch.no_grad()
def test_SteeringVector_apply_matches_original_caa(
    empty_llama_model: LlamaForCausalLM, llama_tokenizer: PreTrainedTokenizer
) -> None:
    model = empty_llama_model
    tokenizer = llama_tokenizer
    inputs = tokenizer("Hello, world", return_tensors="pt")
    patch1 = torch.randn(1024)
    patch2 = torch.randn(1024)
    steering_vector = SteeringVector(
        layer_activations={1: patch1, 2: patch2},
        layer_type="decoder_block",
    )
    original_logits = model(**inputs, output_hidden_states=True).logits
    with steering_vector.apply(model):
        sv_logits = model(**inputs, output_hidden_states=True).logits

    caa_model = LlamaWrapper(model, tokenizer)
    caa_model.set_add_activations(1, patch1)
    caa_model.set_add_activations(2, patch2)

    caa_logits = caa_model.get_logits(inputs["input_ids"])
    assert not torch.allclose(original_logits, sv_logits)
    assert torch.allclose(sv_logits, caa_logits)


@torch.no_grad()
def test_SteeringVector_patch_activations_with_min_token_index(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizer,
) -> None:
    inputs = tokenizer(
        "What is cheesier than cheese? Nothing is cheesier than cheese",
        return_tensors="pt",
    )
    original_hidden_states = model(**inputs, output_hidden_states=True).hidden_states
    patch = torch.randn(768)
    steering_vector = SteeringVector(
        layer_activations={1: patch},
        layer_type="decoder_block",
    )
    steering_vector.patch_activations(model, min_token_index=5)
    patched_hidden_states = model(**inputs, output_hidden_states=True).hidden_states

    # The first 5 tokens should not be patched, due to min_token_index
    assert torch.equal(
        original_hidden_states[2][0, :5], patched_hidden_states[2][0, :5]
    )
    assert not torch.equal(
        original_hidden_states[2][0, 5:], patched_hidden_states[2][0, 5:]
    )

    expected_hidden_state = original_hidden_states[2][0, 5:] + patch
    assert torch.equal(expected_hidden_state, patched_hidden_states[2][0, 5:])


@torch.no_grad()
def test_SteeringVector_handle_reverts_model_changes(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizer,
) -> None:
    inputs = tokenizer("Hello, world", return_tensors="pt")
    original_logits = model(**inputs, output_hidden_states=False).logits
    steering_vector = SteeringVector(
        layer_activations={
            1: torch.randn(768),
            -1: torch.randn(768),
        },
        layer_type="decoder_block",
    )
    handle = steering_vector.patch_activations(model)
    patched_logits = model(**inputs, output_hidden_states=False).logits
    handle.remove()
    unpatched_logits = model(**inputs, output_hidden_states=False).logits

    assert not torch.equal(original_logits, patched_logits)
    assert torch.equal(original_logits, unpatched_logits)
