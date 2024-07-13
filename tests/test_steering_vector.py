import pytest
import torch
from transformers import (
    GemmaForCausalLM,
    GPT2LMHeadModel,
    LlamaForCausalLM,
    MistralForCausalLM,
    PreTrainedTokenizer,
)

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
def test_SteeringVector_apply_gemma(
    empty_gemma_model: GemmaForCausalLM,
    tokenizer: PreTrainedTokenizer,
) -> None:
    model = empty_gemma_model
    inputs = tokenizer("Hello, world", return_tensors="pt")
    original_hidden_states = model(**inputs, output_hidden_states=True).hidden_states
    patch = torch.randn(1024)
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
def test_SteeringVector_apply_mistral(
    empty_mistral_model: MistralForCausalLM,
    tokenizer: PreTrainedTokenizer,
) -> None:
    model = empty_mistral_model
    inputs = tokenizer("Hello, world", return_tensors="pt")
    original_hidden_states = model(**inputs, output_hidden_states=True).hidden_states
    patch = torch.randn(1024)
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


@pytest.mark.parametrize(
    "target_token_indices, non_target_token_indices, verification_token_indices",
    [
        ([2, 4, 7], [i for i in range(13) if i not in [2, 4, 7]], [2, 4, 7]),
        (
            torch.tensor([1 if i in [2, 4, 7] else 0 for i in range(13)]),
            [i for i in range(13) if i not in [2, 4, 7]],
            [2, 4, 7],
        ),
        (
            slice(1, 13, 2),
            [i for i in range(13) if i not in list(range(1, 13, 2))],
            list(range(1, 13, 2)),
        ),
    ],
)
@torch.no_grad()
def test_SteeringVector_patch_activations_with_token_indices(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizer,
    target_token_indices: list[int] | torch.Tensor | slice,
    non_target_token_indices: list[int],
    verification_token_indices: list[int],
) -> None:
    """verify that patch_activations works both when target indices is a list of indices or a mask
    target_token_indices: a list, slice, or tensor that is passed to patch_activations to select indices to patch
    non_target_token_indices: a list of token indices that should not be patched, used to test that they haven't changed after patching
    verification_token_indices: a list of token indices that should be patched, used to test that they have changed after patching. We can't use target_token_indices for this as it might be a mask.
    """
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
    steering_vector.patch_activations(model, token_indices=target_token_indices)
    patched_hidden_states = model(**inputs, output_hidden_states=True).hidden_states

    # only the target tokens (with indices 2, 4, and 7) should be patched
    assert torch.equal(
        original_hidden_states[2][0, non_target_token_indices],
        patched_hidden_states[2][0, non_target_token_indices],
    )
    assert not torch.equal(
        original_hidden_states[2][0, verification_token_indices],
        patched_hidden_states[2][0, verification_token_indices],
    )

    expected_hidden_state = (
        original_hidden_states[2][0, verification_token_indices] + patch
    )
    assert torch.equal(
        expected_hidden_state, patched_hidden_states[2][0, verification_token_indices]
    )


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


def test_SteeringVector_to_dtype() -> None:
    vec = SteeringVector(
        layer_activations={
            1: torch.randn(768),
            -1: torch.randn(768),
        },
        layer_type="decoder_block",
    )
    vec2 = vec.to(torch.float16)
    assert vec2.layer_activations[1].dtype == torch.float16
    assert vec2.layer_activations[-1].dtype == torch.float16
    assert vec2.layer_type == vec.layer_type
    assert vec.layer_activations[1].dtype == torch.float32
    assert vec.layer_activations[-1].dtype == torch.float32


@torch.no_grad()
def test_SteeringVector_mask_batched_activations(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizer,
) -> None:
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    prompts = [
        "Hello world!",
        "This is a prompt.",
        "This is also a prompt.",
        "This is the last pompt.",
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    original_hidden_states = model(**inputs, output_hidden_states=True).hidden_states
    patch = torch.randn(768)
    steering_vector = SteeringVector(
        layer_activations={1: patch},
        layer_type="decoder_block",
    )

    # mask the last token in each prompt of the batch
    last_token_idxs = [2, 4, 5, 7]
    token_mask = torch.zeros_like(inputs["input_ids"])
    token_mask[range(4), last_token_idxs] = 1

    steering_vector.patch_activations(model, token_indices=token_mask)
    patched_hidden_states = model(**inputs, output_hidden_states=True).hidden_states

    # The first hidden state is the input embeddings, which are not patched
    assert torch.equal(original_hidden_states[0], patched_hidden_states[0])
    # next is the first decoder block, which is not patched
    assert torch.equal(original_hidden_states[1], patched_hidden_states[1])

    # At layer 1, only the activations of the last token in each prompt should be patched

    # Create boolean mask for all tokens except the last token in each prompt
    non_last_token_mask = torch.ones_like(token_mask).bool()
    non_last_token_mask[range(4), last_token_idxs] = False

    # Verify non-last token activations are unchanged
    assert torch.equal(
        original_hidden_states[2][non_last_token_mask],
        patched_hidden_states[2][non_last_token_mask],
    )

    # Verify the last token activations are patched correctly
    expected_hidden_states = (
        original_hidden_states[2][range(4), last_token_idxs] + patch
    )
    assert torch.equal(
        expected_hidden_states, patched_hidden_states[2][range(4), last_token_idxs]
    )
