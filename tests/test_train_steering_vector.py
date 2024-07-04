from collections.abc import Callable

import pytest
import torch
from transformers import GPT2LMHeadModel, LlamaForCausalLM, PreTrainedTokenizer

from steering_vectors.train_steering_vector import (
    SteeringVectorTrainingSample,
    _get_token_index,
    train_steering_vector,
)
from tests._original_caa.llama_wrapper import LlamaWrapper  # type: ignore


def test_train_steering_vector_reads_from_final_token_by_default(
    model: GPT2LMHeadModel, tokenizer: PreTrainedTokenizer
) -> None:
    pos_train_sample = "[INST] 2 + 2 = ? A: 2, B: 4 [/INST] The answer is B"
    neg_train_sample = "[INST] 2 + 2 = ? A: 2, B: 4 [/INST] The answer is A"
    training_data = [(pos_train_sample, neg_train_sample)]

    pos_inputs = tokenizer(pos_train_sample, return_tensors="pt")
    neg_inputs = tokenizer(neg_train_sample, return_tensors="pt")
    pos_train_outputs = model(**pos_inputs, output_hidden_states=True)
    neg_train_outputs = model(**neg_inputs, output_hidden_states=True)

    steering_vector = train_steering_vector(
        model, tokenizer, training_data, layers=[2, 3, 4]
    )

    assert sorted(list(steering_vector.layer_activations.keys())) == [2, 3, 4]
    for layer, vector in steering_vector.layer_activations.items():
        pos_vector = pos_train_outputs.hidden_states[layer + 1][0, -1, :]
        neg_vector = neg_train_outputs.hidden_states[layer + 1][0, -1, :]
        assert torch.allclose(vector, pos_vector - neg_vector)


def test_train_steering_vector_works_with_multiple_token_indices_by_passing_indices(
    model: GPT2LMHeadModel, tokenizer: PreTrainedTokenizer
) -> None:
    def get_x_index(prompt: str) -> int:
        tokenization = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))
        return tokenization.index("ĠX")

    training_data: list[SteeringVectorTrainingSample] = [
        SteeringVectorTrainingSample(
            "This is a short positive example. X <- probe here.",
            "This is a short negative example with different token length. X <- probe here.",
            get_x_index("This is a short positive example. X <- probe here."),
            get_x_index(
                "This is a short negative example with different token length. X <- probe here."
            ),
        ),
        SteeringVectorTrainingSample(
            "Dummy text. This is a much longer positive example. X <- probe here. More dummy text.",
            "Dummy text. This is a much longer negative example with different token length. X <- probe here. More dummy text.",
            get_x_index(
                "Dummy text. This is a much longer positive example. X <- probe here. More dummy text."
            ),
            get_x_index(
                "Dummy text. This is a much longer negative example with different token length. X <- probe here. More dummy text."
            ),
        ),
    ]
    pos_examples = [p.positive_str for p in training_data]
    neg_examples = [p.negative_str for p in training_data]

    x_indices = [p.read_positive_token_index for p in training_data] + [
        p.read_negative_token_index for p in training_data
    ]
    pos_acts = []
    neg_acts = []

    for pos_example in pos_examples:
        pos_inputs = tokenizer(pos_example, return_tensors="pt")
        pos_outputs = model(**pos_inputs, output_hidden_states=True)
        pos_acts.append(pos_outputs.hidden_states)

    for neg_example in neg_examples:
        neg_inputs = tokenizer(neg_example, return_tensors="pt")
        neg_outputs = model(**neg_inputs, output_hidden_states=True)
        neg_acts.append(neg_outputs.hidden_states)

    steering_vector = train_steering_vector(
        model, tokenizer, training_data, layers=[2, 3, 4]
    )

    for layer, vector in steering_vector.layer_activations.items():
        diffs = []
        for pos_act, neg_act, pos_token, neg_token in zip(
            pos_acts, neg_acts, x_indices[:2], x_indices[2:]
        ):
            pos_act = pos_act[layer + 1][0, pos_token, :]
            neg_act = neg_act[layer + 1][0, neg_token, :]
            diff = pos_act - neg_act
            diffs.append(diff)
        mean_diff = torch.stack(diffs).mean(dim=0)
        assert torch.allclose(vector, mean_diff)


def test_train_steering_vector_works_with_multiple_token_indices_by_passing_callable(
    model: GPT2LMHeadModel, tokenizer: PreTrainedTokenizer
) -> None:
    def get_x_index(prompt: str) -> int:
        tokenization = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))
        return tokenization.index("ĠX")

    training_data = [
        (
            "This is a short positive example. X <- probe here.",
            "This is a short negative example with different token length. X <- probe here.",
        ),
        (
            "Dummy text. This is a much longer positive example. X <- probe here. More dummy text.",
            "Dummy text. This is a much longer negative example with different token length. X <- probe here. More dummy text.",
        ),
    ]
    pos_examples = [p[0] for p in training_data]
    neg_examples = [p[1] for p in training_data]

    x_indices = [get_x_index(p[0]) for p in training_data] + [
        get_x_index(p[1]) for p in training_data
    ]
    pos_acts = []
    neg_acts = []

    for pos_example in pos_examples:
        pos_inputs = tokenizer(pos_example, return_tensors="pt")
        pos_outputs = model(**pos_inputs, output_hidden_states=True)
        pos_acts.append(pos_outputs.hidden_states)

    for neg_example in neg_examples:
        neg_inputs = tokenizer(neg_example, return_tensors="pt")
        neg_outputs = model(**neg_inputs, output_hidden_states=True)
        neg_acts.append(neg_outputs.hidden_states)

    steering_vector = train_steering_vector(
        model, tokenizer, training_data, layers=[2, 3, 4], read_token_index=get_x_index
    )

    for layer, vector in steering_vector.layer_activations.items():
        diffs = []
        for pos_act, neg_act, pos_token, neg_token in zip(
            pos_acts, neg_acts, x_indices[:2], x_indices[2:]
        ):
            pos_act = pos_act[layer + 1][0, pos_token, :]
            neg_act = neg_act[layer + 1][0, neg_token, :]
            diff = pos_act - neg_act
            diffs.append(diff)
        mean_diff = torch.stack(diffs).mean(dim=0)
        assert torch.allclose(vector, mean_diff)


def test_train_steering_vector_custom_aggregator(
    model: GPT2LMHeadModel, tokenizer: PreTrainedTokenizer
) -> None:
    pos_train_sample = "[INST] 2 + 2 = ? A: 2, B: 4 [/INST] The answer is B"
    neg_train_sample = "[INST] 2 + 2 = ? A: 2, B: 4 [/INST] The answer is A"
    training_data = [(pos_train_sample, neg_train_sample)]

    pos_inputs = tokenizer(pos_train_sample, return_tensors="pt")
    neg_inputs = tokenizer(neg_train_sample, return_tensors="pt")
    pos_train_outputs = model(**pos_inputs, output_hidden_states=True)
    neg_train_outputs = model(**neg_inputs, output_hidden_states=True)

    steering_vector = train_steering_vector(
        model,
        tokenizer,
        training_data,
        layers=[2, 3, 4],
        # custom aggregator adds 1 to the mean difference between the positive and negative
        aggregator=lambda pos, neg: (pos - neg).mean(dim=0) + 1,
    )

    assert sorted(list(steering_vector.layer_activations.keys())) == [2, 3, 4]
    for layer, vector in steering_vector.layer_activations.items():
        pos_vector = pos_train_outputs.hidden_states[layer + 1][0, -1, :]
        neg_vector = neg_train_outputs.hidden_states[layer + 1][0, -1, :]

        assert torch.allclose(vector, (pos_vector - neg_vector) + 1)


def test_train_steering_vector_batching_gives_identical_result_to_unbatched(
    model: GPT2LMHeadModel, tokenizer: PreTrainedTokenizer
) -> None:
    training_data = [
        (
            "This is a short positive example.",
            "This is a short negative example with different token length.",
        ),
        (
            "Dummy text. This is a much longer positive example.",
            "Dummy text. This is a much longer negative example with different token length.",
        ),
    ]

    steering_vector_batch_1 = train_steering_vector(
        model, tokenizer, training_data, layers=[2, 3, 4], batch_size=1
    )
    steering_vector_batch_2 = train_steering_vector(
        model, tokenizer, training_data, layers=[2, 3, 4], batch_size=2
    )
    for layer in steering_vector_batch_1.layer_activations.keys():
        assert torch.allclose(
            steering_vector_batch_1.layer_activations[layer],
            steering_vector_batch_2.layer_activations[layer],
            atol=1e-5,
        )


def test_train_steering_vector_matches_original_caa(
    empty_llama_model: LlamaForCausalLM, llama_tokenizer: PreTrainedTokenizer
) -> None:
    model = empty_llama_model
    tokenizer = llama_tokenizer

    layers = [0, 1, 2]

    training_data = [
        (
            "[INST] 2 + 2 = ? A: 2, B: 4 [/INST] The answer is B",
            "[INST] 2 + 2 = ? A: 2, B: 4 [/INST] The answer is A",
        ),
        (
            "[INST] 3 + 2 = ? A: 5, B: 7 [/INST] The answer is A",
            "[INST] 3 + 2 = ? A: 5, B: 7 [/INST] The answer is B",
        ),
    ]

    steering_vector = train_steering_vector(
        model, tokenizer, training_data, layers=layers, read_token_index=-2
    )

    # hackily translated from generate_vectors.py script
    tokenized_data = [
        (tokenizer.encode(pos), tokenizer.encode(neg)) for pos, neg in training_data
    ]
    pos_activations: dict[int, list[torch.Tensor]] = dict(
        [(layer, []) for layer in layers]
    )
    neg_activations: dict[int, list[torch.Tensor]] = dict(
        [(layer, []) for layer in layers]
    )
    wrapped_model = LlamaWrapper(model, tokenizer)

    for p_tokens, n_tokens in tokenized_data:
        p_tokens = torch.tensor(p_tokens).unsqueeze(0).to(model.device)
        n_tokens = torch.tensor(n_tokens).unsqueeze(0).to(model.device)
        wrapped_model.reset_all()
        wrapped_model.get_logits(p_tokens)
        for layer in layers:
            p_activations = wrapped_model.get_last_activations(layer)
            p_activations = p_activations[0, -2, :].detach().cpu()
            pos_activations[layer].append(p_activations)
        wrapped_model.reset_all()
        wrapped_model.get_logits(n_tokens)
        for layer in layers:
            n_activations = wrapped_model.get_last_activations(layer)
            n_activations = n_activations[0, -2, :].detach().cpu()
            neg_activations[layer].append(n_activations)

    caa_vecs_by_layer = {}
    for layer in layers:
        all_pos_layer = torch.stack(pos_activations[layer])
        all_neg_layer = torch.stack(neg_activations[layer])
        caa_vecs_by_layer[layer] = (all_pos_layer - all_neg_layer).mean(dim=0)

    for layer in layers:
        assert torch.allclose(
            steering_vector.layer_activations[layer], caa_vecs_by_layer[layer]
        )


@pytest.mark.parametrize(
    "custom_idx, default_idx, prompt, expected_output",
    [
        (None, 0, "prompt1", 0),
        (1, 0, "prompt2", 1),
        (None, lambda x: len(x), "prompt3", 7),
        (2, lambda x: len(x), "prompt4", 2),
    ],
)
def test_get_token_index(
    custom_idx: int | None,
    default_idx: int | Callable[[str], int],
    prompt: str,
    expected_output: int,
) -> None:
    actual_output = _get_token_index(custom_idx, default_idx, prompt)
    assert actual_output == expected_output
