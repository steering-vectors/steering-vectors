import torch
from transformers import GPT2LMHeadModel, LlamaForCausalLM, PreTrainedTokenizer

from steering_vectors.train_steering_vector import train_steering_vector
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
