import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizer

from steering_vectors.record_activations import record_activations


def test_record_activations_matches_decoder_hidden_states(
    model: GPT2LMHeadModel, tokenizer: PreTrainedTokenizer
) -> None:
    with record_activations(model) as recorded_activations:
        inputs = tokenizer("Hello world", return_tensors="pt")
        hidden_states = model(**inputs, output_hidden_states=True).hidden_states
    # the first hidden state is the input embeddings, which we don't record
    # the last hidden state (at least for neox) has the final layer norm applied, so skip that
    assert len(recorded_activations) == len(hidden_states) - 1
    for recorded_activation, hidden_state in zip(
        list(recorded_activations.values())[:-1], hidden_states[1:-1]
    ):
        assert torch.equal(recorded_activation[0], hidden_state)


def test_record_activations_positive_vs_negative_indices(
    model: GPT2LMHeadModel, tokenizer: PreTrainedTokenizer
) -> None:
    positive_layer_nums = [11, 9]  # positive layer indices
    negative_layer_nums = [-1, -3]  # corresponding negative layer indices

    with record_activations(
        model, layer_nums=positive_layer_nums
    ) as recorded_activations_positive:
        inputs = tokenizer("Hello world", return_tensors="pt")
        model(**inputs)

    with record_activations(
        model, layer_nums=negative_layer_nums
    ) as recorded_activations_negative:
        inputs = tokenizer("Hello world", return_tensors="pt")
        model(**inputs)

    assert list(recorded_activations_positive.keys()) == list(
        recorded_activations_negative.keys()
    )

    for key in recorded_activations_positive.keys():
        assert torch.equal(
            recorded_activations_positive[key][0], recorded_activations_negative[key][0]
        )
