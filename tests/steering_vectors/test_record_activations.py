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
