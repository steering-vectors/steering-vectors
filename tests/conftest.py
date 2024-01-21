import pytest
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizer,
)


@pytest.fixture
def model() -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return model.eval()


@pytest.fixture
def tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def empty_llama_model() -> LlamaForCausalLM:
    config = LlamaConfig(
        num_hidden_layers=3,
        hidden_size=1024,
        intermediate_size=2752,
    )
    model = LlamaForCausalLM(config)
    return model.eval()


@pytest.fixture
def llama_tokenizer() -> PreTrainedTokenizer:
    # using vicuna tokenizer since llama requires logging in to hugginface, which is annoying for CI / tests
    return AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
