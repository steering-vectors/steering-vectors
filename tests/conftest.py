import pytest
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    GPTNeoXForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizer,
)


@pytest.fixture
def model() -> GPTNeoXForCausalLM:
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m",
        token=True,
    )
    return model.eval()


@pytest.fixture
def tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m",
        model_max_length=128,
    )


@pytest.fixture
def gpt2_model() -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return model.eval()


@pytest.fixture
def gpt2_tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def empty_llama_model() -> LlamaForCausalLM:
    config = LlamaConfig(
        num_hidden_layers=3,
        hidden_size=1024,
        intermediate_size=2752,
    )
    return LlamaForCausalLM(config)
