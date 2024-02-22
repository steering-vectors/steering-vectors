import pytest
from transformers import (
    AutoTokenizer,
    GemmaConfig,
    GemmaForCausalLM,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
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


@pytest.fixture
def empty_gemma_model() -> GemmaForCausalLM:
    config = GemmaConfig(
        num_hidden_layers=3,
        hidden_size=1024,
        intermediate_size=2752,
    )
    model = GemmaForCausalLM(config)
    return model.eval()


@pytest.fixture
def empty_mistral_model() -> MistralForCausalLM:
    config = MistralConfig(
        num_hidden_layers=3,
        hidden_size=1024,
        intermediate_size=2752,
    )
    model = MistralForCausalLM(config)
    return model.eval()
