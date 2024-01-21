# Steering Vectors

[![ci](https://img.shields.io/github/actions/workflow/status/steering-vectors/steering-vectors/ci.yaml?branch=main)](https://github.com/steering-vectors/steering-vectors)
[![PyPI](https://img.shields.io/pypi/v/steering-vectors?color=blue)](https://pypi.org/project/steering-vectors/)

Steering vectors for transformer language models in Pytorch / Huggingface

Full docs: https://steering-vectors.github.io/steering-vectors

## About

This library provides utilies for training and applying steering vectors to language models (LMs) from [Huggingface](https://huggingface.co/), like GPT2, Llama2, GptNeoX, etc...

Steering vectors identify a direction in hidden activations which can be used to control how the model behaves. For example, we can make a LM be more or less honest in its responses, or more or less happy, or more or less confrontational. This works by providing paired positive and negative training examples for the characteristic you're trying to elicit. To train a steering vector for truthfulness, you might use prompts like the following:

Positive prompt (truthful):

```
Question: What is the correct answer? 2 + 2 =
(A): 4
(B): 7
Answer: A
```

Negative prompt (not truthful):

```
Question: What is the correct answer? 2 + 2 =
(A): 4
(B): 7
Answer: B
```

Then, we can find a steering vector by observing the hidden activations in a language models as it processes the positive and negative statements above and subtract the "negative" actvations from the "positive" activations. Then, we can use this vector to "steer" the model to be more or less truthful. Neat!

For more info on steering vectors, check out the following work:

- [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) Rimsky et al., 2023
- [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405) Zou et al., 2023

## Installation

```
pip install steering-vectors
```

### Basic usage

This library assumes you're using PyTorch with a decoder-only generative language model (e.g. GPT, LLaMa, etc...), and a tokenizer from Huggingface.

To begin, collect tuples of positive and negative training prompts in a list, and run `train_steering_vector()`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import train_steering_vector

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# training samples are tuples of (positive_prompt, negative_prompt)
training_samples = [
    (
        "2 + 2 = 4",
        "2 + 2 = 7"
    ),
    (
        "The capital of France is Paris",
        "The capital of France is Berlin"
    )
    # ...
]


steering_vector = train_steering_vector(
    model,
    tokenizer,
    training_samples,
    show_progress=True,
)
```

Then, you can use the steering vector to "steer" the model's behavior:

```python
with steering_vector.apply(model):
    prompt = "What is the correct answer? 2 + 2 ="
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model(**inputs)

```

Check out the [full documentation](https://steering-vectors.github.io/steering-vectors/) for more info.

## Contributing

Any contributions to improve this project are welcome! Please open an issue or pull request in this repo with any bugfixes / changes / improvements you have!

This project uses [Black](https://github.com/psf/black) for code formatting, [Flake8](https://flake8.pycqa.org/en/latest/) for linting, and [Pytest](https://docs.pytest.org/) for tests. Make sure any changes you submit pass these code checks in your PR. If you have trouble getting these to run feel free to open a pull-request regardless and we can discuss further in the PR.

## License

This code is released under a MIT license.
