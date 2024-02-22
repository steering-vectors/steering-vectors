# Steering Vectors

[![ci](https://img.shields.io/github/actions/workflow/status/steering-vectors/steering-vectors/ci.yaml?branch=main)](https://github.com/steering-vectors/steering-vectors)
[![PyPI](https://img.shields.io/pypi/v/steering-vectors?color=blue)](https://pypi.org/project/steering-vectors/)

Steering vectors / representation engineering for transformer language models in Pytorch / Huggingface

Full docs: https://steering-vectors.github.io/steering-vectors

## About

This library provides utilies for training and applying steering vectors to language models (LMs) from [Huggingface](https://huggingface.co/), like GPT, LLaMa, Gemma, Mistral, Pythia, and many more!

For more info on steering vectors and representation engineering, check out the following work:

- [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) Rimsky et al., 2023
- [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405) Zou et al., 2023

## Installation

```
pip install steering-vectors
```

Check out the [full documentation](https://steering-vectors.github.io/steering-vectors/) for more usage info.

## Contributing

Any contributions to improve this project are welcome! Please open an issue or pull request in this repo with any bugfixes / changes / improvements you have.

This project uses [Black](https://github.com/psf/black) for code formatting, [Flake8](https://flake8.pycqa.org/en/latest/) for linting, and [Pytest](https://docs.pytest.org/) for tests. Make sure any changes you submit pass these code checks in your PR. If you have trouble getting these to run feel free to open a pull-request regardless and we can discuss further in the PR.

## License

This code is released under a MIT license.
