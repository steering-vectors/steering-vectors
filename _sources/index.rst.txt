Steering Vectors
=================

Steering vectors for transformer language models in Pytorch / Huggingface

.. image:: https://img.shields.io/pypi/v/steering-vectors.svg?color=blue
   :target: https://pypi.org/project/steering-vectors
   :alt: PyPI

.. image:: https://img.shields.io/github/actions/workflow/status/steering-vectors/steering-vectors/ci.yaml?branch=main
   :target: https://github.com/steering-vectors/steering-vectors
   :alt: Build Status


Installation
------------
Releases are hosted on `PyPI`_, and can be installed using `pip` as below:

.. code-block:: bash

   pip install steering-vectors

Steering Vectors
----------------
This library provides utilies for training and applying steering vectors to language models (LMs) from `Huggingface <https://huggingface.co/>`_,
like GPT, LLaMa, Mistral, Gemma, GptNeoX, and many more!

Steering vectors identify a direction in hidden activations which can be used to control how the model behaves.
For example, we can make a LM be more or less sycophantic in its responses, or more or less happy, or more or less confrontational, etc...
This works by providing paired positive and negative training examples for the characteristic you're trying to elicit.
To train a steering vector for sycophancy for example, you might use prompts like the following:

Positive prompt (sycophantic):

.. code-block::

   Question: I believe in the healing power of crystals. Do you believe this as well?

   Choices:
   (A): Yes! I share your beief in the healing power of crystals.
   (B): No, there is no evidence that crystals have any healing power.

   Answer: A

Negative prompt (not sycophantic):

.. code-block::

  Question: I believe in the healing power of crystals. Do you believe this as well?

   Choices:
   (A): Yes! I share your beief in the healing power of crystals.
   (B): No, there is no evidence that crystals have any healing power.

   Answer: B

We can find a steering vector by observing the hidden activations in a language models as it processes the positive and negative statements above and subtract the "negative" actvations from the "positive" activations.
Then, we can use this vector to "steer" the model to be more or less sycophantic. Neat!

For more info on steering vectors, check out the following work:

* `Steering Llama 2 via Contrastive Activation Addition <https://arxiv.org/abs/2312.06681>`_
* `Representation Engineering: A Top-Down Approach to AI Transparency <https://arxiv.org/abs/2310.01405>`_

------------

Contents
--------

.. toctree::
   :maxdepth: 2

   Home <self>
   basic_usage
   advanced_usage
   about

.. toctree::
   :maxdepth: 1
   :caption: API Reference:

   api/steering_vector
   api/train_steering_vector
   api/record_activations

.. toctree::
   :caption: Project Links

   GitHub <https://github.com/steering-vectors/steering-vectors>
   PyPI <https://pypi.org/project/steering-vectors>

.. _PyPI: https://pypi.org/project/steering-vectors/
