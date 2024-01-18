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
This library provides utilies for training and applying steering vectors to language models (LMs) from `Huggingface <https://huggingface.co/>`_, like GPT2, Llama2, GptNeoX, etc...

Steering vectors try to identify a direction in hidden activations which can be used to control how the model behaves. For example, we can make a LM be more or less honest in its responses, or more or less sycophantic. This works by providing paired positive and negative training examples for the characteristic you're trying to elicit. To train a steering vector for truthfulness, you might use prompts like the following:

Positive prompt (truthful):

.. code-block::

   Question: What is the correct answer? 2 + 2 =
   (A): 4
   (B): 7
   Answer: A

Negative prompt (not truthful):

.. code-block::

   Question: What is the correct answer? 2 + 2 =
   (A): 4
   (B): 7
   Answer: B

Then, we can find a steering vector by observing the hidden activations in a language models as it processing the positive and negative statements above and subtract the activation from the "negative" sample from the activation for the "positive" example. Then, we can use this vector to "steer" the model to be more or less truthful. Neat!

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
   :maxdepth: 2
   :caption: API Reference:

   api/steering_vector
   api/train_steering_vector
   api/record_activations

.. toctree::
   :caption: Project Links

   GitHub <https://github.com/steering-vectors/steering-vectors>
   PyPI <https://pypi.org/project/steering-vectors>

.. _PyPI: https://pypi.org/project/steering-vectors/
