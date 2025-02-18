About
=====

Acknowledgements
----------------

This library is inspired by the following excellent papers:

* `Steering Llama 2 via Contrastive Activation Addition <https://arxiv.org/abs/2312.06681>`_
* `Representation Engineering: A Top-Down Approach to AI Transparency <https://arxiv.org/abs/2310.01405>`_

How this works
--------------
To apply steering vectors to the model, we must have some way of modifying the model's forward pass.
The approach taken in the official codebases for Representation Engineering and Contrastive Activation
Addition do this by modifying the underlying model and replacing decoder blocks with custom wrappers.
While this is conceptually simple, it has some major drawbacks. This won't work for arbitrary models, as a new
wrapper needs to be built for every model and every type of layer. This also changes the model's architecture,
which can lead to unexpected behavior if the end-user isn't aware of the changes.

The steering_vectors library uses Pytorch hooks instead of custom layer wrappers to modify the underlying model's forward pass.
You can read more about hooks at the `official PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html>`_.
This allows us to modify the forward pass of any model without changing the model's architecture, and makes it possible
to apply steering to arbitrary models from Huggingface without any custom wrapping code.


Contributing
------------

Any contributions to improve this project are welcome! Please open an issue or pull request in the `Github repo <https://github.com/steering-vectors/steering-vectors>`_ with bugfixes, changes, and improvements.

License
-------

Steering Vectors is released under a MIT license.
