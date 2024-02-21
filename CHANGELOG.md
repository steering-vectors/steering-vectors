# CHANGELOG



## v0.9.0 (2024-02-21)

### Chore

* chore: replace pre-3.10 types with modern typing, and adding linting (#28)

* chore: replace pre-3.10 types with modern typing, and adding linting

* adding isort CI check

* fixing linting ([`729ea82`](https://github.com/steering-vectors/steering-vectors/commit/729ea82cb2c48bbbc7851d5601d19c242beee498))

### Feature

* feat: add batch training support (#30)

* feat: add batch training support

* ensure indexing tensor is on same device as model ([`625d826`](https://github.com/steering-vectors/steering-vectors/commit/625d8267ce74888ff19c92ae7507a1decceb5927))

* feat: renaming PatchOperator to PatchDeltaOperator (#27) ([`f326823`](https://github.com/steering-vectors/steering-vectors/commit/f3268231dab8dfa6d7eae099871312be532c9818))


## v0.8.0 (2024-02-21)

### Chore

* chore: adding info about pytorch hooks to docs site (#29) ([`5410ebe`](https://github.com/steering-vectors/steering-vectors/commit/5410ebeb1610f48c14bc98948d822577ed0a819b))

### Feature

* feat: renaming prompts to strs in SteeringVectorTrainingSample (#32) ([`16118a9`](https://github.com/steering-vectors/steering-vectors/commit/16118a9f704c07849e1ddffff985972d14a4d268))


## v0.7.0 (2024-02-10)

### Feature

* feat: Add aggregators for logistic and linear regression (#22)

* add aggregators for logistic and linear regression

* apply pairwise mean-centering before regression and improve regression unit-tests

* proposing CR changes to add-regression-aggregator (#26)

* proposing CR changes to add-regression-aggregator

* adding a test for passing sklearn_kwargs

* tweaking tensor.to call

---------

Co-authored-by: David Chanin &lt;chanindav@gmail.com&gt; ([`a346c6f`](https://github.com/steering-vectors/steering-vectors/commit/a346c6f15c7934463b86ccce54832ad578cc791b))


## v0.6.0 (2024-02-10)

### Feature

* feat: Support selecting specific token indices when applying and training steering vector (#21)

* support steering at specific indices using integer list of indices

* support steering at specific indices using mask

* support passing training indices as part of trianing sample or via callable

* use list instead of List and update docstrings

* parametrize patch activations test to verify that slices and masks work to select indices

* remove handling of impossible case where token_indices is none in _create_additive_hook

* formatting and typing fixing

* make SteeringVectorTrainingSample a DataClass

* extract _get_token_index as a top level function ([`9f2a0c5`](https://github.com/steering-vectors/steering-vectors/commit/9f2a0c5af67556625a70509752b7152ec50ffe1f))

### Unknown

* Add example of how to extract and apply CAA-style steering vectors (#18)

* Add example dependencies; fix missing torch bug

* Fix data type bug in train_steering_vector

* [WIP] Add CAA example notebook

* Add CAA example

* minor

* fix: update version

I forgot to bump the version previously so here it is.

* Revert &#34;Fix data type bug in train_steering_vector&#34;

This reverts commit 8c80db5e2c2fd3aae60086b4cf652e276ce032ae.

* Remove PDM

* Add example dependencies in example group

* Delete raw data from examples

* Remove example dependencies from pyproject.toml

* Fix nits in notebook

* Fix more nits

* Restructure examples dir

* Fix minor bugs

* Fix other nits

---------

Co-authored-by: dtch1997 &lt;dtch1997@users.noreply.github.com&gt; ([`94443cd`](https://github.com/steering-vectors/steering-vectors/commit/94443cd9d247621c005533fb21d4370182644d67))


## v0.5.0 (2024-01-25)

### Feature

* feat: adding a PCA aggregator (#9) ([`77c1b7b`](https://github.com/steering-vectors/steering-vectors/commit/77c1b7b74acb40545d4c2b8a3559a079fe319602))


## v0.4.0 (2024-01-25)

### Chore

* chore: add colab demo to README.md ([`52b5f73`](https://github.com/steering-vectors/steering-vectors/commit/52b5f73ef934f38bad2ed9a497729adc66085d6c))

* chore: update basic_usage.rst example to use generate() ([`8d9edee`](https://github.com/steering-vectors/steering-vectors/commit/8d9edeeecc56917455367c0033d66d4841f8ab97))

* chore: update example in README.md ([`63f8d05`](https://github.com/steering-vectors/steering-vectors/commit/63f8d053d4b8995cfbd54203452c4c9f5efe45d5))

* chore: adding a test to assert that steering works identically to CAA ([`3cbbeb3`](https://github.com/steering-vectors/steering-vectors/commit/3cbbeb38ea01dc6bd245adaa75c5360085136c2a))

### Feature

* feat: adding helper to move SteeringVector to a given device or dtype ([`e28cce3`](https://github.com/steering-vectors/steering-vectors/commit/e28cce39a7cddf1c6c68cf843b0933b972ebd059))

### Unknown

* Merge pull request #11 from steering-vectors/vec-to-device

feat: adding helper to move SteeringVector to a given device or dtype ([`65f7a8b`](https://github.com/steering-vectors/steering-vectors/commit/65f7a8b999693d26d6b222d0a60bdeddc367f296))


## v0.3.0 (2024-01-23)

### Feature

* feat: adding support for custom aggregators during training (#7) ([`220d4f7`](https://github.com/steering-vectors/steering-vectors/commit/220d4f77aa13fdcade1e9a9d09dd354c2ae9d699))


## v0.2.0 (2024-01-21)

### Chore

* chore: adding some basic training tests (#3) ([`d1f3311`](https://github.com/steering-vectors/steering-vectors/commit/d1f33110b7882c89ea2719f0befb966abc4dc489))

* chore: simplifying test structure ([`5fc7e50`](https://github.com/steering-vectors/steering-vectors/commit/5fc7e505708a5e5e13d0349e16f7aa876ac0511c))

* chore: fix code minor highlighting typo in docs ([`9d2659b`](https://github.com/steering-vectors/steering-vectors/commit/9d2659b39b8fe456a52c2426eb70bcedfd8f0424))

* chore: updating docs index to match README ([`1c14f80`](https://github.com/steering-vectors/steering-vectors/commit/1c14f80ccc3a2fcfb6b2da4ca80469aa0c18a621))

* chore: fix typo in docs site ([`5f6186b`](https://github.com/steering-vectors/steering-vectors/commit/5f6186b8e416da64ee5cf30e2d86c5948da8078e))

### Feature

* feat: adding option to show_progress during training (#6) ([`48cbbf1`](https://github.com/steering-vectors/steering-vectors/commit/48cbbf1212c1ceb68e318b2c3af9165b01fda1c2))


## v0.1.0 (2024-01-19)

### Chore

* chore: updating docs ([`d69e4ef`](https://github.com/steering-vectors/steering-vectors/commit/d69e4efb615ddd39367371efa5bf52750a5b5745))

* chore: disable docs build in CI until we add a docs site ([`dbb1bc3`](https://github.com/steering-vectors/steering-vectors/commit/dbb1bc3a26d59cfb93c86f09f65d2e767404ffcf))

* chore: removing Pythia tests since they require huggingface token ([`4dd2038`](https://github.com/steering-vectors/steering-vectors/commit/4dd2038437a7d46b2874e39dfd0befbc7ce000b0))

### Feature

* feat: initial release ([`8ee0de8`](https://github.com/steering-vectors/steering-vectors/commit/8ee0de83c7bef878a1537f0ba847249966790596))

### Unknown

* adding docs website ([`e8cfb2a`](https://github.com/steering-vectors/steering-vectors/commit/e8cfb2a9d772b153ad52a69311e29dd6f8f66407))

* adding basic README content ([`459eb5c`](https://github.com/steering-vectors/steering-vectors/commit/459eb5c6680536e00f2076f23ec618ae02635d14))

* initial commit ([`20ec6d6`](https://github.com/steering-vectors/steering-vectors/commit/20ec6d63c3b78fd034692426fce1965f5930d1ea))

* Initial commit ([`b02fd3e`](https://github.com/steering-vectors/steering-vectors/commit/b02fd3e24b7c1272ff6d0240ae50a1d93fd78b2d))
