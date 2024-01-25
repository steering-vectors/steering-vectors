__version__ = "0.4.0"

from .layer_matching import (
    LayerMatcher,
    LayerType,
    ModelLayerConfig,
    get_num_matching_layers,
    guess_and_enhance_layer_config,
)
from .record_activations import record_activations
from .steering_vector import PatchOperator, SteeringPatchHandle, SteeringVector
from .train_steering_vector import (
    Aggregator,
    SteeringVectorTrainingSample,
    mean_aggregator,
    train_steering_vector,
)

__all__ = [
    "Aggregator",
    "mean_aggregator",
    "LayerType",
    "LayerMatcher",
    "ModelLayerConfig",
    "get_num_matching_layers",
    "guess_and_enhance_layer_config",
    "PatchOperator",
    "record_activations",
    "SteeringVector",
    "SteeringPatchHandle",
    "train_steering_vector",
    "SteeringVectorTrainingSample",
]
