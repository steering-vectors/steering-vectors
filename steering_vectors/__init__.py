__version__ = "0.9.0"

from .aggregators import (
    Aggregator,
    logistic_aggregator,
    mean_aggregator,
    pca_aggregator,
)
from .layer_matching import (
    LayerMatcher,
    LayerType,
    ModelLayerConfig,
    get_num_matching_layers,
    guess_and_enhance_layer_config,
)
from .record_activations import record_activations
from .steering_vector import PatchDeltaOperator, SteeringPatchHandle, SteeringVector
from .train_steering_vector import (
    SteeringVectorTrainingSample,
    aggregate_activations,
    extract_activations,
    train_steering_vector,
)

__all__ = [
    "Aggregator",
    "mean_aggregator",
    "pca_aggregator",
    "logistic_aggregator",
    "LayerType",
    "LayerMatcher",
    "ModelLayerConfig",
    "get_num_matching_layers",
    "guess_and_enhance_layer_config",
    "PatchDeltaOperator",
    "record_activations",
    "SteeringVector",
    "SteeringPatchHandle",
    "train_steering_vector",
    "SteeringVectorTrainingSample",
    "aggregate_activations",
    "extract_activations",
]
