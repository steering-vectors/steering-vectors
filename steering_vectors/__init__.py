__version__ = "0.12.1"

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
from .steering_operators import (
    ablation_operator,
    ablation_then_addition_operator,
    addition_operator,
)
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
    "ablation_operator",
    "addition_operator",
    "ablation_then_addition_operator",
]
