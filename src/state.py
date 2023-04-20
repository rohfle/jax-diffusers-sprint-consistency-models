from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct                # Flax dataclasses
from flax import core
from typing import Any

@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')

class TrainState(train_state.TrainState):
    loss_fn: Any
    ema_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    metrics: Metrics
