from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct                # Flax dataclasses
from flax import core
from typing import Any, Callable

class TrainState(train_state.TrainState):
    loss_fn: Callable = struct.field(pytree_node=False)
    ema_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    N: Any
