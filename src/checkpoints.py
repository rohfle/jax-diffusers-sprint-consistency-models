import jax
from flax.training import checkpoints


def restore_checkpoint(state, workdir):
  return checkpoints.restore_checkpoint(workdir, state)

def save_checkpoint(state, workdir):
  if jax.process_index() == 0:
    # get train state from the first replica
    state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint(workdir, state, step, keep=3)