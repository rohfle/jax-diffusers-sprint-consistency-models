import jax
import jax.numpy as jnp
from flax import jax_utils


def sample_loop(rng, config, state, shape, p_sample_step):
    # shape include the device dimension: (device, per_device_batch_size, H,W,C)
    rng, x_rng = jax.random.split(rng)
    list_x0 = []
    # generate the initial sample (pure noise)
    x = jax.random.normal(x_rng, shape)
    # sample step
    for t in reversed(jnp.arange(timesteps)):
        rng, *step_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
        step_rng = jnp.asarray(step_rng)
        x, x0 = p_sample_step(step_rng, state, x, jax_utils.replicate(t), x0)
        list_x0.append(x0)
    # normalize to [0,1]
    samples = jnp.stack(list_x0)
    return (samples + 1) * 0.5


def sample_step(rng, state, sz, sigs):
    SIGS = [2.5, 5.0, 10.0, 20.0, 40.0]
    rng, rng_loop = jax.random.split(rng)
    x = jax.random.normal(rng_loop, sz)
    for sig in SIGS:
        sig = jnp.broadcast_to(sig, (len(x), 1, 1, 1))
        x = state.apply_fn(state.params, x, sig)
    return x
p_sample_step = jax.pmap(sample_step)