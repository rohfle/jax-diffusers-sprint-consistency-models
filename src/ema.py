
import jax
import jax.numpy as jnp


def calculate_mu(N):
    return jnp.exp(2 * jnp.log(0.95) / N)

@jax.pmap
def update_ema_params(state):
    mu = calculate_mu(state.N)
    update_ema_fn = lambda p_ema, p: p_ema * mu + p * (1. - mu)
    params_ema = jax.tree_map(update_ema_fn, state.params_ema, state.params)
    return state.replace(params_ema=params_ema)