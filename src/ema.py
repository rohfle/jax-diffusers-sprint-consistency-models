
import jax
import jax.numpy as jnp


def ema_decay_schedule(step, config):
    count = jnp.clip(step - config.update_after_step - 1, a_min = 0.)
    value = 1 - (1 + count / config.inv_gamma) ** - config.power
    ema_rate = jnp.clip(value, a_min = config.min_value, a_max = config.beta)
    return ema_rate


def copy_params_to_ema(state):
    state = state.replace(params_ema=state.params)
    return state
p_copy_params_to_ema = jax.pmap(copy_params_to_ema)


def apply_ema_decay(state, ema_decay):
    update_ema_fn = lambda p_ema, p: p_ema * ema_decay + p * (1. - ema_decay)
    params_ema = jax.tree_map(update_ema_fn, state.params_ema, state.params)
    state = state.replace(params_ema=params_ema)
    return state
p_apply_ema_decay = jax.pmap(apply_ema_decay)


def update_ema(state, step, config):
    if (step + 1) <= config.ema.update_after_step:
        state = p_copy_params_to_ema(state)
    elif (step + 1) % config.ema.update_every == 0:
        ema_decay = ema_decay_schedule(step)
        logging.info(f'update ema parameters with decay rate {ema_decay}')
        state = p_apply_ema_decay(state, ema_decay)
    return state