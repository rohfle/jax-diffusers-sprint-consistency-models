
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


def update_ema_params(state, step, config):
    if (step + 1) <= config.ema.update_after_step:
        state = p_copy_params_to_ema(state)
    elif (step + 1) % config.ema.update_every == 0:
        ema_decay = ema_decay_schedule(step)
        state = p_apply_ema_decay(state, ema_decay)
    return state

# THIS IS THE OLD CODE
# TODO: simplify above using below as reference
    # def after_step(self, learn):
    #     with torch.no_grad():
    #         mu = math.exp(2 * math.log(0.95) / self.N)
    #         # update \theta_{-}
    #         # ema_p = ema_p * mu + p * (1 - mu)
    #         for p, ema_p in zip(learn.model.parameters(), self.ema_model.parameters()):
    #             ema_p.mul_(mu).add_(p, alpha=1 - mu)