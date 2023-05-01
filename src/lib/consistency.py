from typing import Tuple
import jax.numpy as jnp
import jax



def calculate_N(N, epoch, num_epochs):
    return jnp.ceil(jnp.sqrt((epoch+1 * (N**2 - 4) / num_epochs) + 4) - 1) + 1


def update_N(state, epoch, num_epochs):
    new_N = calculate_N(state.N, epoch, num_epochs)
    # because the schedule is always flipped, switch 0, 1 -> 1, 0
    ramp = jnp.linspace(1, 0, int(new_N[0]))
    # tile to match the dimensions of N
    new_N_ramp = jnp.tile(ramp, (len(new_N), 1))
    state = state.replace(N=new_N, N_ramp=new_N_ramp)
    return state


# Page 3, Network and preconditioning (Section 5), column Ours ("EDM")
def scalings(sig : jax.Array, eps : float, sig_data=0.5) -> Tuple[jax.Array, jax.Array, jax.Array]:
    c_skip = sig_data ** 2 / ((sig - eps) ** 2 + sig_data ** 2)
    c_out = (sig - eps) * sig_data / jnp.sqrt(sig ** 2 + sig_data ** 2)
    c_in = 1 / jnp.sqrt((sig - eps) ** 2 + sig_data ** 2)
    return c_skip, c_out, c_in


# Page 3, Sampling (Section 3), column Ours ("EDM")
def sigmas_karras(n_ramp : jax.Array, sigma_min=0.002, sigma_max=80., rho=7.) -> jax.Array:
    # this is "Time steps" formula
    min_inv_rho : float = sigma_min ** (1. / rho)
    max_inv_rho : float = sigma_max ** (1. / rho)
    sigmas = (max_inv_rho + n_ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


@jax.jit
def training(rng, x0, state):
    schedule = sigmas_karras(state.N_ramp)
    rng_n, rng_z = jax.random.split(rng)
    n = jax.random.randint(rng_n, minval=0, maxval=state.N-1, shape=[len(x0)])
    t = schedule[n].reshape(-1,1,1,1)
    t2 = schedule[n + 1].reshape(-1,1,1,1)
    # creates gaussian noise with same shape as x0
    z = jax.random.normal(rng_z, x0.shape, dtype=x0.dtype)
    x_t = x0 + t * z
    # TRAINING SPECIFIC
    x_t2 = x0 + t2 * z
    # END SPECIFIC
    new_ema_batch = (x_t, t)
    new_batch = (x_t2, t2)
    return new_ema_batch, new_batch


@jax.jit
def distillation(rng, x0, state):
    schedule = sigmas_karras(state.N_ramp)
    rng_n, rng_z = jax.random.split(rng)
    n = jax.random.randint(rng_n, minval=0, maxval=state.N-1, shape=[len(x0)])
    t = schedule[n].reshape(-1,1,1,1)
    t2 = schedule[n + 1].reshape(-1,1,1,1)
    # creates gaussian noise with same shape as x0
    z = jax.random.normal(rng_z, x0.shape, dtype=x0.dtype)
    x_t = x0 + t * z
    # DISTILLATION SPECIFIC
    # first step
    teach_x = state.teacher_apply_fn({'params': state.teacher_params}, x_t, t.squeeze())
    d = (x_t - teach_x) / t   #  if teach_x were perfect, d would be z
    x_tmp = x_t + d * (t2 - t)
    # second step
    teach_x2 = state.teacher_apply_fn({'params': state.teacher_params}, x_tmp, t2.squeeze())
    d2 = (x_tmp - teach_x2) / t2
    x_t2 = x_t + (d + d2) * (t2 - t) / 2  # takes average of imperfect noise
    # END SPECIFIC
    new_ema_batch = (x_t, t)  # TODO: verify these lines
    new_batch = (x_t2, t2)    # TODO: verify these lines
    return new_ema_batch, new_batch


def model(apply_fn, epsilon):
    """Consistency model as a wrapper for .apply"""
    def apply(params, x, sigma):
        c_skip, c_out, c_in = scalings(sigma.reshape(-1,1,1,1), epsilon)
        return c_skip * x + c_out * apply_fn(params, x, sigma.squeeze())
    return apply


def sample(rng, epsilon, state, shape):
    SIGS = TIMESTEPS = [2.5, 5.0, 10.0, 20.0, 40.0]
    rng, rng_loop = jax.random.split(rng)
    x = jax.random.normal(rng_loop, shape)
    sig = jnp.broadcast_to(TIMESTEPS[0], (len(x), 1, 1, 1))
    x = state.apply_fn({'params': state.params}, x, sig)
    for sig in TIMESTEPS[1:]:
        rng, rng_loop = jax.random.split(rng)
        z = jax.random.normal(rng_loop, shape)
        x = x + jnp.sqrt(sig ** 2 - epsilon ** 2) * z
        sig = jnp.broadcast_to(sig, (len(x), 1, 1, 1))
        x = state.apply_fn({'params': state.params}, x, sig)
    return (x + 1) * 0.5