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


# this is adding some non linear noise to fun
def ct_sample(rng, x0, N, N_ramp):
    # image = x0
    # noise = z
    noise_sched = sigmas_karras(N_ramp)
    #  0 =< t < self.N-1, obtain a random integer tensor of length len(x0)
    rng_t, rng_z = jax.random.split(rng)
    t = jax.random.randint(rng_t, minval=0, maxval=N-1, shape=[len(x0)])
    sig_n = noise_sched[t].reshape(-1,1,1,1)
    sig_n_1 = noise_sched[t + 1].reshape(-1,1,1,1)
    # creates gaussian noise with same shape as x0
    z = jax.random.normal(rng_z, x0.shape, dtype=x0.dtype)
    noised_input_n = x0 + sig_n * z
    noised_input_n_1 = x0 + sig_n_1 * z
    new_ema_batch = (noised_input_n, sig_n)
    new_batch = (noised_input_n_1, sig_n_1)
    return new_ema_batch, new_batch


def model_wrapper(apply_fn, epsilon):
    def apply(params, x, sigma):
        c_skip, c_out, c_in = scalings(sigma.reshape(-1,1,1,1), epsilon)
        return c_skip * x + c_out * apply_fn(params, x, sigma.squeeze())
    return apply

