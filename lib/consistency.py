import jax.numpy as jnp
from jax import jit, Array
from flax import linen as nn
from jax import random

from typing import Tuple


# Page 3, Sampling (Section 3), column Ours ("EDM")
# not jitable due to jnp.linspace
def sigmas_karras(n : int, sigma_min=0.002, sigma_max=80., rho=7.) -> Array:
    # rising from 0 to 1 over n steps
    ramp : Array = jnp.linspace(0, 1, n)
    # this is "Time steps" formula
    min_inv_rho : float = sigma_min ** (1. / rho)
    max_inv_rho : float = sigma_max ** (1. / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas

# Page 3, Network and preconditioning (Section 5), column Ours ("EDM")
def scalings(sig : Array, eps : float, sig_data=0.5) -> Tuple[Array, Array, Array]:
    c_skip = sig_data ** 2 / ((sig - eps) ** 2 + sig_data ** 2)
    c_out = (sig - eps) * sig_data / jnp.sqrt(sig ** 2 + sig_data ** 2)
    c_in = 1 / jnp.sqrt((sig - eps) ** 2 + sig_data ** 2)
    return c_skip, c_out, c_in

scalings_jit = jit(scalings)


if __name__ == "__main__":
    print(sigmas_karras(10))

    c_skip, c_out, c_in = scalings_jit(jnp.array([0.1, 0.5, 0.2]), 0.002)
    print(c_skip)
    print(c_out)
    print(c_in)



