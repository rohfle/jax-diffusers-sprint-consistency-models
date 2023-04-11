from ..consistency import scalings_jit as scalings
from typing import Callable

from jax import jit
from flax import linen as nn

# ConsistencyUNet implements the above function "Appendix C: Additional Experimental Details"
class ConsistencyUNet(nn.Module):
    eps : float
    F : Callable

    def __init__(self, eps, model):
        self.eps = eps
        self.F = model

    @nn.compact
    def __call__(self, inputs):
        x, sigma = inputs
        c_skip, c_out, c_in = scalings(sigma.reshape(-1,1,1,1), self.eps)
        return c_skip * x + c_out * self.F((x, sigma.squeeze()))
