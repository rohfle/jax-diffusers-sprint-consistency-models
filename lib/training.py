import jax.numpy as jnp
from jax import jit, Array
from flax import linen as nn
from jax import random, disable_grad

from .consistency import *

from typing import Tuple


class ConsistencyCB(TrainCB):
    def before_batch(batch, epoch, num_epochs, lastN):
        key = random.PRNGKey(0)
        # this value slowly increases over the epochs - Progressive Resizing (PR) training technique?
        # Is this link relevant? https://docs.mosaicml.com/en/latest/method_cards/progressive_resizing.html
        N = jnp.ceil(jnp.sqrt((epoch+1 * (lastN**2 - 4) / num_epochs) + 4) - 1) + 1
        # .flip(dims=(-1,)) flips along its last dimension eg [1,2,3,4] -> [4,3,2,1]
        # descending to ascending ramp
        noise_sched = sigmas_karras(N).flip(dims=(-1,))
        x0 = batch # original images, x_0
        #  0 =< t < self.N-1, obtain a random integer tensor of length len(x0)
        t = random.randint(key, minval=0, maxval=N-1, shape=[len(x0)])
        # adds 1 to every item in t
        t_1 = t+1
        sig_n = noise_sched[t].reshape(-1,1,1,1)
        sig_n_1 = noise_sched[t_1].reshape(-1,1,1,1)
        # creates gaussian noise with same shape as x0
        key, subkey = random.split(key)
        z = random.normal(subkey, x0.shape, dtype=x0.dtype)
        noised_input_n = x0 + sig_n*z
        noised_input_n_1 = x0 + sig_n_1*z
        new_batch = ((noised_input_n, sig_n), (noised_input_n_1, sig_n_1))
        return new_batch, N

    # called to make predictions on input data
    def predict(self, learn):
        # n_inp is the number of inputs the model expects to receive
        with disable_grad():
            learn.preds = self.ema_model(*learn.batch[:self.n_inp])

    # called to calculated the loss between predicted and actual values
    def get_loss(self, learn):
        learn.loss = learn.loss_func(learn.preds, learn.model(*learn.batch[self.n_inp:]))

    # called after each optimization step, post processing
    def after_step(self, params, ema_params, N):
        with disable_grad():
            mu = jnp.exp(2 * jnp.log(0.95) / N)

            for p, ema_p in zip(params, ema_params):
                ema_p.mul_(mu).add_(p, alpha=1 - mu)
