import functools
import logging
import jax
import optax
import tqdm
import jax.numpy as jnp
import ml_collections

from . import consistency
from .state import TrainState
from .models import Unet
from .datasets import get_dataset
from .loss import get_loss_function
from .ema import update_ema_params
from .optimizers import create_optimizer


def create_model(*, model_cls, half_precision, **kwargs):
    if half_precision:
        platform = jax.local_devices()[0].platform
        if platform == 'tpu':
            model_dtype = jnp.bfloat16
        else:
            model_dtype = jnp.float16
    else:
        model_dtype = jnp.float32
    return model_cls(dtype=model_dtype, **kwargs)


def init_model(key, image_size, image_channels, model):
    input_shape = (1, image_size, image_size, image_channels)
    init = jax.jit(model.init)
    variables = init(
            {'params': key},
            jnp.ones(input_shape, model.dtype), # x noisy image
            jnp.ones(input_shape[:1], model.dtype) # t
        )

    return variables['params']


def create_train_state(rng, config: ml_collections.ConfigDict):
    model = create_model(
        model_cls=Unet,
        half_precision=config.training.half_precision,
        dim=config.model.dim,
        out_dim=config.data.channels,
        dim_mults=config.model.dim_mults)

    rng, rng_params = jax.random.split(rng)
    params = init_model(rng_params, config.data.image_size, config.data.channels, model)
    tx = create_optimizer(config.optim)
    loss_fn = get_loss_function(config.training.loss_type)
    apply_fn = consistency.model_wrapper(model.apply, config.training.epsilon)

    state = TrainState.create(
        apply_fn=apply_fn,
        params=params,
        tx=tx,
        loss_fn=loss_fn,
        params_ema=params)
    return state


def train_step(rng, state, batch, pmap_axis='batch'):
    x0 = batch['image']
    batch_t_ema, batch_t = consistency.ct_sample(rng, x0, state.N)
    preds_ema = state.apply_fn({'params': state.ema_params}, batch_t_ema[0], batch_t_ema[1])

    def compute_loss(params):
        preds = state.apply_fn({'params': params}, batch_t[0], batch_t[1])
        loss = state.loss_fn(preds_ema, preds)
        return loss
    compute_loss_and_grads = jax.value_and_grad(compute_loss, has_aux=True)

    loss, grads = compute_loss_and_grads(state.params)
    grads = jax.lax.pmean(grads, axis_name=pmap_axis)
    loss = jax.lax.pmean(loss, axis_name=pmap_axis)
    metrics = {
        'loss': loss,
    }

    new_state = state.apply_gradients(grads=grads)
    return new_state, metrics
p_train_step = jax.pmap(train_step, axis_name='batch')


def train(config: ml_collections.ConfigDict):
    rng = jax.random.PRNGKey(config.seed)
    rng, d_rng = jax.random.split(rng)
    ds_train, ds_valid = get_dataset(d_rng, config)
    rng, state_rng = jax.random.split(rng)
    state = create_train_state(state_rng, config)

    step_offset = 0  # dont know what this is
    train_metrics = []

    for epoch in range(config.training.num_epochs):
        if config.data.use_streaming:
            ds_train.set_epoch(epoch)  # randomize the batches
        pbar = tqdm(ds_train)
        for step, batch in enumerate(pbar):
            pbar.set_description('Training...')
            state = consistency.update_N(state, epoch, config.training.num_epochs)
            rng, *train_step_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
            train_step_rng = jnp.asarray(train_step_rng)
            state, metrics = p_train_step(train_step_rng, state, batch)
            train_metrics.append(metrics)
            # TODO: profile
            if step == step_offset:
                logging.info('Initial compilation completed.')
                logging.info(f"Number of devices: {batch['image'].shape[0]}")
                logging.info(f"Batch size per device {batch['image'].shape[1]}")
                logging.info(f"input shape: {batch['image'].shape[2:]}")

            state = update_ema_params(state, step, config)
            # TODO: log every steps
            # TODO: save a checkpoint periodically
            # TODO: generate samples