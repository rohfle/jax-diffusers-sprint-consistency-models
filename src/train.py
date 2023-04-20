import functools
import logging
import jax
import optax
import tqdm
import jax.numpy as jnp
import ml_collections

from .state import TrainState
from .models import Unet
from .datasets import get_dataset
from .loss import get_loss_function
from .ema import update_ema


def create_optimizer(config):
    if config.optimizer == 'Adam':
        optimizer = optax.adam(
            learning_rate=config.lr,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps,
        )
    elif config.optimizer == 'AdamW':
        optimizer = optax.adamw(
            learning_rate=config.lr,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps,
        )
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


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
    image_size = config.data.image_size
    if config.ddpm.self_condition:
        image_channels = config.data.channels * 2
    else:
        image_channels = config.data.channels

    params = init_model(rng_params, image_size, image_channels, model)
    tx = create_optimizer(config.optim)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        # loss_fn should be set here
        # also other things that are in partials
        params_ema=params)
    return state







## BAD AFTER HERE FOR A WHILE

def q_sample(x, t, noise, ddpm_params):

    sqrt_alpha_bar = ddpm_params['sqrt_alphas_bar'][t, None, None, None]
    sqrt_1m_alpha_bar = ddpm_params['sqrt_1m_alphas_bar'][t,None,None,None]
    x_t = sqrt_alpha_bar * x + sqrt_1m_alpha_bar * noise

    return x_t



def noise_the_inputs(x0, N):
    noise_sched = sigmas_karras_schedule(N).flip(dims=(-1,))
    #  0 =< t < self.N-1, obtain a random integer tensor of length len(x0)
    key, subkey1, subkey2 = jax.random.split(key, num=3)
    t = jax.random.randint(subkey1, minval=0, maxval=N-1, shape=[len(x0)])
    sig_n = noise_sched[t].reshape(-1,1,1,1)
    sig_n_1 = noise_sched[t + 1].reshape(-1,1,1,1)
    # creates gaussian noise with same shape as x0
    z = jax.random.normal(subkey2, x0.shape, dtype=x0.dtype)
    noised_input_n = x0 + sig_n * z
    noised_input_n_1 = x0 + sig_n_1 * z
    new_batch = ((noised_input_n, sig_n), (noised_input_n_1, sig_n_1))
    return new_batch



def train_step(rng, state, batch, config, pmap_axis='batch'):
    x = batch['image']
    # create batched timesteps: t with shape (B,)


# ......

    rng, t_rng = jax.random.split(rng)
    batched_t = jax.random.randint(t_rng, shape=(B,), dtype = jnp.int32, minval=0, maxval= len(ddpm_params['betas']))

    # sample a noise (input for q_sample)
    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, x.shape)
    # if is_pred_x0 == True, the target for loss calculation is x, else noise
    target = x if is_pred_x0 else noise

    # generate the noisy image (input for denoise model)
    x_t = q_sample(x, batched_t, noise, ddpm_params)

    p2_loss_weight = ddpm_params['p2_loss_weight']


    def compute_loss(params):
        pred = state.apply_fn({'params':params}, x_t, batched_t)
        loss = loss_fn(flatten(pred),flatten(target))
        loss = jnp.mean(loss, axis= 1)
        assert loss.shape == (B,)
        loss = loss * p2_loss_weight[batched_t]
        return loss.mean()

    ## GOOD AFTER HERE
    grad_fn = jax.value_and_grad(compute_loss)
    loss, grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name=pmap_axis)

    loss = jax.lax.pmean(loss, axis_name=pmap_axis)
    loss_ema = jax.lax.pmean(compute_loss(state.params_ema), axis_name=pmap_axis)

    metrics = {'loss': loss,
               'loss_ema': loss_ema}

    new_state = state.apply_gradients(grads=grads)
    return new_state, metrics
p_train_step = jax.pmap(train_step, axis_name='batch')


def train(config: ml_collections.ConfigDict):
    rng = jax.random.PRNGKey(config.seed)
    rng, d_rng = jax.random.split(rng)
    ds_train, ds_valid = get_dataset(d_rng, config)
    rng, state_rng = jax.random.split(rng)
    state = create_train_state(state_rng, config)

    train_metrics = []

    for epoch in range(NUM_EPOCHS):
        if config.data.use_streaming:
            ds_train.set_epoch(epoch)  # randomize the batches
        pbar = tqdm(ds_train)
        for step, batch in enumerate(pbar):
            pbar.set_description('Training...')
            rng, *train_step_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
            train_step_rng = jnp.asarray(train_step_rng)
            state, metrics = train_step(train_step_rng, state, batch)
            train_metrics.append(metrics)
            # TODO: profile
            if step == step_offset:
                logging.info('Initial compilation completed.')
                logging.info(f"Number of devices: {batch['image'].shape[0]}")
                logging.info(f"Batch size per device {batch['image'].shape[1]}")
                logging.info(f"input shape: {batch['image'].shape[2:]}")

            state = update_ema(state, step, config)
            # TODO: log every steps
            # TODO: save a checkpoint periodically
            # TODO: generate samples


# def create_model(*, model_cls, half_precision, **kwargs):
# def initialized(key, image_size,image_channel, model):
# def create_train_state(rng, config: ml_collections.ConfigDict):
# def create_optimizer(config):
# def get_loss_fn(config):
def q_sample(x, t, noise, ddpm_params):
def p_loss(rng, state, batch, ddpm_params, loss_fn, self_condition=False, is_pred_x0=False, pmap_axis='batch'):
# def create_ema_decay_schedule(config):
# def copy_params_to_ema(state):
# def apply_ema_decay(state, ema_decay):
# def restore_checkpoint(state, workdir):
# def save_checkpoint(state, workdir):
def train(config: ml_collections.ConfigDict,
          workdir: str,
          wandb_artifact: str = None) -> TrainState:
