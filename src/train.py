import functools
import logging
import time
import jax
import jax.numpy as jnp
import ml_collections
from tqdm import tqdm
from flax import jax_utils
from clu import metric_writers
from clu import periodic_actions

from lib import monitoring
from lib import consistency
from lib.state import TrainState
from lib.models import Unet
from lib.loaders import get_dataset
from lib.loss import get_loss_function
from lib.ema import update_ema_params
from lib.optimizers import create_optimizer


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
        ema_params=params,
        N=config.training.N,)
    return state


def train_step(rng, state, batch, pmap_axis='batch'):
    x0 = batch
    batch_t_ema, batch_t = consistency.ct_sample(rng, x0, state.N, state.N_ramp)
    preds_ema = state.apply_fn({'params': state.ema_params}, batch_t_ema[0], batch_t_ema[1])

    def compute_loss(params):
        preds = state.apply_fn({'params': params}, batch_t[0], batch_t[1])
        loss = state.loss_fn(preds_ema, preds)
        return loss
    compute_loss_and_grads = jax.value_and_grad(compute_loss)

    loss, grads = compute_loss_and_grads(state.params)
    grads = jax.lax.pmean(grads, axis_name=pmap_axis)
    loss = jax.lax.pmean(loss, axis_name=pmap_axis)
    metrics = {
        'loss': loss,
    }

    new_state = state.apply_gradients(grads=grads)
    return new_state, metrics
p_train_step = jax.pmap(train_step, axis_name='batch')


def train(config: ml_collections.ConfigDict,
          workdir: str,
          wandb_artifact: str = None):
    first_jax_process = jax.process_index() == 0
    # setup wandb
    if config.wandb.log_train and first_jax_process:
        monitoring.setup_wandb(config)
    # create a writer for logging training
    writer = metric_writers.create_default_writer(
        logdir=workdir,
        just_logging=not first_jax_process,
    )
    training_logger = None
    if config.training.get('log_every_steps'):
        training_logger = monitoring.training_logger(config, writer, first_jax_process)
    sample_logger = None
    if config.training.get('save_and_sample_every'):
        sample_logger = monitoring.sample_logger(config, workdir, consistency.p_sample_step)
    profile_logger = None
    if first_jax_process:
        profile_logger = periodic_actions.Profile(num_profile_steps=5, logdir=workdir)

    # setup random number generator
    local_device_count = jax.local_device_count()
    rng = jax.random.PRNGKey(config.seed)
    rng, d_rng = jax.random.split(rng)
    # load dataset
    ds_train, ds_valid = get_dataset(d_rng, config, split_into=local_device_count)
    rng, state_rng = jax.random.split(rng)
    # create initial train state
    state = create_train_state(state_rng, config)
    state = jax_utils.replicate(state)

    step_offset = 0  # dont know what this is
    train_metrics = []

    last_logged_at = time.time()
    # start training
    logging.info('Initial compilation, this might take some minutes...')
    for epoch in range(config.training.num_epochs):
        if config.data.use_streaming:
            ds_train.set_epoch(epoch)  # randomize the batches
        pbar = tqdm(ds_train)
        for step, batch in enumerate(pbar):
            pbar.set_description('Training...')
            state = consistency.update_N(state, epoch, config.training.num_epochs)
            rng, *train_step_rng = jax.random.split(rng, num=local_device_count + 1)
            train_step_rng = jnp.asarray(train_step_rng)
            state, metrics = p_train_step(train_step_rng, state, batch['image'])
            # TODO: profile
            if step == step_offset:
                logging.info('Initial compilation completed.')
                logging.info(f"Number of devices: {batch['image'].shape[0]}")
                logging.info(f"Batch size per device {batch['image'].shape[1]}")
                logging.info(f"input shape: {batch['image'].shape[2:]}")

            state = update_ema_params(state)

            if profile_logger:
                profile_logger(step)

            if training_logger:
                train_metrics += [metrics]
                train_metrics, last_logged_at = training_logger(step, train_metrics, last_logged_at)

            if sample_logger:
                rng, sample_rng = jax.random.split(rng)
                sample_logger(sample_rng, step, config.training.num_sample)

            # TODO: save a checkpoint periodically

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    return(state)