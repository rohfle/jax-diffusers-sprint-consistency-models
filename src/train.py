import logging
import os
import time
from copy import deepcopy
import jax
import einops
import jax.numpy as jnp
import ml_collections
from tqdm import tqdm
from flax import jax_utils
from clu import metric_writers
from clu import periodic_actions
from diffusers.models import FlaxUNet2DConditionModel

from lib import checkpoints
from lib import monitoring
from lib import consistency
from lib.state import TrainState
from lib.models import Unet
from lib.loaders import get_dataset
from lib.loss import get_loss_function
from lib.ema import update_ema_params
from lib.optimizers import create_optimizer


def determine_dtype(half_precision):
    if half_precision:
        platform = jax.local_devices()[0].platform
        if platform == 'tpu':
            return jnp.bfloat16
        else:
            return jnp.float16
    else:
        return jnp.float32


def create_model(*, model_cls, half_precision, **kwargs):
    model_dtype = determine_dtype(half_precision)
    return model_cls(dtype=model_dtype, **kwargs)


def load_teacher_model(rng, model_path, half_precision, hidden_states_shape=(4, 77, 768), **kwargs):
    model_dtype = determine_dtype(half_precision)
    model, params = FlaxUNet2DConditionModel.from_pretrained(
        model_path,
        subfolder="unet",
        dtype=model_dtype,
        revision="bf16" if half_precision else None,
        **kwargs
    )
    hidden_states = jax.random.normal(rng, hidden_states_shape, dtype=model_dtype)
    return model, params, hidden_states


def init_model(key, image_size, image_channels, model):
    input_shape = (1, image_size, image_size, image_channels)
    init = jax.jit(model.init)
    variables = init(
            {'params': key},
            jnp.ones(input_shape, model.dtype), # x_t
            jnp.ones(input_shape[:1], model.dtype) # t
        )

    return variables['params']


def create_train_state(rng, config: ml_collections.ConfigDict):
    model_dtype = determine_dtype(config.training.half_precision)
    model = FlaxUNet2DConditionModel(
        dtype=model_dtype,
        sample_size=config.data.image_size,
        in_channels=config.data.channels,
        out_channels=config.data.channels,
        use_memory_efficient_attention=True,
        # from stable-diffusion-2
        attention_head_dim=(
            5,
            10,
            20,
            20,
        ),
    )

    rng, rng_params, rng_hidden = jax.random.split(rng, num=3)
    params = model.init_weights(rng_params)
    encoder_hidden_states_shape = (config.data.batch_size // 4, 1, model.cross_attention_dim)
    encoder_hidden_states = jax.random.normal(rng_hidden, encoder_hidden_states_shape, dtype=model.dtype)
    apply_fn = lambda params, x_t, t: model.apply(params, x_t, t, encoder_hidden_states).sample

    apply_fn = consistency.model(apply_fn, config.training.epsilon)
    ema_params = deepcopy(params)  # probably not necessary
    tx = create_optimizer(config.optim)
    loss_fn = get_loss_function(config.training.loss_type)

    if config.training.get('mode') == 'distill':
        rng, rng_teach = jax.random.split(rng)
        teacher_model, teacher_params, hidden_states = load_teacher_model(
            rng_teach,
            config.training.teacher_model,
            # TODO: move to config
            use_memory_efficient_attention=True,
            hidden_states_shape=(4, 1, 1024),  # batch size, seq length, embedding dim
            half_precision=config.training.half_precision)
        # hide the hidden states and also allow reorder of dimensions
        def teacher_model_apply_fn(params, x_t, t):
            results = []
            MINIBATCH_SIZE = 4  # smaller model
            for idx in range(0, len(x_t), MINIBATCH_SIZE):
                x_t_mini = x_t[idx:idx + MINIBATCH_SIZE]
                t_mini = t[idx:idx + MINIBATCH_SIZE]
                x_mini = teacher_model.apply(params, x_t_mini, t_mini, encoder_hidden_states=hidden_states).sample
                results += [x_mini]
            return jnp.concatenate(results)
        consistency_fn = consistency.distillation
    else:
        teacher_model_apply_fn = None
        teacher_params = None
        consistency_fn = consistency.training

    state = TrainState.create(
        teacher_apply_fn=teacher_model_apply_fn,
        teacher_params=teacher_params,
        consistency_fn=consistency_fn,
        apply_fn=apply_fn,
        params=params,
        tx=tx,
        loss_fn=loss_fn,
        ema_params=ema_params,
        N=config.training.N)
    return state


def train_step(rng, state, batch, pmap_axis='batch'):
    batch_t_ema, batch_t = state.consistency_fn(rng, batch, state)
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
    sampledir = os.path.join(workdir, 'samples')
    os.makedirs(sampledir, exist_ok=True)
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
        sample_logger = monitoring.sample_logger(config, sampledir)
    profile_logger = None
    if first_jax_process:
        profile_logger = periodic_actions.Profile(num_profile_steps=5, logdir=workdir)

    # setup random number generator
    local_device_count = jax.local_device_count()
    rng = jax.random.PRNGKey(config.seed)
    rng, d_rng, state_rng = jax.random.split(rng, num=3)
    # create initial train state
    state = create_train_state(state_rng, config)
    state = checkpoints.restore(state, workdir)
    step_offset = int(state.step)
    step = 0
    # load dataset
    ds_train, ds_valid = get_dataset(d_rng, config, split_into=local_device_count, skip_batches=step_offset)

    state = jax_utils.replicate(state)
    # start training
    logging.info('Initial compilation, this might take some minutes...')
    for epoch in range(config.training.num_epochs):
        ds_train.set_epoch(epoch)  # randomize the batches
        pbar = tqdm(ds_train)
        for batch in pbar:
            step += 1
            if batch['image'] is None:
                print("skipping batch for step", step, "...")
                continue  # skipping steps until at step offset
            step_start_time = time.time()
            pbar.set_description('Training...')
            state = consistency.update_N(state, epoch, config.training.num_epochs)
            rng, *train_step_rng = jax.random.split(rng, num=local_device_count + 1)
            train_step_rng = jnp.asarray(train_step_rng)
            state, metrics = p_train_step(train_step_rng, state, batch['image'])

            if step == step_offset:
                logging.info('Initial compilation completed.')
                logging.info(f"Number of devices: {batch['image'].shape[0]}")
                logging.info(f"Batch size per device {batch['image'].shape[1]}")
                logging.info(f"input shape: {batch['image'].shape[2:]}")

            state = update_ema_params(state)

            if profile_logger:
                profile_logger(step)

            if training_logger:
                training_logger(step, metrics, step_start_time)

            if sample_logger:
                rng, sample_rng = jax.random.split(rng)
                sample_logger(sample_rng, step, state, batch)

            checkpoints.save(state, workdir)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    return(state)