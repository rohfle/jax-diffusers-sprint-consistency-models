import functools
import os
import logging
import time
import jax
import wandb
from tqdm import trange
from jax import numpy as jnp
from flax import jax_utils
from ml_collections import ConfigDict
import jax

from . import consistency
from . import imagetools


def wandb_log_image(samples_array, step):
    sample_images = wandb.Image(samples_array, caption=f"step {step}")
    wandb.log({'samples': sample_images })


def wandb_log_model(workdir, step):
    artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="ddpm_model")
    artifact.add_file(f"{workdir}/checkpoint_{step}")
    wandb.run.log_artifact(artifact)


def to_wandb_config(d: ConfigDict, parent_key: str = '', sep: str ='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, ConfigDict):
            items.extend(to_wandb_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def setup_wandb(config):
    wandb_config = to_wandb_config(config)
    wandb.init(
        entity=config.wandb.entity,
        project=config.wandb.project,
        job_type=config.wandb.job_type,
        config=wandb_config)


def training_logger(config, writer, use_wandb):
    def inner(step, metrics, step_start_time):
        if (step + 1) % config.training.log_every_steps != 0:
            return
        # time to log some stuff
        summary = {
            f'train/{k}': v
            for k, v in jax.tree_map(lambda x: x.mean(), metrics).items()
        }
        summary['time/seconds_per_step'] =  (time.time() - step_start_time)
        # write to disk
        if writer is not None:
            writer.write_scalars(step + 1, summary)
        # write to wandb
        if use_wandb and config.wandb.log_train:
            wandb.log({
                'train/step': step + 1,
                **summary,
            })
        return
    return inner


def sample_logger(config, sample_dir):
    def inner(rng, step, state, batch):
        if (step + 1) % config.training.save_and_sample_every != 0:
            return
        logging.info(f'generating samples....')
        samples = sample_many(rng, config, state, batch, config.training.num_samples)
        samples_array = imagetools.make_grid(samples, config.training.num_samples)
        sample_path = os.path.join(sample_dir, f'iter_{(step + 1):04}_host_{jax.process_index()}.png')
        imagetools.save_image(samples_array, sample_path)
        if config.wandb.log_sample:
            wandb_log_image(samples_array, step + 1)

    return inner


def sample_many(rng, config, state, batch, num_samples):
    samples = []
    state = jax_utils.unreplicate(state)
    shape = batch['image'][0].shape
    # divide required samples by batch size
    runs = (num_samples - 1) // shape[0] + 1

    for i in trange(runs):
        rng, sample_rng = jax.random.split(rng)
        sample_rng = jnp.asarray(sample_rng)
        sample = consistency.sample(
            sample_rng,
            config.training.epsilon,
            state,
            shape,
        )
        samples.append(sample)
    return jnp.concatenate(samples)







