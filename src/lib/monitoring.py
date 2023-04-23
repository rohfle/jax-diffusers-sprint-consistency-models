import functools
import os
import logging
import time
import jax
import wandb
from tqdm import trange
from jax import numpy as jnp
from ml_collections import ConfigDict
import jax

from . import imagetools
from . import sampling


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
    @functools.wraps
    def inner(step, train_metrics, last_log_time):
        if (step + 1) % config.training.log_every_steps != 0:
            return train_metrics, False
        # time to log some stuff
        summary = {
            f'train/{k}': v
            for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()
        }
        summary['time/seconds_per_step'] =  (time.time() - last_log_time) / config.training.log_every_steps
        # write to disk
        if writer is not None:
            writer.write_scalars(step + 1, summary)
        # write to wandb
        if use_wandb and config.wandb.log_train:
            wandb.log({
                'train/step': step,
                **summary,
            })
        train_metrics = []
        last_log_time = time.time()
        return train_metrics, last_log_time
    return inner


def sample_logger(config, workdir, p_sample_step, num_samples):
    sample_dir = os.path.join(workdir, 'samples')
    os.makedirs(exist_ok=True)

    @functools.wraps
    def inner(rng, step, state, batch):
        if (step + 1) % config.training.save_and_sample_every != 0:
            return
        logging.info(f'generating samples....')
        samples = sample(rng, config, state, batch, p_sample_step, num_samples)
        samples_array = imagetools.make_grid(samples, num_samples)
        sample_path = os.path.join(sample_dir, f'iter_{step + 1}_host_{jax.process_index()}.png')
        imagetools.save_image(samples, sample_path)
        if config.wandb.log_sample:
            wandb_log_image(samples_array, step + 1)

    return inner


def sample(rng, config, state, batch, p_sample_step, num_samples):
    samples = []
    for i in trange(num_samples):
        rng, sample_rng = jax.random.split(rng)
        sample = sampling.sample_loop(
            sample_rng,
            config,
            state,
            tuple(batch['image'].shape),
            p_sample_step
        )
        samples.append(sample)
    return jnp.concatenate(samples)







