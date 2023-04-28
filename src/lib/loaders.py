import jax
from jax import numpy as jnp
import numpy as np
import ml_collections
from flax import jax_utils
from datasets import load_dataset

from . import imagetools

def get_dataset(rng, config : ml_collections.ConfigDict, split_into=1):
    batch_size = config.data.batch_size
    if batch_size % split_into != 0:
        raise ValueError('Batch size must be divisible by the number of sub batches')

    if config.data.use_streaming:
        dataset = load_dataset(config.data.dataset, streaming=True).shuffle(
            seed=42, # rng currently not working
            buffer_size=config.data.shuffle_buffer_size
        )
    else:
        dataset = load_dataset(config.data.dataset).shuffle(seed=42)
        dataset['train'] = dataset['train'].flatten_indices()
        if 'test' in dataset:
            dataset['test'] = dataset['test'].flatten_indices()
    dataset = dataset.with_format('jax')

    def transform_and_collate(batch):
        images = imagetools.crop_resize_bulk(batch['image'], config.data.image_size)
        images = np.stack(images)  # stack all the images into one giant array
        # TODO: look at JIT / JAX optimization for image manipulation
        images = images.reshape(*images.shape[:3], -1)  # to allow for grayscale 1 channel images
        images = images / 255  # range 0.0 to 1.0
        images = np.pad(images, [[0], [2], [2], [0]])  # add padding to width and height
        images = jnp.array(images) * 2 - 1  # output range will be -1.0 to 1.0

        # reshape in a way that supports pmap
        result = {
            'image': images.reshape((1, split_into, -1) + images.shape[1:])
        }

        if 'label' in batch:
            labels = jnp.array(batch['label'])
            result['label'] = labels.reshape((1, split_into, -1) + labels.shape[1:])

        return result

    ds_train = dataset['train'].map(transform_and_collate, batched=True, batch_size=config.data.batch_size, drop_last_batch=True)
    if 'test' in dataset:
        ds_valid = dataset['test'].map(transform_and_collate, batched=True, batch_size=config.data.batch_size, drop_last_batch=True)
    else:
        ds_valid = None
    return ds_train, ds_valid