import jax
from jax import numpy as jnp
import numpy as np
import ml_collections
from flax import jax_utils
from datasets import load_dataset

from . import imagetools


class JAXImageTransform:
    """Adds batch skipping to map functions"""
    def __init__(self, config, skip_batches=0, split_into=1):
        self.skip_batches = skip_batches
        self.split_into = split_into
        self.image_channels = config.data.channels
        self.image_size = config.data.image_size
        self.image_pad = 2

    def __call__(self, batch):
        if self.skip_batches > 0:
            # gonna skip this batch anyway, dont bother processing it
            self.skip_batches -= 1
            return {k: [None] for k in batch.keys()}  # empty batch

        # make sure the result is always divisible by split_into
        # even if it means dropping images
        max_idx = len(batch['image']) // self.split_into * self.split_into
        images = imagetools.normalize_images(
            batch['image'][:max_idx],
            self.image_channels,
            self.image_size,
            self.image_pad,
            dtype=jnp.bfloat16,
        )

        # reshape in a way that supports pmap
        result = {
            'image': images.reshape((1, self.split_into, -1) + images.shape[1:])
        }

        # if 'label' in batch:
        #     labels = jnp.array(batch['label'][:max_idx])
        #     result['label'] = labels.reshape((1, self.split_into, -1) + labels.shape[1:])

        return result


def get_dataset(rng, config : ml_collections.ConfigDict, split_into=1, skip_batches=0):
    batch_size = config.data.batch_size
    if batch_size % split_into != 0:
        raise ValueError('Batch size must be divisible by the number of sub batches')

    dataset = load_dataset(config.data.dataset, streaming=True).shuffle(
        seed=42, # rng currently not working
        buffer_size=config.data.shuffle_buffer_size
    ).with_format('jax')

    # TODO: there is a skip() function? how would this work with multiple epochs
    image_transform = JAXImageTransform(config, skip_batches=skip_batches, split_into=split_into)
    ds_train = dataset['train'].map(image_transform, batched=True, batch_size=config.data.batch_size, drop_last_batch=True)
    if 'test' in dataset:
        image_transform = JAXImageTransform(config, split_into=split_into)
        ds_valid = dataset['test'].map(image_transform, batched=True, batch_size=config.data.batch_size, drop_last_batch=True)
    else:
        ds_valid = None
    return ds_train, ds_valid