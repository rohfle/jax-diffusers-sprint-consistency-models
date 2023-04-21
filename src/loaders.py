import numpy as np
import ml_collections
from datasets import load_dataset

def get_dataset(rng, config : ml_collections.ConfigDict):
    if config.data.use_streaming:
        dataset = load_dataset(config.data.dataset, streaming=True).shuffle(
            seed=42, # rng currently not working
            buffer_size=config.data.shuffle_buffer_size
        )
    else:
        dataset = load_dataset(config.data.dataset).shuffle(seed=42)
        dataset['train'] = dataset['train'].flatten_indices()
        dataset['test'] = dataset['test'].flatten_indices()
    dataset = dataset.with_format('jax')

    def transform_and_collate(batch):
        images = np.stack(batch['image'])  # stack all the images into one giant array
        # TODO: look at JIT / JAX optimization for image manipulation
        images = images.reshape(*images.shape[:3], -1)  # to allow for grayscale 1 channel images
        images = images / 255  # range 0.0 to 1.0
        images = np.pad(images, [[0], [2], [2], [0]])  # add padding to width and height
        images = np.array(images * 2 - 1)  # output range will be -1.0 to 1.0
        labels = np.array(batch['label'])

        return {
            # by wrapping with a list here, its possible to keep the batch all together
            'image': images.reshape(1, *images.shape),
            'label': labels.reshape(1, *labels.shape),
        }

    ds_train = dataset['train'].map(transform_and_collate, batched=True, batch_size=config.data.batch_size, drop_last_batch=True)
    ds_valid = dataset['test'].map(transform_and_collate, batched=True, batch_size=config.data.batch_size, drop_last_batch=True)
    return ds_train, ds_valid