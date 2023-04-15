import torch
torch.set_printoptions(precision=4, linewidth=140, sci_mode=False)
torch.manual_seed(1)
from datasets import load_dataset
from torch.utils.data import DataLoader, default_collate
import jax
import jax.numpy as jnp
import numpy as np

from utils import timeit

BATCH_SIZE = 512
NUM_WORKERS = 1 # TODO: change to number of cpu cores
IMAGE_DTYPE = jnp.bfloat16


# convert from PIL format to JAX format
def transform_dataset_images(batch):
    @jax.jit
    def convert(orig):
        image = jnp.array(orig / 255, dtype=IMAGE_DTYPE)
        return jnp.pad(image, 2) * 2 - 1

    for idx, orig in enumerate(batch['image']):
        batch['image'][idx] = convert(np.array(orig, dtype=float))
    return batch


def collate_images(batch):
    return [item['image'] for item in batch]


dataset = load_dataset('fashion_mnist').with_transform(transform_dataset_images)

dl_train = DataLoader(
    dataset['train'],
    batch_size=BATCH_SIZE,
    collate_fn=collate_images,
    num_workers=NUM_WORKERS
)

dl_valid = DataLoader(
    dataset['test'],
    batch_size=BATCH_SIZE,
    collate_fn=collate_images,
    num_workers=NUM_WORKERS
)
