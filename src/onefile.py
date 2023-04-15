import torch
torch.set_printoptions(precision=4, linewidth=140, sci_mode=False)
torch.manual_seed(1)
from datasets import load_dataset
from torch.utils.data import DataLoader, default_collate

import jax.numpy as jnp
import numpy as np
import jax

BATCH_SIZE = 512
NUM_WORKERS = 1 # TODO: change to number of cpu cores
IMAGE_DTYPE = jnp.bfloat16

# convert from PIL format to JAX format
def transform_dataset_images(batch):
    for idx, orig in enumerate(batch['image']):
        image = jnp.array(np.array(orig, dtype=float) / 255, dtype=IMAGE_DTYPE)
        image = jnp.pad(image, 2)
        image = image * 2 - 1  # change range to -1, 1
        batch['image'][idx] = image
    return batch

def collate_jax(batch):
    elem = batch[0]
    print(type(elem['image']))
    if isinstance(elem, jax.Array):  # Some custom condition
        return []
    else:  # Fall back to `default_collate`
        return default_collate(batch)

dataset = load_dataset('fashion_mnist').with_transform(transform_dataset_images)

dl_train = DataLoader(
    dataset['train'],
    batch_size=BATCH_SIZE,
    collate_fn=lambda b: collate_jax(b)['image'],
    num_workers=NUM_WORKERS
)

dl_valid = DataLoader(
    dataset['test'],
    batch_size=BATCH_SIZE,
    collate_fn=lambda b: collate_jax(b)['image'],
    num_workers=NUM_WORKERS
)

for item in dl_train:
    print(item)
    exit()