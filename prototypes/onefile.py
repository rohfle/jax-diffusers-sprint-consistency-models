from datasets import load_dataset
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from tqdm import tqdm
import time
RUN_TIMESTAMP = int(time.time())

np.set_printoptions(precision=2, linewidth=140)

import time
# a decorator to time functions
def timeit(func):
    def runit(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        print("time ", func.__name__, ":", duration)
        return result
    return runit

def timediter(iterable, label):
    it = iter(iterable)
    while True:
        try:
            start = time.time()
            value = next(it)
            end = time.time()
            print(f"time {label} yield: {end - start:.6f} seconds")
            yield value
        except StopIteration:
            break

def profile(name, limit=3):
    def outer(func):
        left = limit
        def runit(*args, **kwargs):
            if left > 0:
                with jax.profiler.trace(name, create_perfetto_link=True):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return runit
    return outer

def profileiter(iterable, label):
    it = iter(iterable)
    try:
        with jax.profiler.trace(label, create_perfetto_link=True):
            yield next(it)

        while True:
            yield next(it)
    except StopIteration:
        pass



class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


BATCH_SIZE = 32 # TODO: change
NUM_EPOCHS = 1 # TODO: change to 100 or something
IMAGE_DTYPE = jnp.bfloat16

SHUFFLE_SEED = 42
SHUFFLE_BUFFER_SIZE = 10_000

USE_STREAMING = True
if USE_STREAMING:
    dataset = load_dataset('fashion_mnist', streaming=True).shuffle(seed=SHUFFLE_SEED, buffer_size=SHUFFLE_BUFFER_SIZE)
else:
    dataset = load_dataset('fashion_mnist').shuffle(seed=SHUFFLE_SEED)
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

ds_train = dataset['train'].map(transform_and_collate, batched=True, batch_size=BATCH_SIZE, drop_last_batch=True)
ds_valid = dataset['test'].map(transform_and_collate, batched=True, batch_size=BATCH_SIZE, drop_last_batch=True)
image_shape = np.array(next(iter(ds_train))['image']).shape[1:]

from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct                # Flax dataclasses
import optax                           # Common loss functions and optimizers

@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')

class TrainState(train_state.TrainState):
    metrics: Metrics

def create_train_state(module, rng, learning_rate, momentum, image_shape):
    """Creates an initial `TrainState`."""
    params = module.init(rng, jnp.ones([1, *image_shape]))['params'] # initialize parameters by passing a template image
    tx = optax.sgd(learning_rate, momentum)
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        metrics=Metrics.empty()
    )

@jax.jit
def train_step(state, batch):
    """Train for a single step."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=batch['label']
        ).mean()
        return loss
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state

@jax.jit
def compute_metrics(*, state, batch):
    logits = state.apply_fn({'params': state.params}, batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=batch['label']
    ).mean()
    metric_updates = state.metrics.single_from_model_output(
        logits=logits,
        labels=batch['label'],
        loss=loss
    )
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state

def main():
    cnn = CNN()
    print(cnn.tabulate(jax.random.PRNGKey(0), jnp.ones((1, *image_shape))))
    learning_rate = 0.01
    momentum = 0.9
    init_rng = jax.random.PRNGKey(0)
    state = create_train_state(cnn, init_rng, learning_rate, momentum, image_shape=image_shape)
    del init_rng  # Must not be used anymore.

    metrics_history = {'train_loss': [],
                    'train_accuracy': [],
                    'test_loss': [],
                    'test_accuracy': []}

    for epoch in range(NUM_EPOCHS):
        if USE_STREAMING:
            ds_train.set_epoch(epoch)  # randomize the batches
        pbar = tqdm(ds_train)
        for batch in pbar:
            pbar.set_description('Training...')
            state = train_step(state, batch) # get updated train state (which contains the updated parameters)
            pbar.set_description('Computing metrics...')
            state = compute_metrics(state=state, batch=batch) # aggregate batch metrics


        for metric, value in state.metrics.compute().items(): # compute metrics
            metrics_history[f'train_{metric}'].append(value) # record metrics
        state = state.replace(metrics=state.metrics.empty()) # reset train_metrics for next training epoch

        # Compute metrics on the test set after each training epoch
        test_state = state
        for test_batch in ds_train:
            test_state = compute_metrics(state=test_state, batch=test_batch)

        for metric, value in test_state.metrics.compute().items():
            metrics_history[f'test_{metric}'].append(value)

        print(f"train epoch: {epoch+1}, "
            f"loss: {metrics_history['train_loss'][-1]}, "
            f"accuracy: {metrics_history['train_accuracy'][-1] * 100}")
        print(f"test epoch: {epoch+1}, "
            f"loss: {metrics_history['test_loss'][-1]}, "
            f"accuracy: {metrics_history['test_accuracy'][-1] * 100}")

    import matplotlib.pyplot as plt  # Visualization

    # Plot loss and accuracy in subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.set_title('Loss')
    ax2.set_title('Accuracy')
    for dataset in ('train','test'):
        ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
        ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
    ax1.legend()
    ax2.legend()
    plt.savefig(f'oneshot_loss_accuracy_{RUN_TIMESTAMP}.png')


if __name__ == "__main__":
    main()