import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

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


BATCH_SIZE = 32
NUM_EPOCHS = 1 # TODO: change to 100 or something
NUM_WORKERS = 1 # TODO: change to number of cpu cores
IMAGE_DTYPE = jnp.bfloat16
IMAGE_NUM_CHANNELS = 1
IMAGE_SIZE = 32

torch.set_num_threads(4)
torch.set_printoptions(precision=4, linewidth=140, sci_mode=False)
torch.manual_seed(1)

# convert from PIL image format, pad, -1 to 1 range
def transform_dataset_images(batch):
    for idx, orig in enumerate(batch['image']):
        image = np.array(orig, dtype=float)
        image = np.array(image / 255, dtype=IMAGE_DTYPE)
        image = np.pad(image, 2) * 2 - 1
        # this will standardize grayscale images as a WxHx1 array
        batch['image'][idx] = image.reshape(32, 32, -1)
    return batch


def collate_dataset(batch):
    images = []
    labels = []
    for item in batch:
        images += [item['image']]
        labels += [item['label']]
    return {
        'image': jnp.array(images),
        'label': jnp.array(labels),
    }


dataset = load_dataset('fashion_mnist').with_transform(transform_dataset_images)

dl_train = DataLoader(
    dataset['train'],
    batch_size=BATCH_SIZE,
    collate_fn=collate_dataset,
    num_workers=NUM_WORKERS
)

dl_valid = DataLoader(
    dataset['test'],
    batch_size=BATCH_SIZE,
    collate_fn=collate_dataset,
    num_workers=NUM_WORKERS
)


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

def create_train_state(module, rng, learning_rate, momentum):
    """Creates an initial `TrainState`."""
    params = module.init(rng, jnp.ones([1, 32, 32, 1]))['params'] # initialize parameters by passing a template image
    tx = optax.sgd(learning_rate, momentum)
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=tx,
        metrics=Metrics.empty())

@jax.jit
def train_step(state, batch):
    """Train for a single step."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']).mean()
        return loss
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state

@jax.jit
def compute_metrics(*, state, batch):
    logits = state.apply_fn({'params': state.params}, batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']).mean()
    metric_updates = state.metrics.single_from_model_output(
        logits=logits, labels=batch['label'], loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


cnn = CNN()
print(cnn.tabulate(jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 1))))
learning_rate = 0.01
momentum = 0.9
init_rng = jax.random.PRNGKey(0)
state = create_train_state(cnn, init_rng, learning_rate, momentum)
del init_rng  # Must not be used anymore.


metrics_history = {'train_loss': [],
                   'train_accuracy': [],
                   'test_loss': [],
                   'test_accuracy': []}

for epoch in range(NUM_EPOCHS):
    for batch_idx, batch in enumerate(dl_train):
        print("BATCH", batch_idx, "/", len(batch))
        state = train_step(state, batch) # get updated train state (which contains the updated parameters)
        state = compute_metrics(state=state, batch=batch) # aggregate batch metrics

    for metric, value in state.metrics.compute().items(): # compute metrics
        metrics_history[f'train_{metric}'].append(value) # record metrics
    state = state.replace(metrics=state.metrics.empty()) # reset train_metrics for next training epoch

    # Compute metrics on the test set after each training epoch
    test_state = state
    for test_batch in dl_train.as_numpy_iterator():
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
plt.show()
plt.clf()