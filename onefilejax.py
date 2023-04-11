import jax.numpy as jnp
from jax import jit, Array, random
from flax import linen as nn
import jax
import optax
import pickle
from copy import deepcopy

from lib.models import UNet, ConsistencyUNet

# Hyperparameters
learning_rate = 3e-5
num_epochs = 100
batch_size = 512
epsilon = 0.002
N = 150

# Initialize the models
unet = UNet(in_channels=1, out_channels=1, block_out_channels=(16, 32, 64, 64), norm_num_groups=8)
consistency_model = ConsistencyUNet(epsilon, unet)
ema_model = deepcopy(consistency_model)
ema_model.load_state_dict(consistency_model.state_dict())

# Define the loss function - MSELoss in pytorch
def loss_fn(params, inputs, targets):
    logits = consistency_model.apply(params, inputs)
    preds = jnp.softmax(logits)
    return jnp.mean((preds - targets) ** 2)

# Initialize the optimizer
optimizer = optax.adamw(learning_rate=learning_rate)
params = {'w': jnp.ones((num_weights,))}
opt_state = optimizer.init(params)

# Prepare the data
train_ds, test_ds = get_datasets()
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# Training loop
for epoch in range(num_epochs):
    # Iterate over the batches of training data
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Compute the gradients and update the main model parameters
        params = optimizer.target
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params, inputs, targets)
        optimizer = optimizer.apply_gradient(grads)
        # after_step?

    # Validation loop
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        # Evaluate the performance of the model on the validation data
        params = optimizer.target
        loss = loss_fn(params, inputs, targets)

    # Save the model
    jax.tree_map(lambda x: x.block_until_ready(), params)
    with open(f'model_{epoch}.pkl', 'wb') as f:
        pickle.dump(params, f)