"""
Lets talk about this in plain text first


unet2d

sample_size : width / height of image
in / out channels: 1=grayscale, 3=rgb, 4=rgba eg

down block types

up block types

block out channels

layers per block

attention_head_dim
cross_attention_dim
dropout

"""

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

from flax import linen as nn  # Linen API

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


class UNet2D(nn.Module):
    def setup(self):
        self.encoder = nn.Sequential(
            nn.Conv(features=64, kernel_size=(3, 3), padding='SAME'),
            nn.relu,
            nn.Conv(features=64, kernel_size=(3, 3), padding='SAME'),
            nn.relu,
            nn.max_pool(window_shape=(2, 2), strides=(2, 2), padding='SAME')
        )
        self.middle = nn.Sequential(
            nn.Conv(features=128, kernel_size=(3, 3), padding='SAME'),
            nn.relu,
            nn.Conv(features=128, kernel_size=(3, 3), padding='SAME'),
            nn.relu
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose(features=64, kernel_size=(2, 2), strides=(2, 2), padding='SAME'),
            nn.relu,
            nn.Conv(features=64, kernel_size=(3, 3), padding='SAME'),
            nn.relu,
            nn.Conv(features=64, kernel_size=(3, 3), padding='SAME'),
            nn.relu
        )
        self.final = nn.Conv(features=1, kernel_size=(1, 1), padding='SAME')

    def __call__(self, x):
        enc = self.encoder(x)
        mid = self.middle(enc)
        dec = self.decoder(mid)
        out = self.final(dec)
        return out
