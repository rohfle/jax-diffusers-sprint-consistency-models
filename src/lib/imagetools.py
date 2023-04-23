from jax import numpy as jnp
import numpy as np
from PIL import Image


def make_grid(samples, n_samples, padding=2, pad_value=0.0):
    ndarray = samples.reshape((-1, *samples.shape[2:]))[:n_samples]
    nrow = int(jnp.sqrt(ndarray.shape[0]))

    if not (isinstance(ndarray, jnp.ndarray) or
            (isinstance(ndarray, list) and
            all(isinstance(t, jnp.ndarray) for t in ndarray))):
        raise TypeError("array_like of tensors expected, got {}".format(
            type(ndarray)))

    ndarray = jnp.asarray(ndarray)

    if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
        ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

    # make the mini-batch of images into a grid
    nmaps = ndarray.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(jnp.ceil(float(nmaps) / xmaps))
    height, width = int(ndarray.shape[1] + padding), int(ndarray.shape[2] +
                                                        padding)
    num_channels = ndarray.shape[3]
    grid = jnp.full(
        (height * ymaps + padding, width * xmaps + padding, num_channels),
        pad_value).astype(jnp.float32)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid = grid.at[y * height + padding:(y + 1) * height,
                            x * width + padding:(x + 1) * width].set(ndarray[k])
            k = k + 1
    return grid


def save_image(grid, path):
    ndarr = jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8)
    ndarr = np.array(ndarr)
    im = Image.fromarray(ndarr)
    im.save(path)
