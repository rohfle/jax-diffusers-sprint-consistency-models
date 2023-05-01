from jax import numpy as jnp
import numpy as np
from PIL import Image
from PIL.Image import Resampling
import jax
from functools import partial


def make_grid(samples, n_samples, padding=2, pad_value=0.0):
    ndarray = samples.reshape((-1, *samples.shape[-3:]))[:n_samples]
    nrow = int(jnp.sqrt(ndarray.shape[0]))

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
    return im


@partial(jax.jit, static_argnums=1)
def ensure_channels(image, channels : int):
    # ensure channel axis
    image = image.reshape(*image.shape[:2], -1)
    shape = image.shape
    if shape[2] == 1 and channels != 1:
        # handle grayscale to rgb
        image = image.repeat(channels, axis=2)
    if shape[2] == 4 and channels == 3:
        # drop the alpha
        image = image[:, :, :channels]
    if shape[2] == 3 and channels == 4:
         # add an alpha channel
        image = jnp.concatenate((image, jnp.full((*shape[:2], 1), 255)), axis=2, dtype=image.dtype)
    return image


@partial(jax.jit, static_argnums=1)
def crop_resize(image, resolution):
    '''Resize and crop image to fill a square'''
    width, height = image.shape[:2]
    if width == resolution and height == resolution:
        return image
    # left, top, right, bottom
    if width != height:
        crop = jnp.minimum(width, height)
        left = (width - crop) // 2
        top = (height - crop) // 2
        image = jax.lax.dynamic_slice(image, (top, left, 0), (resolution, resolution, image.shape[-1]))
    return jax.image.resize(image, (resolution, resolution, image.shape[-1]), 'bicubic')


@partial(jax.jit, static_argnums=(1, 2, 3))
def normalize_images(images, channels, resolution, pad):
    output = []
    for im in images:
        im = ensure_channels(im, channels)
        im = crop_resize(im, resolution - 2 * pad)
        output += [im]

    stack = jnp.stack(output)
    stack = jnp.pad(stack, [(0,0), (pad,pad), (pad,pad), (0,0)])
    stack = stack * 2 / 255 - 1
    return stack
