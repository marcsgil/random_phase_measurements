import jax.numpy as jnp
import numpy as np


def crop_center(img, size):
    """
    Crop an image around its center to a square size.

    Parameters
    ----------
    img : array-like
        Input image to crop.
    size : int
        Size of the output square (height and width).

    Returns
    -------
    array-like
        Cropped image.
    """
    h, w = img.shape[-2], img.shape[-1]
    start_h = (h - size) // 2
    start_w = (w - size) // 2
    return img[..., start_h : start_h + size, start_w : start_w + size]


def set_phase_reference(data, posY=0.5, posX=0.5):
    """
    Set a phase reference by defining the center of the beam to have a phase of -pi
    """
    Nx = data.shape[-1]
    Ny = data.shape[-2]
    reference = data[..., int(Ny * posY), int(Nx * posX)]

    # Compute the regularization term
    angles = jnp.mod(data - reference[..., None, None], 2 * jnp.pi) - jnp.pi

    return angles

