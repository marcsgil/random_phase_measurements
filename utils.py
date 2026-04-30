import jax.numpy as jnp
import numpy as np
import numpy as np
from scipy.ndimage import affine_transform
from slmcontrol import generate_hologram


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


def resize_and_center(img, target_shape, scale, order=1, cval=0):
    """
    Resize an image by a scale factor and place it centered in a target shape,
    using a single affine transformation.

    Parameters
    ----------
    img : np.ndarray
        Input image (H, W) or (H, W, C)
    target_shape : tuple
        ցանկ output shape (H_t, W_t)
    scale : float
        Scaling factor (>1 enlarges, <1 shrinks)
    order : int
        Interpolation order (default 1 = bilinear)
    cval : float
        Constant value for padding (default 0)

    Returns
    -------
    np.ndarray
        Transformed image of shape (H_t, W_t) or (H_t, W_t, C)
    """

    input_shape = np.array(img.shape[:2])
    target_shape = np.array(target_shape)

    # Inverse scaling (because affine_transform maps output -> input)
    A = np.eye(2) / scale

    # Centers
    input_center = (input_shape - 1) / 2
    output_center = (target_shape - 1) / 2

    # Offset to align centers
    offset = input_center - A @ output_center

    # Handle grayscale vs multi-channel
    if img.ndim == 2:
        return affine_transform(
            img,
            A,
            offset=offset,
            output_shape=tuple(target_shape),
            order=order,
            mode='constant',
            cval=cval
        )
    else:
        channels = [
            affine_transform(
                img[..., c],
                A,
                offset=offset,
                output_shape=tuple(target_shape),
                order=order,
                mode='constant',
                cval=cval
            )
            for c in range(img.shape[2])
        ]
        return np.stack(channels, axis=-1)
    
def generate_amplitude_and_phase_hologram(mode, phase, two_pi_modulation, xperiod, yperiod):
    # xs, ys = np.indices(mode.shape)
    # phase_total = phase + 2*np.pi*(xs/xperiod + ys/yperiod)
    # phase_wrapped = np.mod(phase_total, 2*np.pi)

    # holo1 = np.uint8(np.round(
    #     phase_wrapped * (two_pi_modulation / (2*np.pi))
    # ))

    holo1 = generate_hologram(np.exp(1j * np.flip(phase)), two_pi_modulation, xperiod, yperiod)
    holo2 = generate_hologram(mode, two_pi_modulation, xperiod, yperiod)

    return np.concatenate([holo1, holo2], axis=1)