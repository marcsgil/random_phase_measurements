import jax.numpy as jnp
import numpy as np
from scipy.ndimage import affine_transform
from slmcontrol import generate_hologram
from numpy.linalg import qr


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
        return np.stack(channels, axis=0)
    
def fourier_transform(mode):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(mode)))

def inverse_fourier_transform(mode):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(mode)))

def linear_transformation(input, A, output_shape=None):
    if not output_shape:
        output_shape = input.shape
    input_center = (np.array([*input.shape])) / 2
    output_center = (np.array([*output_shape])) / 2

    # Offset to align centers
    offset = input_center -  A @ output_center

    return affine_transform(input, A, offset, output_shape=output_shape)
    
def generate_amplitude_and_phase_hologram(mode, phase, two_pi_modulation, xperiod, yperiod, unitary=None, slm_shape=None):
    if slm_shape is not None:
        mode = resize_and_center(mode, slm_shape, 1)
        phase = resize_and_center(phase, slm_shape, 1)

    phase_transformation = np.exp(1j * phase)

    if unitary is not None:
        phase_transformation = linear_transformation(phase_transformation, unitary)

    ys, xs = np.indices(mode.shape)

    phase_total = -np.angle(phase_transformation) - 2*np.pi*(xs/xperiod + ys/yperiod)
    phase_wrapped = np.mod(phase_total, 2*np.pi)

    holo1 = np.uint8(np.round(
        phase_wrapped * (two_pi_modulation / (2*np.pi))
    ))

    holo2 = generate_hologram(mode, two_pi_modulation, xperiod, yperiod)

    return np.concatenate([holo1, holo2], axis=1)

def complex_randn(*shape):
    """
    Generate an array of complex numbers with random real and imaginary parts.

    Parameters:
        shape (tuple): The shape of the output array.

    Returns:
        (ArrayLike): An array of complex numbers with the specified shape.
    """
    return (np.random.randn(*shape).astype(np.float32)
            + 1j * np.random.randn(*shape).astype(np.float32))

def sample_haar_vectors(n_samples: int, dim: int):
    """
    Generate random Haar vectors.

    Args:
        n_samples (int): Number of Haar vectors to generate.
        dim (int): Dimension of the Haar vectors.

    Returns:
        ArrayLike: Array of random Haar vectors.

    References:
        https://pennylane.ai/qml/demos/tutorial_haar_measure/
    """

    Zs = complex_randn(n_samples, dim, dim)
    result = np.empty((n_samples, dim), dtype=np.complex64)

    for n, Z in enumerate(Zs):
        Q, R = qr(Z)
        lambd = np.diag(R)
        result[n, :] = (Q @ np.diag(lambd) / np.abs(lambd))[0, :]

    return result

def mean_capture(camera, n, roi=None):
    first_image = camera.capture(roi=roi)
    buffer = np.empty((n, *first_image.shape), dtype=first_image.dtype)
    buffer[0] = first_image
    for i in range(1, n):
        buffer[i] = camera.capture(roi=roi)
    return np.mean(buffer, axis=0).astype(first_image.dtype)

def optimize_exposure(camera, initial_exposure, min_exposure, max_exposure, alpha=1.05, ncycles=20, nmean=10, threshold=50, nbins=20, roi=None):
    exposure = np.clip(initial_exposure, min_exposure, max_exposure)

    for i in range(ncycles):
        camera.set_exposure(exposure)
        x = mean_capture(camera, nmean, roi=roi)
        x = x[x>threshold]
        counts, _ = np.histogram(x, bins=nbins, range=(threshold, 255))

        signal = 1 - counts[-1] / max(counts[-2], 1)

        exposure = np.clip(exposure * (alpha ** signal), min_exposure, max_exposure)

    return np.clip(exposure * 0.8, min_exposure, max_exposure)

def remove_background(img, bg):
    return np.where(img > bg, img - bg, 0)