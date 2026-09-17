import numpy as np
from scipy.ndimage import affine_transform
from slmcontrol import generate_hologram
from numpy.linalg import qr
import h5py
from typing import Optional, Union, Tuple
from numpy.typing import ArrayLike


def resize_and_center(img, target_shape, scale=1, order=1, cval=0):
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
            mode="constant",
            cval=cval,
        )
    else:
        channels = [
            affine_transform(
                img[..., c],
                A,
                offset=offset,
                output_shape=tuple(target_shape),
                order=order,
                mode="constant",
                cval=cval,
            )
            for c in range(img.shape[2])
        ]
        return np.stack(channels, axis=0)


def fourier_transform(mode):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(mode), norm="ortho"))


def inverse_fourier_transform(mode):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(mode), norm="ortho"))


def linear_transformation(input, A, output_shape=None):
    if not output_shape:
        output_shape = input.shape
    input_center = (np.array([*input.shape])) / 2
    output_center = (np.array([*output_shape])) / 2

    # Offset to align centers
    offset = input_center - A @ output_center

    return affine_transform(input, A, offset, output_shape=output_shape)


def generate_amplitude_and_phase_hologram(
    mode, phase, two_pi_modulation, xperiod, yperiod, unitary=None, slm_shape=None
):
    if slm_shape is not None:
        mode = resize_and_center(mode, slm_shape, 1)
        phase = resize_and_center(phase, slm_shape, 1)

    phase_transformation = np.exp(1j * phase)

    if unitary is not None:
        phase_transformation = linear_transformation(phase_transformation, unitary)

    ys, xs = np.indices(mode.shape)

    phase_total = -np.angle(phase_transformation) - 2 * np.pi * (
        xs / xperiod + ys / yperiod
    )
    phase_wrapped = np.mod(phase_total, 2 * np.pi)

    holo1 = np.uint8(np.round(phase_wrapped * (two_pi_modulation / (2 * np.pi))))

    holo2 = generate_hologram(mode, two_pi_modulation, xperiod, yperiod)

    return np.concatenate([holo1, holo2], axis=1)


def complex_randn(
    shape: Tuple[int, ...], seed: Optional[Union[int, np.random.Generator]] = None
) -> ArrayLike:
    """
    Generate an array of complex numbers with random real and imaginary parts.

    Parameters:
        shape (tuple): The shape of the output array.
        seed (int or Generator, optional): Seed or Generator instance for reproducibility.

    Returns:
        (ArrayLike): An array of complex numbers with the specified shape.
    """
    # 1. Initialize the modern isolated random generator
    rng = np.random.default_rng(seed)

    # 2. Sample real and imaginary parts using standard_normal
    # (Since shape is passed as a tuple, we don't need * unpacking for standard_normal)
    real_part = rng.standard_normal(shape, dtype=np.float32)
    imag_part = rng.standard_normal(shape, dtype=np.float32)

    return real_part + 1j * imag_part


def sample_haar_vectors(
    n_samples: int, dim: int, seed: Optional[Union[int, np.random.Generator]] = None
):
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

    Zs = complex_randn((n_samples, dim, dim), seed)
    result = np.empty((n_samples, dim), dtype=np.complex64)

    for n, Z in enumerate(Zs):
        Q, R = qr(Z)
        lambd = np.diag(R)
        result[n, :] = (Q @ np.diag(lambd) / np.abs(lambd))[0, :]

    return result


def set_phase_reference(data, posY=0.5, posX=0.5):
    """
    Set a phase reference by defining the center of the beam to have a phase of -pi
    """
    Nx = data.shape[-1]
    Ny = data.shape[-2]
    reference = data[..., int(Ny * posY), int(Nx * posX)]

    # Compute the regularization term
    angles = np.mod(data - reference[..., None, None], 2 * np.pi) - np.pi

    return angles


def extraction_linear_combination(modes, idx):
    coefficients, basis = modes
    Cs = coefficients[idx].reshape(-1, 1, 1)
    return np.sum(Cs * basis, axis=0)


def remove_background(img, bg):
    return np.where(img > bg, img - bg, 0)


def load_data(path, key, background=None, calibration_result=None, index=None):
    with h5py.File(path) as f:
        if index is None:
            raw_image = np.asarray(f[key])
        else:
            raw_image = np.asarray(f[key][index])  # type: ignore

        if background is not None:
            raw_image = remove_background(raw_image, background)

        if calibration_result is None:
            return raw_image
        else:
            return affine_transform(
                raw_image,
                calibration_result.transform.matrix,
                calibration_result.transform.offset,
                output_shape=calibration_result.output_shape,
            )
