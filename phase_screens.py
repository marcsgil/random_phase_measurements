import jax.numpy as jnp
from jax import random, Array
from typing import Callable


def gaussian_spectrum(qx, qy, amplitude, sigma):
    return amplitude * jnp.exp(-(qx**2 + qy**2) / 2 / sigma**2) / 2 / jnp.pi / sigma**2


def fourier_phase_screen(
    Ny: int,
    Nx: int,
    spectrum: Callable = gaussian_spectrum,
    dx: float = 1,
    dy: float = 1,
    key: Array = random.key(42),
    **kwargs,
) -> Array:
    qxs = jnp.fft.fftfreq(Nx, d=dx / 2 / jnp.pi)
    qys = jnp.fft.fftfreq(Ny, d=dy / 2 / jnp.pi)
    Qxs, Qys = jnp.meshgrid(qxs, qys, sparse=True)

    dqx = qxs[1] - qxs[0]
    dqy = qys[1] - qys[0]

    spectrum_value = spectrum(Qxs, Qys, **kwargs) * dqx * dqy
    random_numbers = random.normal(key, shape=(Ny, Nx), dtype=jnp.complex64)

    return (
        jnp.mod(
            jnp.real(
                jnp.fft.ifft2(random_numbers * jnp.sqrt(spectrum_value), norm="forward")
            ),
            2 * jnp.pi,
        )
        - jnp.pi
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N = 512
    amplitude = 2**20
    sigma = 1 / 2 / jnp.pi

    screen = fourier_phase_screen(N, N, amplitude=amplitude, sigma=sigma)

    im = plt.imshow(screen, cmap="twilight")
    plt.colorbar(im)

    plt.show()
