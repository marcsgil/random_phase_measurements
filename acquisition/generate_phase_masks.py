import argparse
from pathlib import Path
from typing import Callable

import h5py
import jax.numpy as jnp
import numpy as np
from jax import Array, random

from acquisition.config import load_config, snapshot_config


def gaussian_spectrum(qx, qy, amplitude, sigma):
    return amplitude * jnp.exp(-(qx**2 + qy**2) / 2 / sigma**2) / 2 / jnp.pi / sigma**2


def fourier_phase_screen(
    ny: int,
    nx: int,
    spectrum: Callable = gaussian_spectrum,
    dx: float = 1,
    dy: float = 1,
    key: Array = random.key(42),
    num_samples=None,
    **kwargs,
) -> Array:
    qxs = jnp.fft.fftfreq(nx, d=dx / 2 / jnp.pi)
    qys = jnp.fft.fftfreq(ny, d=dy / 2 / jnp.pi)
    qxs, qys = jnp.meshgrid(qxs, qys, sparse=True)
    spectrum_value = spectrum(qxs, qys, **kwargs) * (qxs[1] - qxs[0]) * (qys[1] - qys[0])
    shape = (ny, nx) if num_samples is None else (num_samples, ny, nx)
    random_numbers = random.normal(key, shape=shape, dtype=jnp.complex64)
    return jnp.mod(jnp.real(jnp.fft.ifft2(random_numbers * jnp.sqrt(spectrum_value), norm="forward")), 2 * jnp.pi) - jnp.pi


def main(result_directory, config_path):
    result_directory = Path(result_directory)
    result_directory.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_path)
    config = load_config(config_path)
    snapshot_config(config_path, result_directory)

    size = config["grid"]["size"]
    phase_config = config["phases"]
    sigmas = np.asarray(phase_config["sigmas"])
    keys = random.split(random.key(phase_config["seed"]), len(sigmas))
    phases = np.asarray(
        [
            fourier_phase_screen(
                size,
                size,
                amplitude=phase_config["amplitude"],
                sigma=sigma,
                num_samples=phase_config["num_samples"],
                key=key,
            )
            for sigma, key in zip(sigmas, keys)
        ]
    )
    with h5py.File(result_directory / "phases.h5", "a") as file:
        file["phases"] = phases
        file["sigmas"] = sigmas
        file["amplitude"] = phase_config["amplitude"]
        file["grid_shape"] = (size, size)
        file["seed"] = phase_config["seed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate phase masks for an acquisition run."
    )
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    arguments = parser.parse_args()
    main(arguments.result_directory, arguments.config)
