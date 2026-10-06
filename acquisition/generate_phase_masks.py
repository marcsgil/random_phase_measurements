import argparse
from pathlib import Path
from typing import Callable
import tomllib
import h5py
import jax.numpy as jnp
import numpy as np
from jax import Array, random


def gaussian_spectrum(qx, qy, rc, amplitude=2 * np.pi**2):
    return amplitude * jnp.exp(-(rc**2 * (qx**2 + qy**2)) / 4) * rc**2 / 4 / jnp.pi


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
    dqx = qxs[1] - qxs[0]
    dqy = qys[1] - qys[0]
    qxs, qys = jnp.meshgrid(qxs, qys, sparse=True)
    spectrum_value = spectrum(qxs, qys, **kwargs) * dqx * dqy
    shape = (ny, nx) if num_samples is None else (num_samples, ny, nx)
    random_numbers = random.normal(key, shape=shape, dtype=jnp.complex64)
    return jnp.angle(
        jnp.exp(
            1j
            * jnp.real(
                jnp.fft.ifft2(random_numbers * jnp.sqrt(spectrum_value), norm="forward")
            )
        )
    )


def main(result_directory):
    result_directory = Path(result_directory)
    if not result_directory.exists():
        raise ValueError(
            f"Directory {str(result_directory)} does not exist. The calibration must be run beforehand, which creates the directory."
        )
    config_path = result_directory / "config.toml"
    with config_path.open("rb") as file:
        config = tomllib.load(file)

    size = config["grid"]["size"]
    phase_config = config["phases"]
    correlation_lengths = (
        np.asarray(phase_config["correlation_lengths"]) * config["modes"]["waist"]
    )
    keys = random.split(random.key(phase_config["seed"]), len(correlation_lengths))
    phases = np.asarray(
        [
            fourier_phase_screen(
                size,
                size,
                rc=rc,
                num_samples=phase_config["num_samples"],
                key=key,
            )
            for rc, key in zip(correlation_lengths, keys)
        ]
    )
    with h5py.File(result_directory / "phases.h5", "a") as file:
        file["phases"] = phases
        file["correlation_lengths"] = correlation_lengths
        file["grid_shape"] = (size, size)
        file["seed"] = phase_config["seed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate phase masks for an acquisition run."
    )
    parser.add_argument("result_directory", type=Path)
    arguments = parser.parse_args()
    main(arguments.result_directory)
