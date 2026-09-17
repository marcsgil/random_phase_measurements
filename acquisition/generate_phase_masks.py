import argparse
from pathlib import Path

import h5py
import numpy as np
from jax import random

from acquisition.config import load_config, snapshot_config
from acquisition.phase_screens import fourier_phase_screen


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
    with h5py.File(result_directory / "phases.h5", "w") as file:
        file["phases"] = phases
        file["sigmas"] = sigmas
        file["amplitude"] = phase_config["amplitude"]
        file["grid_shape"] = (size, size)
        file["seed"] = phase_config["seed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate phase masks for an acquisition run.")
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    arguments = parser.parse_args()
    main(arguments.result_directory, arguments.config)
