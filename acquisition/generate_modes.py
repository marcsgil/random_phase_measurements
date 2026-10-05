import argparse
from pathlib import Path
import h5py
import numpy as np
import slmcontrol
from common.utils import sample_haar_vectors
import tomllib


def fixed_order_basis(xs, ys, waist, order):
    return np.array([slmcontrol.hg(xs, ys, w=waist, m=order - n, n=n) for n in range(order + 1)])


def up_to_order_basis(xs, ys, waist, order):
    return np.concatenate([fixed_order_basis(xs, ys, waist, current_order) for current_order in range(order + 1)])


def main(result_directory):
    result_directory = Path(result_directory)
    if not result_directory.exists():
        raise ValueError(f"Directory {str(result_directory)} does not exist. The calibration must be run beforehand, which creates the directory.")

    config_path = result_directory / "config.toml"
    with config_path.open("rb") as file:
        config = tomllib.load(file)
    size = config["grid"]["size"]

    mode_config = config["modes"]
    coordinates = np.arange(size) - size // 2
    xs, ys = np.meshgrid(coordinates, coordinates)
    generator = np.random.default_rng(mode_config["seed"])
    for order in mode_config["orders"]:
        order_directory = result_directory / f"up_to_order_{order}"
        order_directory.mkdir(exist_ok=True)

        basis = up_to_order_basis(xs, ys, mode_config["waist"], order)
        coefficients = sample_haar_vectors(mode_config["num_combinations"], len(basis), generator)
        with h5py.File(order_directory / "modes.h5", "w") as file:
            file["basis"] = basis
            file["coefficients"] = coefficients
            file["grid_shape"] = (size, size)
            file["seed"] = mode_config["seed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mode bases and coefficients for an acquisition run.")
    parser.add_argument("result_directory", type=Path)
    arguments = parser.parse_args()
    main(arguments.result_directory)
