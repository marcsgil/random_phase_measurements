import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import affine_transform


def find_order_directories(result_directory):
    return sorted(
        directory
        for directory in result_directory.iterdir()
        if directory.is_dir()
        and (directory / "data.h5").is_file()
        and (directory / "modes.h5").is_file()
    )


def prepare_order(order_directory, matrix, offset, output_shape):
    input_path = order_directory / "data.h5"
    output_path = order_directory / "prepared.h5"

    with h5py.File(input_path) as input_file, h5py.File(output_path, "w") as output_file:
        raw_images = input_file["images_phase_fourier"]
        prepared_images = output_file.create_dataset(
            "images_phase_fourier",
            shape=(*raw_images.shape[:3], *output_shape),
            dtype=np.float32,
            chunks=(1, 1, 1, *output_shape),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        prepared_images.attrs["axis_order"] = "sigma,phase,mode,y,x"

        for index in np.ndindex(raw_images.shape[:3]):
            prepared_images[index] = affine_transform(
                np.asarray(raw_images[index], dtype=np.float32),
                matrix,
                offset,
                output_shape=output_shape,
            )

    print(output_path)


def prepare(result_directory):
    calibration_path = (
        result_directory / "calibration_data" / "calibration_fourier.h5"
    )
    with h5py.File(calibration_path) as calibration_file:
        matrix = np.asarray(calibration_file["transform/matrix"])
        offset = np.asarray(calibration_file["transform/offset"])
        output_shape = tuple(int(x) for x in calibration_file["output_shape"][:])

    for order_directory in find_order_directories(result_directory):
        prepare_order(order_directory, matrix, offset, output_shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_directory", type=Path)
    args = parser.parse_args()
    prepare(args.result_directory)


if __name__ == "__main__":
    main()
