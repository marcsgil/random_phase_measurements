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


def direct_transform(matrix, offset, source_shape, camera_shape):
    source_shape = np.asarray(source_shape)
    camera_shape = np.asarray(camera_shape)
    source_center = source_shape // 2
    camera_center = camera_shape // 2

    normalized_matrix = (
        np.diag(1 / camera_shape) @ matrix @ np.diag(source_shape)
    )
    normalized_offset = (
        matrix @ source_center + offset - camera_center
    ) / camera_shape

    direct_matrix = normalized_matrix.T
    direct_offset = source_center - direct_matrix @ camera_center
    direct_scale = abs(np.linalg.det(normalized_matrix)) * np.sqrt(
        np.prod(camera_shape) / np.prod(source_shape)
    )

    coordinates = np.indices(camera_shape, dtype=float).reshape(2, -1)
    coordinates -= camera_center[:, None]
    phase_ramp = np.exp(
        2j * np.pi * (normalized_offset @ coordinates)
    ).reshape(camera_shape)

    return direct_matrix, direct_offset, direct_scale, phase_ramp


def prepare_phases(
    output_file,
    phases,
    direct_matrix,
    direct_offset,
    phase_ramp,
    camera_shape,
):
    phase_factors = output_file.create_dataset(
        "phase_factors",
        shape=(*phases.shape[:2], *camera_shape),
        dtype=np.complex64,
        chunks=(1, 1, *camera_shape),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )
    phase_factors.attrs["axis_order"] = "sigma,phase,y,x"

    for index in np.ndindex(phases.shape[:2]):
        transformed = affine_transform(
            np.exp(1j * phases[index]).astype(np.complex64),
            direct_matrix,
            direct_offset,
            output_shape=camera_shape,
        )
        phase_factors[index] = transformed * phase_ramp


def prepare_basis(
    output_file,
    order_directory,
    direct_matrix,
    direct_offset,
    direct_scale,
    camera_shape,
):
    with h5py.File(order_directory / "modes.h5") as modes_file:
        basis = modes_file["basis"]
        dtype = (
            np.complex64
            if np.issubdtype(basis.dtype, np.complexfloating)
            else np.float32
        )
        prepared_basis = output_file.create_dataset(
            f"orders/{order_directory.name}/basis",
            shape=(basis.shape[0], *camera_shape),
            dtype=dtype,
            chunks=(1, *camera_shape),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        prepared_basis.attrs["axis_order"] = "mode,y,x"

        for mode_index in range(basis.shape[0]):
            prepared_basis[mode_index] = direct_scale * affine_transform(
                np.asarray(basis[mode_index], dtype=dtype),
                direct_matrix,
                direct_offset,
                output_shape=camera_shape,
            )


def prepare(result_directory):
    order_directories = find_order_directories(result_directory)

    with h5py.File(order_directories[0] / "data.h5") as data_file:
        camera_shape = data_file["images_phase_fourier"].shape[-2:]

    with h5py.File(
        result_directory / "calibration_data" / "calibration_fourier.h5"
    ) as calibration_file:
        matrix = np.asarray(calibration_file["transform/matrix"])
        offset = np.asarray(calibration_file["transform/offset"])
        source_shape = tuple(int(x) for x in calibration_file["output_shape"][:])

    direct_matrix, direct_offset, direct_scale, phase_ramp = direct_transform(
        matrix,
        offset,
        source_shape,
        camera_shape,
    )

    with h5py.File(result_directory / "phases.h5") as phases_file, h5py.File(
        result_directory / "prepared.h5", "w"
    ) as output_file:
        prepare_phases(
            output_file,
            phases_file["phases"],
            direct_matrix,
            direct_offset,
            phase_ramp,
            camera_shape,
        )

        for order_directory in order_directories:
            prepare_basis(
                output_file,
                order_directory,
                direct_matrix,
                direct_offset,
                direct_scale,
                camera_shape,
            )

    print(result_directory / "prepared.h5")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_directory", type=Path)
    args = parser.parse_args()
    prepare(args.result_directory)


if __name__ == "__main__":
    main()
