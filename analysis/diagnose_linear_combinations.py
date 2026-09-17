from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import affine_transform
from slm_camera_calibration import CalibrationResult

from common.utils import extraction_linear_combination, fourier_transform, remove_background


def camera_grid(field, calibration_result, camera_shape):
    matrix = calibration_result.transform.matrix
    inverse_matrix = np.linalg.inv(matrix)
    inverse_offset = -inverse_matrix @ calibration_result.transform.offset
    return affine_transform(field, inverse_matrix, inverse_offset, output_shape=camera_shape)


def main(folder, num_samples=4, extraction=extraction_linear_combination):
    folder = Path(folder)
    result_directory = folder.parent
    calibration_direct = CalibrationResult.load(result_directory / "calibration_data" / "calibration_direct.h5")
    calibration_fourier = CalibrationResult.load(result_directory / "calibration_data" / "calibration_fourier.h5")

    with h5py.File(folder / "modes.h5") as file:
        coefficients = np.asarray(file["coefficients"])
        basis = np.asarray(file["basis"])
    modes = (coefficients, basis)

    with h5py.File(result_directory / "phases.h5") as file:
        phases = np.asarray(file["phases"])

    with h5py.File(result_directory / "background.h5") as file:
        background_direct = np.asarray(file["images_direct"])
        background_fourier = np.asarray(file["images_fourier"])
        background_phase_fourier = np.asarray(file["images_phase_fourier"])

    num_modes = coefficients.shape[0]
    num_sigmas, num_phases = phases.shape[:2]

    with h5py.File(folder / "data.h5") as file:
        fourier_shape = file["images_fourier"].shape[-2:]

        for index in range(min(num_samples, num_modes)):
            mode = extraction(modes, index)
            phase = phases[0, 0]
            mode_fourier = fourier_transform(mode)
            mode_phase_fourier = fourier_transform(mode * np.exp(1j * phase))

            image_direct = affine_transform(
                remove_background(file["images_direct"][index], background_direct),
                calibration_direct.transform.matrix,
                calibration_direct.transform.offset,
                output_shape=calibration_direct.output_shape,
            )
            image_fourier = remove_background(file["images_fourier"][index], background_fourier)
            image_phase_fourier = remove_background(file["images_phase_fourier"][0, 0, index], background_phase_fourier[0])
            theory_direct = np.abs(mode) ** 2
            theory_fourier = np.abs(camera_grid(mode_fourier, calibration_fourier, fourier_shape)) ** 2
            theory_phase_fourier = np.abs(camera_grid(mode_phase_fourier, calibration_fourier, fourier_shape)) ** 2

            figure, axes = plt.subplots(2, 3, figsize=(10, 8))
            axes[0, 0].imshow(image_direct, cmap="hot", vmin=0, vmax=255)
            axes[0, 0].set_title("Direct (Experiment)")
            axes[0, 1].imshow(image_fourier, cmap="hot", vmin=0, vmax=255)
            axes[0, 1].set_title("Fourier (Experiment)")
            axes[0, 2].imshow(image_phase_fourier, cmap="hot", vmin=0, vmax=255)
            axes[0, 2].set_title("Phase Fourier (Experiment)")
            axes[1, 0].imshow(theory_direct, cmap="hot")
            axes[1, 0].set_title("Direct (Theory)")
            axes[1, 1].imshow(theory_fourier, cmap="hot")
            axes[1, 1].set_title("Fourier (Theory)")
            axes[1, 2].imshow(theory_phase_fourier, cmap="hot")
            axes[1, 2].set_title("Phase Fourier (Theory)")
            plt.savefig(folder / f"mode_{index}.png")
            plt.close()

        for index in range(min(num_samples, num_sigmas)):
            mode = extraction(modes, 0)
            phase = phases[index, 0]
            mode_phase_fourier = fourier_transform(mode * np.exp(1j * phase))
            image_phase_fourier = remove_background(file["images_phase_fourier"][index, 0, 0], background_phase_fourier[index])
            theory_phase_fourier = np.abs(camera_grid(mode_phase_fourier, calibration_fourier, fourier_shape)) ** 2

            figure, axes = plt.subplots(1, 3, figsize=(10, 4))
            axes[0].imshow(image_phase_fourier, cmap="hot", vmin=0, vmax=255)
            axes[0].set_title("Phase Fourier (Experiment)")
            axes[1].imshow(theory_phase_fourier, cmap="hot")
            axes[1].set_title("Phase Fourier (Theory)")
            axes[2].imshow(phase, cmap="twilight")
            axes[2].set_title("Transformation Phase")
            plt.savefig(folder / f"sigma_{index}.png")
            plt.close()

        for index in range(min(num_samples, num_phases)):
            mode = extraction(modes, 0)
            phase = phases[0, index]
            mode_phase_fourier = fourier_transform(mode * np.exp(1j * phase))
            image_phase_fourier = remove_background(file["images_phase_fourier"][0, index, 0], background_phase_fourier[0])
            theory_phase_fourier = np.abs(camera_grid(mode_phase_fourier, calibration_fourier, fourier_shape)) ** 2

            figure, axes = plt.subplots(1, 3, figsize=(10, 4))
            axes[0].imshow(image_phase_fourier, cmap="hot", vmin=0, vmax=255)
            axes[0].set_title("Phase Fourier (Experiment)")
            axes[1].imshow(theory_phase_fourier, cmap="hot")
            axes[1].set_title("Phase Fourier (Theory)")
            axes[2].imshow(phase, cmap="twilight")
            axes[2].set_title("Transformation Phase")
            plt.savefig(folder / f"phase_{index}.png")
            plt.close()


if __name__ == "__main__":
    for order in range(1, 6):
        main(f"results/test/up_to_order_{order}", 4)
