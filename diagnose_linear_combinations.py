import matplotlib.pyplot as plt
import h5py
import os
import numpy as np
from utils import fourier_transform, resize_and_center, extraction_linear_combination
from scipy.ndimage import affine_transform
from slm_camera_calibration import CalibrationResult
from pathlib import Path
from utils import remove_background


def main(folder, NUM_SAMPLES, extraction = extraction_linear_combination):
    parent_folder = Path(folder).parent
    calib_res_direct = CalibrationResult.load(os.path.join(parent_folder, "calibration_data", "calibration_direct.h5"))
    calib_res_fourier = CalibrationResult.load(os.path.join(parent_folder, "calibration_data", "calibration_fourier.h5"))

    with h5py.File(os.path.join(folder, "modes.h5")) as f:
        coefficients = np.asarray(f["coefficients"])
        basis = np.asarray(f["basis"])
        modes = (coefficients, basis)

    NUM_MODES = coefficients.shape[0]

    with h5py.File(os.path.join(parent_folder, "phases.h5")) as f:
        phases = np.asarray(f["phases"])

    NUM_SIGMAS, NUM_PHASES = phases.shape[:2]

    with h5py.File(os.path.join(folder, "data.h5")) as f:
        # Fix phase and sigma
        for n in range(min(NUM_SAMPLES, NUM_MODES)):
            mode = extraction(modes, n)
            phase = phases[0, 0]
            mode_fourier = fourier_transform(mode)
            mode_phase_fourier = fourier_transform(mode * np.exp(1j * phase))

            image_direct = affine_transform(remove_background(f["images_direct"][n], 5), calib_res_direct.transform.matrix, calib_res_direct.transform.offset, output_shape=calib_res_direct.output_shape)
            image_fourier = affine_transform(remove_background(f["images_fourier"][n], 5), calib_res_fourier.transform.matrix, calib_res_fourier.transform.offset, output_shape=calib_res_fourier.output_shape)
            image_phase_fourier = affine_transform(remove_background(f["images_phase_fourier"][0, 0, n], 5), calib_res_fourier.transform.matrix, calib_res_fourier.transform.offset, output_shape=calib_res_fourier.output_shape)


            fig, axs = plt.subplots(2, 3, figsize=(10, 8))
            axs[0, 0].imshow(image_direct, cmap="hot", vmin=0, vmax=255)
            axs[0, 0].set_title("Direct (Experiment)")

            axs[0, 1].imshow(resize_and_center(image_fourier,(32, 32), 1), cmap="hot", vmin=0, vmax=255)
            axs[0, 1].set_title("Fourier (Experiment)")

            axs[0, 2].imshow(resize_and_center(image_phase_fourier,(32, 32), 1), cmap="hot", vmin=0, vmax=255)
            axs[0, 2].set_title("Phase Fourier (Experiment)")

            axs[1, 0].imshow(np.abs(mode)**2, cmap="hot")
            axs[1, 0].set_title("Direct (Theory)")

            axs[1, 1].imshow(resize_and_center(np.abs(mode_fourier)**2,(32, 32), 1), cmap="hot")
            axs[1, 1].set_title("Fourier (Theory)")

            axs[1, 2].imshow(resize_and_center(np.abs(mode_phase_fourier)**2,(32, 32), 1), cmap="hot")
            axs[1, 2].set_title("Phase Fourier (Theory)")

            plt.savefig(os.path.join(folder, f"mode_{n}.png"))
            plt.close()

        # Fix Mode and phase
        for n in range(min(NUM_SAMPLES, NUM_SIGMAS)):
            mode = extraction(modes, 0)
            phase = phases[n, 0]
            mode_phase_fourier = fourier_transform(mode * np.exp(1j * phase))
            image_phase_fourier = affine_transform(remove_background(f["images_phase_fourier"][n, 0, 0], 5), calib_res_fourier.transform.matrix, calib_res_fourier.transform.offset, output_shape=calib_res_fourier.output_shape)

            fig, axs = plt.subplots(1, 3, figsize=(10, 4))

            axs[0].imshow(resize_and_center(image_phase_fourier, (32, 32), 1), cmap="hot", vmin=0, vmax=255)
            axs[0].set_title("Phase Fourier (Experiment)")

            axs[1].imshow(np.abs(resize_and_center(mode_phase_fourier,(32, 32), 1))**2, cmap="hot")
            axs[1].set_title("Phase Fourier (Theory)")
            
            axs[2].imshow(phase, cmap="twilight")
            axs[2].set_title("Transformation Phase")

            plt.savefig(os.path.join(folder, f"sigma_{n}.png"))
            plt.close()

        # Fix mode and sigma
        for n in range(min(NUM_SAMPLES, NUM_PHASES)):
            mode = extraction(modes, 0)
            phase = phases[0, n]
            mode_phase_fourier = fourier_transform(mode * np.exp(1j * phase))
            image_phase_fourier = affine_transform(remove_background(f["images_phase_fourier"][0, n, 0], 5), calib_res_fourier.transform.matrix, calib_res_fourier.transform.offset, output_shape=calib_res_fourier.output_shape)

            fig, axs = plt.subplots(1, 3, figsize=(10, 4))

            axs[0].imshow(resize_and_center(image_phase_fourier, (32, 32), 1), cmap="hot", vmin=0, vmax=255)
            axs[0].set_title("Phase Fourier (Experiment)")

            axs[1].imshow(resize_and_center(np.abs(mode_phase_fourier)**2, (32, 32), 1), cmap="hot")
            axs[1].set_title("Phase Fourier (Theory)")
            
            axs[2].imshow(phase, cmap="twilight")
            axs[2].set_title("Transformation Phase")

            plt.savefig(os.path.join(folder, f"phase_{n}.png"))
            plt.close()

if __name__ == "__main__":
    for n in range(1, 6):
        folder = f"results/test/up_to_order_{n}"
        NUM_SAMPLES = 4
    
        main(folder, NUM_SAMPLES)