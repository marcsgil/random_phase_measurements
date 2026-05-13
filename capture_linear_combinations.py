import slmcontrol
from utils import sample_haar_vectors, generate_amplitude_and_phase_hologram, resize_and_center, fourier_transform, optimize_exposure
import h5py
import numpy as np
from cameras.ImagingSourceNew import ImagingSourceCamera
from cameras.Ximea import XimeaCamera
from scipy.linalg import polar
from phase_screens import fourier_phase_screen
from functools import partial
import itertools
import os
import matplotlib.pyplot as plt
from scipy.linalg import inv
from scipy.ndimage import affine_transform
import shutil
import calibrate_slm_camera


def _prepare_no_phase(mode, slm_shape, extraction, n):
    mode = extraction(mode, n)
    phase = np.zeros_like(mode)
    return generate_amplitude_and_phase_hologram(mode, phase, 192, -3, 19, slm_shape=slm_shape)


def _prepare_phase(mode, phases, unitary, indices, slm_shape, extraction, n):
    sigma_idx, phase_idx, coeff_idx = indices[n]
    mode = extraction(mode, coeff_idx)
    phase = phases[sigma_idx, phase_idx]
    return generate_amplitude_and_phase_hologram(mode, phase, 192, -3, 19, unitary @ np.diag([1, -1]), slm_shape=slm_shape)

def _measure_no_phase(images_direct, images_fourier, camera_direct, camera_fourier, roi_direct, roi_fourier, exposures, n):
    # if n == 0:
    #     print("\n Optimizing Exposures \n")
    #     exposures[0] = optimize_exposure(camera_fourier, 150, 100, 1000)
    #     exposures[1] = optimize_exposure(camera_fourier, 100, 15, 1000)
    camera_direct.set_exposure(exposures[0])
    camera_fourier.set_exposure(exposures[1])
    images_direct[n] = camera_direct.capture(roi=roi_direct)
    images_fourier[n] = camera_fourier.capture(roi=roi_fourier)

def _measure_phase(images_phase_fourier, camera_fourier, roi_fourier, indices, exposures, n):
    sigma_idx, phase_idx, coeff_idx = indices[n]
    # if phase_idx == 0 and coeff_idx == 0:
    #     print("\n Optimizing Exposures \n")
    #     exposures[2] = optimize_exposure(camera_fourier, 100, 15, 1000)
    camera_fourier.set_exposure(exposures[sigma_idx])
    images_phase_fourier[sigma_idx, phase_idx, coeff_idx] = camera_fourier.capture(roi=roi_fourier)
        

def main(slm, camera_direct, camera_fourier, modes, phases, exposures_no_phase, exposures_phase, folder, extraction, NUM_SAMPLES = 8, MAX_MODES = None, opening_mode="a"):
    slm_shape = (slm.height, slm.width // 2)

    with h5py.File(os.path.join("calibration", "calibration.h5")) as f:
        A_direct = f["A_direct"][:]
        t_direct = f["t_direct"][:]
        A_fourier = f["A_fourier"][:]
        t_fourier = f["t_fourier"][:]
        roi_direct = f["roi_direct"][:]
        roi_fourier = f["roi_fourier"][:]

    u, _ = polar(A_direct)

    NUM_SIGMAS, NUM_PHASES = phases.shape[:2]

    if isinstance(modes, tuple):
        NUM_MODES = len(modes[0])
    else:
        NUM_MODES = len(modes)

    if MAX_MODES is not None:
        NUM_MODES = min(MAX_MODES, NUM_MODES)

    image_direct = camera_direct.capture(roi=roi_direct)
    image_fourier = camera_fourier.capture(roi=roi_fourier)


    with h5py.File(os.path.join(folder, "data.h5"), opening_mode) as f:
        images_direct = f.create_dataset("images_direct", (NUM_MODES, *image_direct.shape), image_direct.dtype)
        images_fourier = f.create_dataset("images_fourier", (NUM_MODES, *image_fourier.shape), image_fourier.dtype)
        images_phase_fourier = f.create_dataset("images_phase_fourier", (NUM_SIGMAS, NUM_PHASES, NUM_MODES, *image_fourier.shape), image_fourier.dtype)

        print(10 * "-" + "Measuring without phase" + 10 * "-")
        prepare = partial(_prepare_no_phase, modes, slm_shape, extraction)
        measure = partial(_measure_no_phase, images_direct, images_fourier, camera_direct, camera_fourier, roi_direct, roi_fourier, exposures_no_phase)
        slmcontrol.prepare_and_measure(prepare, measure, slm, 0.3, NUM_MODES)

        print(10 * "-" + "Measuring with phase" + 10 * "-")
        indices = list(itertools.product(range(len(sigmas)), range(NUM_PHASES), range(NUM_MODES)))

        prepare = partial(_prepare_phase, modes, phases, u, indices, slm_shape, extraction)
        measure = partial(_measure_phase, images_phase_fourier, camera_fourier, roi_fourier, indices, exposures_phase)
        slmcontrol.prepare_and_measure(prepare, measure, slm, 0.3, len(indices))

    with h5py.File(os.path.join(folder, "data.h5")) as f:
        # Fix phase and sigma
        for n in range(min(NUM_SAMPLES, NUM_MODES)):
            mode = extraction(modes, n)
            phase = resize_and_center(phases[0, 0], mode.shape, 1)
            mode_fourier = fourier_transform(mode)
            mode_phase_fourier = fourier_transform(mode * np.exp(1j * phase))

            theo_mode_direct = affine_transform(mode, inv(A_direct), - inv(A_direct) @ t_direct)
            theo_mode_fourier = affine_transform(mode_fourier, inv(A_fourier), - inv(A_fourier) @ t_fourier)
            theo_mode_phase_fourier = affine_transform(mode_phase_fourier, inv(A_fourier), - inv(A_fourier) @ t_fourier)

            image_direct = f["images_direct"][n]
            image_fourier = f["images_fourier"][n]
            image_phase_fourier = f["images_phase_fourier"][0, 0, n]

            fig, axs = plt.subplots(2, 3, figsize=(10, 8))
            axs[0, 0].imshow(image_direct, cmap="hot", vmin=0, vmax=255)
            axs[0, 0].set_title("Direct (Experiment)")

            axs[0, 1].imshow(image_fourier, cmap="hot", vmin=0, vmax=255)
            axs[0, 1].set_title("Fourier (Experiment)")

            axs[0, 2].imshow(image_phase_fourier, cmap="hot", vmin=0, vmax=255)
            axs[0, 2].set_title("Phase Fourier (Experiment)")

            axs[1, 0].imshow(np.abs(theo_mode_direct)**2, cmap="hot")
            axs[1, 0].set_title("Direct (Theory)")

            axs[1, 1].imshow(np.abs(theo_mode_fourier)**2, cmap="hot")
            axs[1, 1].set_title("Fourier (Theory)")

            axs[1, 2].imshow(np.abs(theo_mode_phase_fourier)**2, cmap="hot")
            axs[1, 2].set_title("Phase Fourier (Theory)")

            plt.savefig(os.path.join(folder, f"mode_{n}.png"))

        # Fix Mode and phase
        for n in range(min(NUM_SAMPLES, len(sigmas))):
            mode = extraction(modes, 0)
            phase = resize_and_center(phases[n, 0], mode.shape, 1)
            mode_phase_fourier = fourier_transform(mode * np.exp(1j * phase))

            theo_mode_phase_fourier = affine_transform(mode_phase_fourier, inv(A_fourier), - inv(A_fourier) @ t_fourier)

            image_phase_fourier = f["images_phase_fourier"][n, 0, 0]

            fig, axs = plt.subplots(1, 3, figsize=(10, 4))

            axs[0].imshow(image_phase_fourier, cmap="hot", vmin=0, vmax=255)
            axs[0].set_title("Phase Fourier (Experiment)")

            axs[1].imshow(np.abs(theo_mode_phase_fourier)**2, cmap="hot")
            axs[1].set_title("Phase Fourier (Theory)")
            
            axs[2].imshow(phase, cmap="twilight")
            axs[2].set_title("Transformation Phase")

            plt.savefig(os.path.join(folder, f"sigma_{n}.png"))

        # Fix mode and sigma
        for n in range(min(NUM_SAMPLES, NUM_PHASES)):
            mode = extraction(modes, 0)
            phase = resize_and_center(phases[0, n], mode.shape, 1)
            mode_phase_fourier = fourier_transform(mode * np.exp(1j * phase))

            theo_mode_phase_fourier = affine_transform(mode_phase_fourier, inv(A_fourier), - inv(A_fourier) @ t_fourier)

            image_phase_fourier = f["images_phase_fourier"][0, n, 0]

            fig, axs = plt.subplots(1, 3, figsize=(10, 4))

            axs[0].imshow(image_phase_fourier, cmap="hot", vmin=0, vmax=255)
            axs[0].set_title("Phase Fourier (Experiment)")

            axs[1].imshow(np.abs(theo_mode_phase_fourier)**2, cmap="hot")
            axs[1].set_title("Phase Fourier (Theory)")
            
            axs[2].imshow(phase, cmap="twilight")
            axs[2].set_title("Transformation Phase")

            plt.savefig(os.path.join(folder, f"phase_{n}.png"))

def extraction_linear_combination(modes, idx):
    coefficients, basis = modes
    Cs = coefficients[idx].reshape(-1, 1, 1)
    return np.sum(Cs * basis, axis=0)

def fixed_order_basis(xs, ys, w, order):
    return np.array([slmcontrol.hg(xs, ys, w=w, m=order-n, n=n) for n in range(order+1)])

def up_to_order_basis(xs, ys, w, order):
    return np.concatenate([fixed_order_basis(xs, ys, w, o) for o in range(order+1)])


SIZE = 640
NUM_MODES = 10
NUM_PHASES = 10
NUM_SIGMAS = 5

amplitude = np.pi
sigmas = np.linspace(0.02, 0.04, NUM_SIGMAS)
folder = "results/controled_exposure"
last_folder = os.path.basename(os.path.normpath(folder))

if last_folder == "test":
    opening_mode = "w"
else:
    opening_mode = "a"

os.makedirs(folder, exist_ok=True)

phases_path = os.path.join(folder, "phases.h5")

if os.path.exists(phases_path):
    with h5py.File(phases_path) as f:
        phases = f["phases"][:]
else:
    with h5py.File(phases_path, "a") as f:
        phases = np.array([fourier_phase_screen(SIZE, SIZE, amplitude=amplitude, sigma=sigma, num_samples=NUM_PHASES) for sigma in sigmas])
        f["phases"] = phases
        f["sigmas"] = sigmas
        f["amplitude"] = amplitude

slm = slmcontrol.SLMDisplay(host="localhost")
camera_direct = ImagingSourceCamera()
camera_fourier = XimeaCamera()

_xs = np.arange(SIZE) - SIZE // 2
_ys = np.arange(SIZE) - SIZE // 2
xs, ys = np.meshgrid(_xs, _ys)

exposures_no_phase = [200, 80]
exposures_phase = np.linspace(80, 220, NUM_SIGMAS)

print(f"Estimated Time: {(NUM_MODES * (1 + NUM_PHASES * NUM_SIGMAS) * 0.3 / 60)} min/main call")

calibrate_slm_camera.main(slm, camera_direct, camera_fourier, SIZE=SIZE)

for n in range(1, 5):
    print(f"Capturing order up to {n} \n")
    sub_folder = os.path.join(folder, f"up_to_order_{n}")
    os.makedirs(sub_folder, exist_ok=True)
    shutil.copytree("calibration", os.path.join(folder, "calibration"), dirs_exist_ok=True)
    basis = up_to_order_basis(xs, ys, 30, n)
    coefficients = sample_haar_vectors(NUM_MODES, len(basis))

    with h5py.File(os.path.join(sub_folder, "modes.h5"), opening_mode) as f:
        f["basis"] = basis
        f["coefficients"] = coefficients
    
    modes = (coefficients, basis)

    main(slm, camera_direct, camera_fourier, modes, phases, exposures_no_phase, exposures_phase, sub_folder, extraction_linear_combination, NUM_SAMPLES = 8, MAX_MODES=None, opening_mode=opening_mode)

# with h5py.File("../turbulence_compensation/data/output.h5") as file:
#     data = file["fields"][:]

#     def extraction(modes, n): 
#         return resize_and_center(np.sqrt(data[n, 0]) * np.exp(1j * data[n, 1]), (SIZE, SIZE), 0.8)

#     main(slm, modes, phases, exposures, folder, extraction, NUM_SAMPLES = 8, MAX_MODES=2)

slm.close()
camera_direct.close()
camera_fourier.close()