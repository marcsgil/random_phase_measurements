import slmcontrol
from common.utils import sample_haar_vectors, generate_amplitude_and_phase_hologram, extraction_linear_combination, linear_transformation
import h5py
import numpy as np
from cameras.ImagingSourceNew import ImagingSourceCamera
from cameras.Ximea import XimeaCamera
from acquisition.phase_screens import fourier_phase_screen
from functools import partial
import itertools
import os
import shutil
from acquisition import calibration
from slm_camera_calibration import CalibrationResult
from analysis import diagnose_linear_combinations
from scipy.linalg import polar, inv


def _prepare_no_phase(mode, slm_shape, extraction, n):
    mode = extraction(mode, n)
    phase = np.zeros_like(mode)
    return generate_amplitude_and_phase_hologram(mode, phase, 192, -3, 19, slm_shape=slm_shape)

def _prepare_phase(mode, phases, indices, slm_shape, extraction, unitary, n):
    sigma_idx, phase_idx, coeff_idx = indices[n]
    mode = extraction(mode, coeff_idx)
    phase = linear_transformation(np.flip(phases[sigma_idx, phase_idx], axis=1), unitary)
    return generate_amplitude_and_phase_hologram(mode, phase, 192, -3, 19, slm_shape=slm_shape)

def _measure_no_phase(images_direct, images_fourier, camera_direct, camera_fourier, roi_fourier, n):
    for _ in range(5):
        # For autoexposure to settle
        camera_fourier.capture(roi=roi_fourier)

    images_direct[n] = camera_direct.capture()
    images_fourier[n] = camera_fourier.capture(roi=roi_fourier)

def _measure_phase(images_phase_fourier, camera_fourier, indices, roi_fourier, n):
    sigma_idx, phase_idx, coeff_idx = indices[n]
    for _ in range(5):
        # For autoexposure to settle
        camera_fourier.capture(roi=roi_fourier)
    images_phase_fourier[sigma_idx, phase_idx, coeff_idx] = camera_fourier.capture(roi=roi_fourier)

def main(slm, camera_direct, camera_fourier, roi_fourier, modes, phases, folder, extraction, unitary, NUM_SAMPLES = 8, MAX_MODES = None, opening_mode="a"):
    slm_shape = (slm.height, slm.width // 2)

    NUM_SIGMAS, NUM_PHASES = phases.shape[:2]

    if isinstance(modes, tuple):
        NUM_MODES = len(modes[0])
    else:
        NUM_MODES = len(modes)

    if MAX_MODES is not None:
        NUM_MODES = min(MAX_MODES, NUM_MODES)

    image_direct = camera_direct.capture()
    image_fourier = camera_fourier.capture(roi=roi_fourier)

    with h5py.File(os.path.join(folder, "data.h5"), opening_mode) as f:
        images_direct = f.create_dataset("images_direct", (NUM_MODES, *image_direct.shape), image_direct.dtype)
        images_fourier = f.create_dataset("images_fourier", (NUM_MODES, *image_fourier.shape), image_fourier.dtype)
        images_phase_fourier = f.create_dataset("images_phase_fourier", (NUM_SIGMAS, NUM_PHASES, NUM_MODES, *image_fourier.shape), image_fourier.dtype)

        print(10 * "-" + "Measuring without phase" + 10 * "-")
        prepare = partial(_prepare_no_phase, modes, slm_shape, extraction)
        measure = partial(_measure_no_phase, images_direct, images_fourier, camera_direct, camera_fourier, roi_fourier)
        slmcontrol.prepare_and_measure(prepare, measure, slm, 0.3, NUM_MODES)

        print(10 * "-" + "Measuring with phase" + 10 * "-")
        indices = list(itertools.product(range(len(sigmas)), range(NUM_PHASES), range(NUM_MODES)))

        prepare = partial(_prepare_phase, modes, phases, indices, slm_shape, extraction, unitary)
        measure = partial(_measure_phase, images_phase_fourier, camera_fourier, indices, roi_fourier)
        slmcontrol.prepare_and_measure(prepare, measure, slm, 0.3, len(indices))

    diagnose_linear_combinations.main(folder, NUM_SAMPLES)
    

def fixed_order_basis(xs, ys, w, order):
    return np.array([slmcontrol.hg(xs, ys, w=w, m=order-n, n=n) for n in range(order+1)])

def up_to_order_basis(xs, ys, w, order):
    return np.concatenate([fixed_order_basis(xs, ys, w, o) for o in range(order+1)])


print("Running calibration...")
calibration.main()

calib_res_direct = CalibrationResult.load(os.path.join("calibration_data", "calibration_direct.h5"))
u, _ = polar(calib_res_direct.transform.matrix)
inv_u = inv(u)

SIZE = 256
NUM_MODES = 8
NUM_PHASES = 4
NUM_SIGMAS = 2

amplitude = np.pi
sigmas = np.linspace(0.02, 0.04, NUM_SIGMAS)
folder = "results/test"
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
camera_direct.set_exposure(100)

camera_fourier = XimeaCamera()

with h5py.File("calibration_data/calibration_fourier.h5") as f:
    roi_fourier = np.asarray(f["roi"])

camera_fourier.camera.enable_aeag()
camera_fourier.camera.set_aeag_roi_width(roi_fourier[3] - roi_fourier[2])
camera_fourier.camera.set_aeag_roi_height(roi_fourier[1] - roi_fourier[0])
camera_fourier.camera.set_aeag_roi_offset_x(roi_fourier[2])
camera_fourier.camera.set_aeag_roi_offset_y(roi_fourier[0])
camera_fourier.camera.set_exp_priority(1.0)
camera_fourier.camera.set_aeag_level(4)

_xs = np.arange(SIZE) - SIZE // 2
_ys = np.arange(SIZE) - SIZE // 2
xs, ys = np.meshgrid(_xs, _ys)

print(f"Estimated Time: {(NUM_MODES * (1 + NUM_PHASES * NUM_SIGMAS) * 0.3 / 60)} min/main call")

for n in range(1, 6):
    print(f"Capturing order up to {n} \n")
    sub_folder = os.path.join(folder, f"up_to_order_{n}")
    os.makedirs(sub_folder, exist_ok=True)
    shutil.copytree("calibration_data", os.path.join(folder, "calibration_data"), dirs_exist_ok=True)
    basis = up_to_order_basis(xs, ys, 30, n)
    coefficients = sample_haar_vectors(NUM_MODES, len(basis))

    with h5py.File(os.path.join(sub_folder, "modes.h5"), opening_mode) as f:
        f["basis"] = basis
        f["coefficients"] = coefficients
    
    modes = (coefficients, basis)

    main(slm, camera_direct, camera_fourier, roi_fourier, modes, phases, sub_folder, extraction_linear_combination, u, NUM_SAMPLES = 8, MAX_MODES=None, opening_mode=opening_mode)

slm.close()
camera_direct.close()
camera_fourier.close()
