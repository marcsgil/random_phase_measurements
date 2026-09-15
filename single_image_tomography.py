from juliacall import Main as jl
jl.seval("using QuantumMeasurements")
import h5py
import os
from utils import load_data, extraction_linear_combination, fourier_transform, resize_and_center
import numpy as np
from pathlib import Path
from slm_camera_calibration import CalibrationResult
import matplotlib.pyplot as plt

folder = "results/test/up_to_order_1"
parent_folder = Path(folder).parent

with h5py.File(os.path.join(parent_folder, "phases.h5")) as f:
    phases = np.asarray(f["phases"])

with h5py.File(os.path.join(folder, "modes.h5")) as f:
	coefficients = np.asarray(f["coefficients"])
	basis = np.asarray(f["basis"])
	N = np.sqrt(np.sum(np.abs(basis)**2, axis=(1, 2), keepdims=True))
	basis = basis / N
	modes = (coefficients, basis)

calib_res_direct = CalibrationResult.load(os.path.join(parent_folder, "calibration_data", "calibration_direct.h5"))
calib_res_fourier = CalibrationResult.load(os.path.join(parent_folder, "calibration_data", "calibration_fourier.h5"))

method = jl.MaximumLikelihood()

i = 0
j = 0
phase = phases[i, j]
phase_fourier_basis = fourier_transform(basis * np.exp(1j * phase.reshape(1, *phase.shape)))
itr = np.conj(phase_fourier_basis.reshape(phase_fourier_basis.shape[0], -1)).T
measurement_matrix = jl.assemble_measurement_matrix([x for x in itr])

k = 0

image_phase_fourier = load_data(f"{folder}/data.h5", "images_phase_fourier", background=2, calibration_result=calib_res_fourier, index=(i, j, k))
experimental_outcomes = np.asarray(
      image_phase_fourier,
      dtype=np.float64,
  ).ravel(order="C")

mode_phase_fourier = extraction_linear_combination((coefficients, phase_fourier_basis), k)

theo_outcomes = np.asarray(measurement_matrix) @ np.asarray(jl.vectorization(coefficients[k]))
theo_image_phase_fourier = np.reshape(theo_outcomes, mode_phase_fourier.shape)

fig, axs = plt.subplots(1, 2)

axs[0].imshow(resize_and_center(image_phase_fourier, (32, 32)))
axs[1].imshow(resize_and_center(theo_image_phase_fourier, (32, 32)))

os.makedirs("plots", exist_ok=True)
plt.savefig("plots/temp.png")

rho = jl.estimate_state(experimental_outcomes, measurement_matrix, method)[0]

print(rho)
print(coefficients[k])

print(jl.fidelity(rho, coefficients[k]))
