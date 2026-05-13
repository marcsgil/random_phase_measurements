from augmented_gs import augmented_gs, fidelity
import matplotlib.pyplot as plt
import h5py
from os.path import join
from scipy.ndimage import affine_transform
from utils import crop_center, fourier_transform, remove_background, set_phase_reference
import jax.numpy as jnp
import jax
import tqdm
from scipy.linalg import inv
import numpy as np

def get_normalized_amplitudes(image):
    amplitude = jnp.sqrt(image.astype(jnp.float32))
    return amplitude / image.sum()

def prepare_data_for_gs(image_direct, image_fourier, image_phase_fourier, A_direct, t_direct, A_fourier, t_fourier):
    amplitude_direct = get_normalized_amplitudes(affine_transform(image_direct, A_direct, t_direct))
    amplitude_fourier = get_normalized_amplitudes(affine_transform(image_fourier, A_fourier, t_fourier))
    amplitude_phase_fourier = get_normalized_amplitudes(affine_transform(image_phase_fourier, A_fourier, t_fourier))

    return amplitude_direct, jnp.fft.fftshift(amplitude_fourier), jnp.fft.fftshift(amplitude_phase_fourier)

def get_matching_slm2camera(mode, phase_transformation, A_direct, t_direct, A_fourier, t_fourier):
    mode_fourier = fourier_transform(mode)
    mode_phase_fourier = fourier_transform(mode * phase_transformation)

    return (
        affine_transform(mode, inv(A_direct), - inv(A_direct) @ t_direct),
        affine_transform(mode_fourier, inv(A_fourier), - inv(A_fourier) @ t_fourier),
        affine_transform(mode_phase_fourier, inv(A_fourier), - inv(A_fourier) @ t_fourier)
    )

def match_phase(phase, A, t):
    psi = jnp.exp(1j * phase)
    return jnp.angle(affine_transform(psi, A, t))

def match_phases_for_plot(phase_theory, phase_predicted, phase_transformation, A_direct, t_direct):
    return (
        match_phase(phase_theory, inv(A_direct), - inv(A_direct) @ t_direct),
        match_phase(phase_predicted, inv(A_direct), - inv(A_direct) @ t_direct),
        match_phase(phase_transformation, inv(A_direct), - inv(A_direct) @ t_direct)
    )

mode_idx = 0
phase_idx = 0
sigma_idx = 0

order = 4

base_path = "results/controled_exposure/"

with h5py.File(join(base_path, "calibration/calibration.h5")) as f:
    A_direct = f["A_direct"][:]
    t_direct = f["t_direct"][:]
    A_fourier = f["A_fourier"][:]
    t_fourier = f["t_fourier"][:]


for order in range(1, 5):
    print(f"Initializing order upt to {order}")


    with h5py.File(join(base_path, f"up_to_order_{order}/data.h5")) as f_data, h5py.File(join(base_path, f"up_to_order_{order}/modes.h5")) as f_modes, h5py.File(join(base_path, "phases.h5")) as f_phases:
        basis = f_modes["basis"][:]
        coefficients = f_modes["coefficients"][:]
        
        nsigmas, nphases, nmodes = f_data["images_phase_fourier"].shape[:3]

        fidelities = np.empty((nsigmas, nphases, nmodes))
        counter = tqdm.tqdm(total=nsigmas*nphases*nmodes)

        for sigma_idx in range(nsigmas):
            for phase_idx in range(nphases):
                phase = f_phases["phases"][sigma_idx, phase_idx]
                phase_transformation = jnp.exp(1j * phase)

                for mode_idx in range(nmodes):
                    image_direct = remove_background(f_data["images_direct"][mode_idx], 2)
                    image_fourier = remove_background(f_data["images_fourier"][mode_idx], 5)
                    image_phase_fourier = remove_background(f_data["images_phase_fourier"][sigma_idx, phase_idx, mode_idx], 5)

                    mode = jnp.sum(coefficients[mode_idx].reshape(-1, 1, 1) * basis, axis=0)

                    initial_phases = jax.random.uniform(jax.random.key(0), (10, *image_direct.shape))

                    data_for_gs = prepare_data_for_gs(image_direct, image_fourier, image_phase_fourier, A_direct, t_direct, A_fourier, t_fourier)

                    predicted_phase = augmented_gs(
                        *data_for_gs,
                        phase_transformation,
                        initial_phases,
                        100,
                    )

                    predicted_mode = data_for_gs[0] * jnp.exp(1j * predicted_phase)

                    fidelities[sigma_idx, phase_idx, mode_idx] = fidelity(predicted_mode, mode)

                    counter.update()

    with h5py.File(join(base_path, "fidelities_gs.h5"), "a") as f:
        f[f"up_to_order_{order}"] = fidelities
        print(fidelities.mean(axis=(1, 2)))

mode_theory = get_matching_slm2camera(mode, phase_transformation, A_direct, t_direct, A_fourier, t_fourier)
phases_for_plot = match_phases_for_plot(jnp.angle(mode), predicted_phase, phase, A_direct, t_direct)


fid = fidelity(predicted_mode, mode)
amplitude_fid = fidelity(jnp.complex64(data_for_gs[0]), jnp.complex64(jnp.abs(mode)))

fig, axs = plt.subplots(3, 3, figsize=(10, 11))
fig.suptitle(f"Amplitude Fidelity: {amplitude_fid:.1%}, Fidelity: {fid:.1%}")
axs[0, 0].imshow(image_direct, cmap="hot")
axs[0, 0].set_title("Direct (Experiment)")

axs[0, 1].imshow(image_fourier, cmap="hot")
axs[0, 1].set_title("Fourier (Experiment)")

axs[0, 2].imshow(image_phase_fourier, cmap="hot")
axs[0, 2].set_title("Phase Fourier (Experiment)")

axs[1, 0].imshow(jnp.abs(mode_theory[0])**2, cmap="hot")
axs[1, 0].set_title("Direct (Theory)")

axs[1, 1].imshow(jnp.abs(mode_theory[1])**2, cmap="hot")
axs[1, 1].set_title("Fourier (Theory)")

axs[1, 2].imshow(jnp.abs(mode_theory[2])**2, cmap="hot")
axs[1, 2].set_title("Phase Fourier (Theory)")

axs[2, 0].imshow(set_phase_reference(phases_for_plot[0], posX = 0.4, posY = 0.4), cmap="twilight", vmin=-jnp.pi, vmax=jnp.pi)
axs[2, 0].set_title("Phase (Theory)")

axs[2, 1].imshow(set_phase_reference(phases_for_plot[1], posX = 0.4, posY = 0.4), cmap="twilight", vmin=-jnp.pi, vmax=jnp.pi)
axs[2, 1].set_title("Predicted Phase (Experiment)")

axs[2, 2].imshow(phases_for_plot[2], cmap="twilight", vmin=-jnp.pi, vmax=jnp.pi)
axs[2, 2].set_title("Transformation Phase")

for ax in axs.flatten():
    ax.axis("off")

plt.tight_layout()

plt.savefig("plots/test.png")