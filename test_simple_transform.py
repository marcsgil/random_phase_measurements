import slmcontrol
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
from cameras.ImagingSourceNew import ImagingSourceCamera
from cameras.Ximea import XimeaCamera
from scipy.linalg import inv, norm
from utils import generate_amplitude_and_phase_hologram
from phase_screens import fourier_phase_screen
from augmented_gs import augmented_gs, fidelity
import jax
from utils import crop_center

def remove_background(img, bg):
    return np.where(img > bg, img - bg, 0)

def capture_images(mode, phase, camera_direct, camera_fourier, roi_direct, roi_fourier, slm, bg_direct, bg_fourier):
    holo1 = generate_amplitude_and_phase_hologram(mode, np.zeros_like(mode), 192, -3, 19)
    slm.updateArray(holo1)
    image_direct = camera_direct.capture(roi=roi_direct)
    image_fourier = camera_fourier.capture(roi=roi_fourier)

    holo2 = generate_amplitude_and_phase_hologram(mode, phase, 192, -3, 19)
    slm.updateArray(holo2)
    image_phase_fourier = camera_fourier.capture(roi=roi_fourier)

    image_direct = remove_background(image_direct, bg_direct)
    image_fourier = remove_background(image_fourier, bg_fourier)
    image_phase_fourier = remove_background(image_phase_fourier, bg_fourier)

    return image_direct, image_fourier, image_phase_fourier

def fourier_transform(mode):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(mode)))

def inverse_fourier_transform(mode):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(mode)))

slm = slmcontrol.SLMDisplay(host="localhost")
Ny, Nx = slm.height, slm.width // 2
_xs = np.arange(Nx) - Nx // 2
_ys = np.arange(Ny) - Ny // 2
xs, ys = np.meshgrid(_xs, _ys)

camera_direct = ImagingSourceCamera()
camera_fourier = XimeaCamera()

with h5py.File("calibration/calibration.h5") as f:
    A_direct = f["A_direct"][:]
    t_direct = f["t_direct"][:]
    A_fourier = f["A_fourier"][:]
    t_fourier = f["t_fourier"][:]
    roi_direct = f["roi_direct"][:]
    roi_fourier = f["roi_fourier"][:]

amplitude = 2**25
sigma = 0.5 / 2 / np.pi

phase_mask = fourier_phase_screen(slm.height, slm.width // 2, amplitude=amplitude, sigma=sigma)
phase_transformation = np.exp(1j * phase_mask)

mode = slmcontrol.lg(xs, ys, p=0, l=0, w=30)
ft_mode = fourier_transform(mode)

camera_direct.set_exposure(200)
camera_fourier.set_exposure(400)

image_direct, image_fourier, _ = capture_images(mode, phase_transformation, camera_direct, camera_fourier, roi_direct, roi_fourier, slm, 2, 5)

conjugate_image_fourier = np.abs(fourier_transform(np.sqrt(image_fourier)))**2

M = A_fourier.T

# Centers
input_center = (np.array([*mode.shape]) - 1) / 2
output_center = (np.array([*image_fourier.shape]) - 1) / 2

# Offset to align centers
offset = input_center - M @ output_center

attempt_transform = affine_transform(mode, M, offset, output_shape=image_fourier.shape)

fig, axs = plt.subplots(1, 3)
axs[0].imshow(np.abs(inverse_fourier_transform(np.sqrt(image_fourier))))
axs[1].imshow(np.abs(attempt_transform)**2)
axs[2].imshow(np.abs(mode)**2)
plt.show()