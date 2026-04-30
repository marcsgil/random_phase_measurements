from statistics import mode

import numpy as np
import slmcontrol
from cameras.ImagingSourceNew import ImagingSourceCamera
from cameras.Ximea import XimeaCamera
from functools import partial
import matplotlib.pyplot as plt
import itertools
import h5py
from utils import resize_and_center
from utils import generate_amplitude_and_phase_hologram
from scipy.linalg import det


def fit_centroid(image, threshold = 0.5):
    """Fit a 2D Gaussian to an image and return the (x, y) centroid."""
    max_val = np.max(image)
    threshold_value = threshold * max_val

    # Create mask for pixels above threshold
    mask = image >= threshold_value

    # Calculate total weight (sum of intensities above threshold)
    total_weight = np.sum(image[mask])

    if total_weight <= 0:
        return np.array([np.nan, np.nan])

    # Get coordinates
    y_coords, x_coords = np.indices(image.shape)

    # Calculate weighted centroids
    x_centroid = np.sum(x_coords[mask] * image[mask]) / total_weight
    y_centroid = np.sum(y_coords[mask] * image[mask]) / total_weight

    return np.array([y_centroid, x_centroid])


def fit_affine(input_points, target_points):
    """Find affine transform mapping SLM coordinates to camera coordinates.

    Solves the least squares problem for the system:
        [y_target]   [a b] [y_input]   [ty]
        [x_target] = [c d] [x_input] + [tx]

    Parameters
    ----------
    input_points : ndarray, shape (N, 2)
        Known (x, y) coordinates in SLM space.
    target_points : ndarray, shape (N, 2)
        Observed (x, y) centroids in camera space.

    Returns
    -------
    A : ndarray, shape (2, 2)
        Linear part of the affine transform.
    t : ndarray, shape (2,)
        Translation vector.
    residuals : ndarray, shape (N,)
        Per-point Euclidean error in camera pixels.
    """
    N = len(input_points)

    # build the design matrix: each point contributes two rows
    # [y_input, x_input, 1, 0      ,       0, 0]   [a ]   [y_target]
    # [0      , 0      , 0, y_input, x_input, 1] * [b ] = [x_target]
    #                                              [ty]
    #                                              [c ]
    #                                              [d ]
    #                                              [tx]
    M = np.zeros((2 * N, 6))
    b = np.zeros(2 * N)

    for i, ((yi, xi), (yt, xt)) in enumerate(zip(input_points, target_points)):
        M[2*i]     = [yi, xi, 1, 0,  0,  0]
        M[2*i + 1] = [0,  0,  0, yi, xi, 1]
        b[2*i]     = yt
        b[2*i + 1] = xt

    params, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
    a, b_, ty, c, d, tx = params

    A = np.array([[a, b_], [c, d]])
    t = np.array([ty, tx])

    predicted = (A @ input_points.T).T + t
    residuals = np.linalg.norm(predicted - target_points, axis=1)

    return A, t, residuals


def direct_prepare(xs, ys, two_pi_modulation, xperiod, yperiod, centers, sigma, n):
    y0, x0 = centers[n]
    mode = np.exp(-((xs - x0)**2 + (ys - y0)**2) / (2 * sigma**2))

    return generate_amplitude_and_phase_hologram(mode, np.zeros_like(mode), two_pi_modulation, xperiod, yperiod)

def fourier_prepare(xs, ys, two_pi_modulation, xperiod, yperiod, centers, sigma, n):
    ky, kx = centers[n]
    Ny, Nx = xs.shape
    mode = np.exp(-((xs - Nx //2)**2 + (ys - Ny //2)**2) / (2 * sigma**2) + 2j * np.pi * ((kx - Nx //2) * xs / Nx + (ky - Ny //2)*ys / Ny))

    return generate_amplitude_and_phase_hologram(mode, np.zeros_like(mode), two_pi_modulation, xperiod, yperiod)

def _measure(dataset, camera, roi, n):
    dataset[n] = camera.capture(roi=roi)

def calibrate(_prepare, camera, roi, xs, ys, centers, sigma):
    prepare = partial(_prepare, xs, ys, 192, -3, 19, centers, sigma) 
    test_img = camera.capture(roi=roi)
    images = np.empty((n**2, *test_img.shape), test_img.dtype)
    measure = partial(_measure, images, camera, roi)

    slmcontrol.prepare_and_measure(prepare, measure, slm, 0.3, n**2)
    print("Finished acquisition. Fitting centroids...")

    # --- fit centroids ---
    cam_points = np.array([fit_centroid(images[i]) for i in range(n**2)])  # (N, 2) in camera (x, y)
    slm_points = np.array(centers)               # (N, 2) in SLM (x, y)

    # center both
    k_slm = slm_points - slm_points.mean(axis=0)
    p_cam = cam_points - cam_points.mean(axis=0)

    # solve p_cam ≈ B @ k_slm
    B, _, _, _ = np.linalg.lstsq(k_slm, p_cam, rcond=None)  # shape (2,2)
    B = B.T  # so p_cam = B @ k_slm

    print("Scaling:", np.sqrt(np.abs(det(B))))

    print("Fitted camera points")

    # --- fit affine transform ---
    A, t, residuals = fit_affine(slm_points, cam_points)

    print("Fitted affine transform from SLM to camera coordinates.")

    print("Affine matrix A:")
    print(A)
    print("Translation t:", t)
    print(f"Residuals — mean: {residuals.mean():.3f} px, max: {residuals.max():.3f} px")

    return A, t, images

def get_most_frequent(arr):
    vals, counts = np.unique(arr, return_counts=True)
    return vals[counts.argmax()]

def determine_center(slm, camera, xs, ys, sigma):
    mode = np.exp(-((xs - xs.mean())**2 + (ys - ys.mean())**2) / (2 * sigma**2))
    holo = generate_amplitude_and_phase_hologram(mode, np.zeros_like(mode), 192, -3, 19)
    slm.updateArray(holo)
    centroid = fit_centroid(camera.capture())
    return int(centroid[0]), int(centroid[1])


if __name__ == "__main__":
    SIZE = 512
    n = 3
    XMAX = 50
    FMAX = 15

    slm = slmcontrol.SLMDisplay(host="localhost")

    Ny = slm.height
    Nx = slm.width // 2

    _xs = np.arange(Nx)
    _ys = np.arange(Ny)
    xs, ys = np.meshgrid(_xs, _ys)

    camera_direct = ImagingSourceCamera()
    camera_direct.set_exposure(100)

    camera_fourier = XimeaCamera()
    camera_fourier.set_exposure(100)

    center_direct = determine_center(slm, camera_direct, xs, ys, 50)
    roi_direct = np.array(
        [center_direct[0] - SIZE // 2, 
         center_direct[0] + SIZE // 2, 
         center_direct[1] - SIZE // 2, 
         center_direct[1] + SIZE // 2])
    
    center_fourier = determine_center(slm, camera_fourier, xs, ys, 50)
    roi_fourier = np.array(
        [center_fourier[0] - SIZE // 2, 
         center_fourier[0] + SIZE // 2, 
         center_fourier[1] - SIZE // 2, 
         center_fourier[1] + SIZE // 2])
    

    x0s = np.linspace(-XMAX, XMAX, n) + Nx // 2
    y0s = np.linspace(-XMAX, XMAX, n) + Ny // 2

    centers_direct = np.array(list(itertools.product(y0s, x0s)))

    fxs = np.linspace(-FMAX, FMAX, n) + Nx // 2
    fys = np.linspace(-FMAX, FMAX, n) + Ny // 2

    centers_fourier = np.array(list(itertools.product(fys, fxs)))

    target_shape = (slm.height, slm.width // 2)

    print(10 * "-" + "Direct Calibration" + 10 * "-")
    A_direct, t_direct, images_direct = calibrate(direct_prepare, camera_direct, roi_direct, xs, ys, centers_direct, 15)

    print(10 * "-" + "fourier Calibration" + 10 * "-")
    A_fourier, t_fourier, images_fourier = calibrate(fourier_prepare, camera_fourier, roi_fourier, xs, ys, centers_fourier, 50)

    plt.clf()
    plt.imshow(np.mean(images_direct, axis=0))
    plt.savefig("calibration/calibration_images_direct.png")

    plt.clf()
    plt.imshow(np.mean(images_fourier, axis=0))
    plt.savefig("calibration/calibration_images_fourier.png")

    with h5py.File("calibration/calibration.h5", "w") as f:
        f["A_direct"] = A_direct
        f["t_direct"] = t_direct
        f["A_fourier"] = A_fourier
        f["t_fourier"] = t_fourier
        f["roi_direct"] = roi_direct
        f["roi_fourier"] = roi_fourier

    camera_direct.close()
    slm.close()