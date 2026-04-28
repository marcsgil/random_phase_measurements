import numpy as np
import slmcontrol
from cameras.ImagingSourceNew import ImagingSourceCamera
from cameras.Ximea import XimeaCamera
from functools import partial
import matplotlib.pyplot as plt
import itertools
import h5py


def gaussian2d(xy, x0, y0, sigma, amplitude, background):
    xs, ys = xy
    return (amplitude * np.exp(-((xs - x0)**2 + (ys - y0)**2) / (2 * sigma**2)) + background).ravel()


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


def fit_affine(slm_points, cam_points):
    """Find affine transform mapping SLM coordinates to camera coordinates.

    Solves the least squares problem for the system:
        [y_cam]   [a b] [y_slm]   [ty]
        [x_slm] = [c d] [x_slm] + [tx]

    Parameters
    ----------
    slm_points : ndarray, shape (N, 2)
        Known (x, y) coordinates in SLM space.
    cam_points : ndarray, shape (N, 2)
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
    N = len(slm_points)

    # build the design matrix: each point contributes two rows
    # [y_slm, x_slm, 1, 0,     0,     0]   [a ]   [y_cam]
    # [0,     0,     0, y_slm, x_slm, 1] * [b ] = [x_cam]
    #                                      [ty]
    #                                      [c ]
    #                                      [d ]
    #                                      [tx]
    M = np.zeros((2 * N, 6))
    b = np.zeros(2 * N)

    for i, ((ys, xs), (yc, xc)) in enumerate(zip(slm_points, cam_points)):
        M[2*i]     = [ys, xs, 1, 0,  0,  0]
        M[2*i + 1] = [0,  0,  0, ys, xs, 1]
        b[2*i]     = yc
        b[2*i + 1] = xc

    params, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
    a, b_, ty, c, d, tx = params

    A = np.array([[a, b_], [c, d]])
    t = np.array([ty, tx])

    predicted = (A @ slm_points.T).T + t
    residuals = np.linalg.norm(predicted - cam_points, axis=1)

    return A, t, residuals


def direct_prepare(xs, ys, two_pi_modulation, xperiod, yperiod, centers, sigma, n):
    y0, x0 = centers[n]
    mode = np.exp(-((xs - x0)**2 + (ys - y0)**2) / (2 * sigma**2))
    return slmcontrol.generate_hologram(
        np.concatenate([np.ones_like(mode), mode], axis=1),
        two_pi_modulation, xperiod, yperiod,
    )

def reciprocal_prepare(xs, ys, two_pi_modulation, xperiod, yperiod, centers, sigma, n):
    ky, kx = centers[n]
    Ny, Nx = xs.shape
    mode = np.exp(-((xs - Nx //2)**2 + (ys - Ny //2)**2) / (2 * sigma**2) + 2j * np.pi * ((kx - Nx //2) * xs / Nx + (ky - Ny //2)*ys / Ny))
    return slmcontrol.generate_hologram(
        np.concatenate([np.ones_like(mode), mode], axis=1),
        two_pi_modulation, xperiod, yperiod,
    )

def _measure(dataset, camera, n):
    dataset[n] = np.flip(camera.capture(), axis=0)

def calibrate(_prepare, camera, xs, ys, centers, sigma):
    prepare = partial(_prepare, xs, ys, 192, -3, 19, centers, sigma) 
    test_img = camera.capture()
    images = np.empty((n**2, *test_img.shape), test_img.dtype)
    measure = partial(_measure, images, camera)

    slmcontrol.prepare_and_measure(prepare, measure, slm, 0.3, n**2)
    print("Finished acquisition. Fitting centroids...")

    # --- fit centroids ---
    cam_points = np.array([fit_centroid(images[i]) for i in range(n**2)])  # (N, 2) in camera (x, y)
    slm_points = np.array(centers)               # (N, 2) in SLM (x, y)

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

if __name__ == "__main__":
    slm = slmcontrol.SLMDisplay(host="localhost")
    camera_is = ImagingSourceCamera()
    camera_is.set_exposure(100)

    camera_xi = XimeaCamera()
    camera_xi.set_exposure(100)

    _xs = np.arange(slm.width // 2)
    _ys = np.arange(slm.height)
    xs, ys = np.meshgrid(_xs, _ys)

    n = 6

    x0s = np.linspace(-70, 70, n) + slm.width // 4
    y0s = np.linspace(-70, 70, n) + slm.height // 2

    centers_direct = np.array(list(itertools.product(y0s, x0s)))

    kxs = np.linspace(-30, 30, n) + slm.width // 4
    kys = np.linspace(-30, 30, n) + slm.height // 2
    centers_reciprocal = np.array(list(itertools.product(kys, kxs)))

    print(10 * "-" + "Direct Calibration" + 10 * "-")
    A_direct, t_direct, images_direct = calibrate(direct_prepare, camera_is, xs, ys, centers_direct, 15)

    print(10 * "-" + "Reciprocal Calibration" + 10 * "-")
    A_reciprocal, t_reciprocal, images_reciprocal = calibrate(reciprocal_prepare, camera_xi, xs, ys, centers_reciprocal, 30)

    fig, axs = plt.subplots(n, n)
    for (im, ax) in zip(images_direct, axs.flatten()):
        ax.imshow(im)
    plt.savefig("calibration/calibration_images_direct.png")

    fig, axs = plt.subplots(n, n)
    for (im, ax) in zip(images_reciprocal, axs.flatten()):
        ax.imshow(im)
    plt.savefig("calibration/calibration_images_reciprocal.png")

    with h5py.File("calibration/calibration.h5", "w") as f:
        f["A_direct"] = A_direct
        f["t_direct"] = t_direct
        f["A_reciprocal"] = A_reciprocal
        f["t_reciprocal"] = t_reciprocal

    camera_is.close()
    slm.close()