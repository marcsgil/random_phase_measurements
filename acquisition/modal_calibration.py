from cameras import ImagingSourceNew, Ximea
from acquisition.generate_modes import up_to_order_basis
import numpy as np
import matplotlib.pyplot as plt
from common.utils import generate_amplitude_and_phase_hologram, resize_and_center
from acquisition.config import load_config
from functools import partial
import slmcontrol


def flat_top(xs, ys, radius):
    return (xs**2 + ys**2) < radius**2


def get_tomography_basis(xs, ys, waist, order, flat_top_radius):
    d = (order + 1) * (order + 2) // 2

    basis = np.empty(
        (3 * d + 1, ys.shape[0], xs.shape[1]),
        dtype=np.complex64
    )

    basis[1:d+1] = up_to_order_basis(
        xs,
        ys,
        waist,
        order
    )

    basis[0] = (
        flat_top(xs, ys, flat_top_radius)
        * np.max(np.abs(basis[1:d+1]))
    )

    for n in range(d):
        basis[d+1+n] = (
            basis[0] + basis[n+1]
        ) / np.sqrt(2)

        basis[2*d+1+n] = (
            basis[0] + 1j * basis[n+1]
        ) / np.sqrt(2)

    return basis


def recover_basis(images):
    d = (images.shape[0] - 1) // 3

    interferences = (
        images[d+1:2*d+1]
        + 1j * images[2*d+1:]
        - (1 + 1j)
        * (images[0] + images[1:d+1])
        / 2
    )

    return interferences / np.sqrt(images[0])


def _prepare(
    basis,
    height,
    width,
    two_pi_modulation,
    xperiod,
    yperiod,
    n
):
    _mode = resize_and_center(
        basis[n],
        (height, width // 2),
        1
    )

    return generate_amplitude_and_phase_hologram(
        _mode,
        np.zeros_like(_mode),
        two_pi_modulation,
        xperiod,
        yperiod
    )


def _measure(images, camera_direct, n):
    images[n] = camera_direct.capture()


if __name__ == "__main__":
    config = load_config()

    # SLM
    slm = slmcontrol.SLMDisplay(
        host=config["slm"]["host"]
    )

    # Grid
    N = config["grid"]["size"]

    _xs = np.arange(N) - N // 2
    _ys = np.arange(N) - N // 2

    xs, ys = np.meshgrid(
        _xs,
        _ys
    )

    # Basis
    flat_top_radius = 100

    tomography_basis = get_tomography_basis(
        xs,
        ys,
        30,
        2,
        flat_top_radius
    )

    # Camera
    camera_direct = ImagingSourceNew.ImagingSourceCamera()

    sample_direct = camera_direct.capture()

    images = np.empty_like(
        sample_direct,
        shape=(
            len(tomography_basis),
            *sample_direct.shape
        )
    )

    prepare = partial(
        _prepare,
        tomography_basis,
        slm.height,
        slm.width,
        config["hologram"]["two_pi_modulation"],
        config["hologram"]["xperiod"],
        config["hologram"]["yperiod"]
    )

    measure = partial(
        _measure,
        images,
        camera_direct
    )

    # slmcontrol.prepare_and_measure(
    #     prepare,
    #     measure,
    #     slm,
    #     0.3,
    #     len(tomography_basis)
    # )

    images = np.abs(
        tomography_basis
    )**2

    background = 3

    # corrected_images = np.where(
    #     images > background,
    #     images - background,
    #     0
    # )

    corrected_images = images

    basis = recover_basis(
        corrected_images
        / np.sum(
            corrected_images,
            axis=(-1, -2),
            keepdims=True
        )
    )

    fig, ax = plt.subplots(
        1,
        2
    )

    for n, element in enumerate(basis):
        ax[0].clear()
        ax[1].clear()

        ax[0].imshow(
            np.abs(element)**2
        )

        ax[1].imshow(
            np.angle(element),
            cmap="twilight"
        )

        fig.savefig(
            f"plots/temp{n}.png"
        )