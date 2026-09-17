import numpy as np
from slm_camera_calibration import calibrate
from cameras.Ximea import XimeaCamera
from cameras.ImagingSourceNew import ImagingSourceCamera
from common.utils import generate_amplitude_and_phase_hologram, resize_and_center, fourier_transform, inverse_fourier_transform
import slmcontrol
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
import h5py

def main():
    shifts1d_direct = np.arange(-40, 60, 20, dtype=int)
    shifts_direct = np.array([[y, x] for y in shifts1d_direct for x in shifts1d_direct])

    camera_direct = ImagingSourceCamera()
    camera_direct.set_exposure(100)
    images_direct = np.empty((len(shifts_direct), *camera_direct.capture().shape), dtype=np.uint8)


    shifts1d_fourier = np.arange(-4, 6, 2, dtype=int)
    shifts_fourier = np.array([[y, x] for y in shifts1d_fourier for x in shifts1d_fourier])

    camera_fourier = XimeaCamera()
    width = 384
    height = 384
    offset_x = 372
    offset_y = 484
    camera_fourier.camera.enable_aeag()
    camera_fourier.camera.set_aeag_roi_width(width)
    camera_fourier.camera.set_aeag_roi_height(height)
    camera_fourier.camera.set_aeag_roi_offset_x(offset_x)
    camera_fourier.camera.set_aeag_roi_offset_y(offset_y)
    camera_fourier.camera.set_exp_priority(1.0)
    camera_fourier.camera.set_aeag_level(4)
    roi_fourier = (offset_y, offset_y + height, offset_x, offset_x + width)
    images_fourier = np.empty((len(shifts_fourier), *camera_fourier.capture(roi=roi_fourier).shape), dtype=np.uint8)

    slm = slmcontrol.SLMDisplay(host="localhost")
    N = 256
    _xs = np.arange(N) - N // 2
    _ys = np.arange(N) - N // 2
    xs, ys = np.meshgrid(_xs, _ys)

    def mode2holo(mode):
        _mode = resize_and_center(mode, (slm.height, slm.width//2), 1)
        return generate_amplitude_and_phase_hologram(
            _mode,
            np.zeros_like(_mode),
            192,
            -3,
            19,
        )

    def measure_direct(n):
        images_direct[n] = camera_direct.capture()

    def measure_fourier(n):
        for _ in range(2):
            # For autoexposure to settle
            camera_fourier.capture()
        images_fourier[n] = camera_fourier.capture(roi=roi_fourier)

    base_mode_direct = slmcontrol.hg(xs, ys, w=20)

    result_direct = calibrate(
        base_mode=base_mode_direct,
        mode2holo=mode2holo,
        measure=measure_direct,
        images=images_direct,
        Xs=shifts_direct,
        slm=slm,
        settle_time=0.3,
    )

    result_direct.save("calibration_data/calibration_direct.h5")

    base_mode_fourier = slmcontrol.hg(xs, ys, w=2)

    result_fourier = calibrate(
        base_mode=base_mode_fourier,
        inverse_transform=inverse_fourier_transform,
        mode2holo=mode2holo,
        measure=measure_fourier,
        images=images_fourier,
        Xs=shifts_fourier,
        slm=slm,
        settle_time=0.3,
    )

    result_fourier.save("calibration_data/calibration_fourier.h5")
    with h5py.File("calibration_data/calibration_fourier.h5", "a") as f:
        f["roi"] = roi_fourier

    mode = slmcontrol.diagonal_hg(xs, ys, m=5, w=20)
    holo = mode2holo(resize_and_center(
            mode,
            (slm.height, slm.width // 2),
            scale=1,
        ))
    slm.updateArray(holo, sleep_time=0.3)

    image_direct = camera_direct.capture()
    image_fourier = camera_fourier.capture(roi=roi_fourier)

    transformed_image_direct = affine_transform(
        image_direct,
        matrix=result_direct.transform.matrix,
        offset=result_direct.transform.offset, # type: ignore
        output_shape=result_direct.output_shape,
    )

    transformed_image_fourier = affine_transform(
        image_fourier,
        matrix=result_fourier.transform.matrix,
        offset=result_fourier.transform.offset, # type: ignore
        output_shape=result_fourier.output_shape,
    )

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(transformed_image_direct)
    axs[1].imshow(np.abs(mode) ** 2)
    plt.savefig("plots/calibration_direct.png")

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(transformed_image_fourier[3*N//8:-3*N//8,3*N//8:-3*N//8])
    axs[1].imshow(np.abs(fourier_transform(mode))[3*N//8:-3*N//8,3*N//8:-3*N//8] ** 2)
    plt.savefig("plots/calibration_fourier.png")

    slm.close()
    camera_direct.close()
    camera_fourier.close()

if __name__ == "__main__":
    main()
