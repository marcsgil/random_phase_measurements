import argparse
import numpy as np
from slm_camera_calibration import calibrate
from common.utils import generate_amplitude_and_phase_hologram, resize_and_center, fourier_transform, inverse_fourier_transform
import slmcontrol
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
import h5py
from pathlib import Path

from acquisition.config import fourier_roi, load_config, snapshot_config


def main(result_directory, config_path=Path("config.toml")):
    from cameras.Ximea import XimeaCamera
    from cameras.ImagingSourceNew import ImagingSourceCamera

    result_directory = Path(result_directory)
    result_directory.mkdir(parents=True, exist_ok=True)
    calibration_directory = result_directory / "calibration_data"
    calibration_directory.mkdir(exist_ok=True)
    plots_directory = calibration_directory / "plots"
    plots_directory.mkdir(exist_ok=True)

    config = load_config(config_path)
    snapshot_config(config_path, result_directory)
    shifts1d_direct = np.arange(-40, 60, 20, dtype=int)
    shifts_direct = np.array([[y, x] for y in shifts1d_direct for x in shifts1d_direct])

    camera_direct = ImagingSourceCamera()
    camera_direct.set_exposure(config["direct_camera"]["exposure"])
    images_direct = np.empty((len(shifts_direct), *camera_direct.capture().shape), dtype=np.uint8)


    shifts1d_fourier = np.arange(-4, 6, 2, dtype=int)
    shifts_fourier = np.array([[y, x] for y in shifts1d_fourier for x in shifts1d_fourier])

    camera_fourier = XimeaCamera()
    fourier_camera = config["fourier_camera"]
    camera_fourier.camera.enable_aeag()
    camera_fourier.camera.set_aeag_roi_width(fourier_camera["width"])
    camera_fourier.camera.set_aeag_roi_height(fourier_camera["height"])
    camera_fourier.camera.set_aeag_roi_offset_x(fourier_camera["offset_x"])
    camera_fourier.camera.set_aeag_roi_offset_y(fourier_camera["offset_y"])
    camera_fourier.camera.set_exp_priority(fourier_camera["exposure_priority"])
    camera_fourier.camera.set_aeag_level(fourier_camera["aeag_level"])
    roi_fourier = fourier_roi(config)
    images_fourier = np.empty((len(shifts_fourier), *camera_fourier.capture(roi=roi_fourier).shape), dtype=np.uint8)

    slm = slmcontrol.SLMDisplay(host=config["slm"]["host"])
    N = config["grid"]["size"]
    _xs = np.arange(N) - N // 2
    _ys = np.arange(N) - N // 2
    xs, ys = np.meshgrid(_xs, _ys)

    def mode2holo(mode):
        _mode = resize_and_center(mode, (slm.height, slm.width//2), 1)
        return generate_amplitude_and_phase_hologram(
            _mode,
            np.zeros_like(_mode),
            config["hologram"]["two_pi_modulation"],
            config["hologram"]["xperiod"],
            config["hologram"]["yperiod"],
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
        settle_time=config["capture"]["settling_time_s"],
    )

    result_direct.save(calibration_directory / "calibration_direct.h5")

    base_mode_fourier = slmcontrol.hg(xs, ys, w=2)

    result_fourier = calibrate(
        base_mode=base_mode_fourier,
        inverse_transform=inverse_fourier_transform,
        mode2holo=mode2holo,
        measure=measure_fourier,
        images=images_fourier,
        Xs=shifts_fourier,
        slm=slm,
        settle_time=config["capture"]["settling_time_s"],
    )

    fourier_calibration_path = calibration_directory / "calibration_fourier.h5"
    result_fourier.save(fourier_calibration_path)
    with h5py.File(fourier_calibration_path, "a") as f:
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
    plt.savefig(plots_directory / "calibration_direct.png")

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(transformed_image_fourier[3*N//8:-3*N//8,3*N//8:-3*N//8])
    axs[1].imshow(np.abs(fourier_transform(mode))[3*N//8:-3*N//8,3*N//8:-3*N//8] ** 2)
    plt.savefig(plots_directory / "calibration_fourier.png")

    slm.close()
    camera_direct.close()
    camera_fourier.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate the direct and Fourier camera images.")
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    arguments = parser.parse_args()
    main(arguments.result_directory, arguments.config)
