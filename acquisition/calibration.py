import argparse
import numpy as np
from slm_camera_calibration import calibrate
from common.utils import generate_amplitude_and_phase_hologram, inverse_fourier_transform
import slmcontrol
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from custom_devices.custom_ximea import CustomXimea
import tomllib
import shutil
from functools import partial


def mode2holo(slm, config, mode):
    return generate_amplitude_and_phase_hologram(
        mode,
        np.zeros_like(mode),
        config["hologram"]["two_pi_modulation"],
        config["hologram"]["xperiod"],
        config["hologram"]["yperiod"],
        slm_shape=(slm.height, slm.width//2)
    )

def measure(images, camera, n):
    images[n] = camera.capture()


def main(result_directory, config_path=Path("config.toml")):
    result_directory = Path(result_directory)
    result_directory.mkdir(parents=True, exist_ok=True)
    calibration_directory = result_directory / "calibration_data"
    calibration_directory.mkdir(exist_ok=True)
    plots_directory = calibration_directory / "plots"
    plots_directory.mkdir(exist_ok=True)

    shutil.copy2(config_path, Path(result_directory) / "config.toml")
    with config_path.open("rb") as file:
        config = tomllib.load(file)

    calibration_path = calibration_directory / "calibration.h5"

    N = config["grid"]["size"]
    slm = slmcontrol.SLMDisplay(host=config["slm"]["host"])
    _xs = np.arange(N) - N // 2
    _ys = np.arange(N) - N // 2
    xs, ys = np.meshgrid(_xs, _ys)


    shifts1d = np.arange(-4, 6, 2, dtype=int)
    shifts = np.array([[y, x] for y in shifts1d for x in shifts1d])

    images = np.empty((len(shifts), N, N), dtype=np.uint8)
    base_mode_fourier = slmcontrol.hg(xs, ys, w=1)

    with CustomXimea() as camera:
        camera.set_exposure(config["fourier_camera"]["calibration_exposure"])
        holo = mode2holo(slm, config, slmcontrol.hg(xs, ys, w=config["fourier_camera"]["coarse_calibration_waist"]))
        slm.updateArray(holo)
        image = camera.capture()
        plt.imshow(image)
        plt.savefig(plots_directory / "coarse_calibration_full.png")
        camera.calibrate(N, image)
        plt.imshow(camera.capture())
        plt.savefig(plots_directory / "coarse_calibration.png")

        result_fourier = calibrate(
            base_mode=base_mode_fourier,
            inverse_transform=inverse_fourier_transform,
            mode2holo=partial(mode2holo, slm, config),
            measure=partial(measure, images, camera),
            images=images,
            Xs=shifts,
            slm=slm,
            settle_time=config["capture"]["settling_time_s"],
        )
        result_fourier.save(calibration_path)
        with h5py.File(calibration_path, "a") as f:
            f["calibration_image"] = image
            f["N"] = N
            f["offsetY"] = camera.get_offsetY()
            f["offsetX"] = camera.get_offsetX()
    slm.close()

    det_matrix = np.linalg.det(result_fourier.transform.matrix)
    
    print(f"Determinant signal: {np.sign(det_matrix)}")
    print(f"Magnification: {np.sqrt(np.abs(det_matrix)):.2f}")
    print(f"Angle: {np.rad2deg(np.atan2(result_fourier.transform.matrix[1, 0], result_fourier.transform.matrix[0, 0])):.4f} degrees")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate the direct and Fourier camera images.")
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    arguments = parser.parse_args()
    main(arguments.result_directory, arguments.config)
