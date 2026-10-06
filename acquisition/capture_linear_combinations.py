import argparse
import itertools
from functools import partial
from pathlib import Path
import tomllib
import h5py
import numpy as np
import slmcontrol
from slm_camera_calibration import CalibrationResult

from analysis import diagnose_linear_combinations
from common.utils import (
    extraction_linear_combination,
    generate_amplitude_and_phase_hologram,
)
from custom_devices.custom_ximea import CustomXimea

def _prepare_phase(mode, phases, indices, slm_shape, extraction, hologram_config, n):
    sigma_idx, phase_idx, coeff_idx = indices[n]
    mode = extraction(mode, coeff_idx)
    return generate_amplitude_and_phase_hologram(mode, phases[sigma_idx, phase_idx], slm_shape=slm_shape, **hologram_config)

def _measure_phase(images_phase_fourier, camera, indices, n):
    images_phase_fourier[*indices[n]] = camera.capture()

def capture_order(
    slm,
    camera,
    modes,
    phases,
    order_directory,
    config
):
    slm_shape = (slm.height, slm.width // 2)
    num_sigmas, num_phases = phases.shape[:2]
    num_modes = len(modes[0]) if isinstance(modes, tuple) else len(modes)
    N = config["grid"]["size"]

    with h5py.File(order_directory / "data.h5", "a") as file:
        images_phase_fourier = file.create_dataset(
            "images_phase_fourier",
            (num_sigmas, num_phases, num_modes, N, N),
            np.uint8,
        )

        print(10 * "-" + "Measuring with phase" + 10 * "-")
        for sigma_index, exposure in enumerate(config["fourier_camera"]["phase_exposures"]):
            camera.set_exposure(exposure)
            indices = list(itertools.product((sigma_index,), range(num_phases), range(num_modes)))
            prepare = partial(
                _prepare_phase,
                modes,
                phases,
                indices,
                slm_shape,
                extraction_linear_combination,
                config["hologram"],
            )
            measure = partial(
                _measure_phase,
                images_phase_fourier,
                camera,
                indices
            )
            slmcontrol.prepare_and_measure(prepare, measure, slm, config["capture"]["settling_time_s"], len(indices))

    # diagnose_linear_combinations.main(order_directory, config["capture"]["diagnostic_samples"])


def order_directories(result_directory):
    return sorted(
        directory
        for directory in result_directory.iterdir()
        if directory.is_dir() and (directory / "modes.h5").is_file()
    )


def main(result_directory, config_path):
    result_directory = Path(result_directory)
    config_path = Path(config_path)
    with config_path.open("rb") as file:
        config = tomllib.load(file)

    with h5py.File(result_directory / "phases.h5") as file:
        phases = np.asarray(file["phases"])

    with h5py.File(result_directory / "calibration_data" / "calibration.h5") as file:
        calibration_image = np.asarray(file["calibration_image"])

    slm = slmcontrol.SLMDisplay(host=config["slm"]["host"])
    try:
        for order_directory in order_directories(result_directory):
            with h5py.File(order_directory / "modes.h5") as file:
                modes = (np.asarray(file["coefficients"]), np.asarray(file["basis"]))

            print(f"Capturing {order_directory.name}")

            with CustomXimea() as camera:
                camera.calibrate(config["grid"]["size"], calibration_image)
                capture_order(
                    slm,
                    camera,
                    modes,
                    phases,
                    order_directory,
                    config
                )
    finally:
        slm.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture prepared linear-combination measurements.")
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    arguments = parser.parse_args()
    main(arguments.result_directory, arguments.config)
