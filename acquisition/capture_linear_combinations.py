import argparse
import itertools
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import slmcontrol
from scipy.linalg import polar
from slm_camera_calibration import CalibrationResult

from analysis import diagnose_linear_combinations
from acquisition.config import fourier_roi, load_config, snapshot_config
from common.utils import (
    extraction_linear_combination,
    generate_amplitude_and_phase_hologram,
    linear_transformation,
)

def mean_capture(camera, samples, *args, **kwargs):
    sample_image = camera.capture(*args,  **kwargs) 
    mean_image = sample_image / samples
    for _ in range(samples-1):
        mean_image += camera.capture(*args,  **kwargs) / samples
    return np.floor(mean_image).astype(sample_image.dtype)


def _prepare_no_phase(mode, slm_shape, extraction, hologram_config, n):
    mode = extraction(mode, n)
    phase = np.zeros_like(mode)
    return generate_amplitude_and_phase_hologram(mode, phase, slm_shape=slm_shape, **hologram_config)


def _prepare_phase(mode, phases, indices, slm_shape, extraction, unitary, hologram_config, n):
    sigma_idx, phase_idx, coeff_idx = indices[n]
    mode = extraction(mode, coeff_idx)
    phase = linear_transformation(np.flip(phases[sigma_idx, phase_idx], axis=1), unitary)
    return generate_amplitude_and_phase_hologram(mode, phase, slm_shape=slm_shape, **hologram_config)


def _measure_no_phase(images_direct, images_fourier, camera_direct, camera_fourier, roi_fourier, aeag_settling_captures, n):
    images_direct[n] = mean_capture(camera_direct, aeag_settling_captures)
    images_fourier[n] = mean_capture(camera_fourier, aeag_settling_captures, roi=roi_fourier)


def _measure_phase(images_phase_fourier, camera_fourier, indices, roi_fourier, aeag_settling_captures, n):
    sigma_idx, phase_idx, coeff_idx = indices[n]
    images_phase_fourier[sigma_idx, phase_idx, coeff_idx] = mean_capture(camera_fourier, aeag_settling_captures, roi=roi_fourier)


def capture_order(
    slm,
    camera_direct,
    camera_fourier,
    roi_fourier,
    modes,
    phases,
    order_directory,
    unitary,
    opening_mode,
    config,
):
    slm_shape = (slm.height, slm.width // 2)
    num_sigmas, num_phases = phases.shape[:2]
    num_modes = len(modes[0]) if isinstance(modes, tuple) else len(modes)

    image_direct = camera_direct.capture()
    image_fourier = camera_fourier.capture(roi=roi_fourier)

    with h5py.File(order_directory / "data.h5", opening_mode) as file:
        images_direct = file.create_dataset("images_direct", (num_modes, *image_direct.shape), image_direct.dtype)
        images_fourier = file.create_dataset("images_fourier", (num_modes, *image_fourier.shape), image_fourier.dtype)
        images_phase_fourier = file.create_dataset(
            "images_phase_fourier",
            (num_sigmas, num_phases, num_modes, *image_fourier.shape),
            image_fourier.dtype,
        )

        print(10 * "-" + "Measuring without phase" + 10 * "-")
        prepare = partial(_prepare_no_phase, modes, slm_shape, extraction_linear_combination, config["hologram"])
        measure = partial(
            _measure_no_phase,
            images_direct,
            images_fourier,
            camera_direct,
            camera_fourier,
            roi_fourier,
            config["capture"]["aeag_settling_captures"],
        )
        slmcontrol.prepare_and_measure(prepare, measure, slm, config["capture"]["settling_time_s"], num_modes)

        print(10 * "-" + "Measuring with phase" + 10 * "-")
        indices = list(itertools.product(range(num_sigmas), range(num_phases), range(num_modes)))
        prepare = partial(
            _prepare_phase,
            modes,
            phases,
            indices,
            slm_shape,
            extraction_linear_combination,
            unitary,
            config["hologram"],
        )
        measure = partial(
            _measure_phase,
            images_phase_fourier,
            camera_fourier,
            indices,
            roi_fourier,
            config["capture"]["aeag_settling_captures"],
        )
        slmcontrol.prepare_and_measure(prepare, measure, slm, config["capture"]["settling_time_s"], len(indices))

    diagnose_linear_combinations.main(order_directory, config["capture"]["diagnostic_samples"])


def order_directories(result_directory):
    return sorted(
        directory
        for directory in result_directory.iterdir()
        if directory.is_dir() and (directory / "modes.h5").is_file()
    )


def main(result_directory, config_path):
    from cameras.ImagingSourceNew import ImagingSourceCamera
    from cameras.Ximea import XimeaCamera

    result_directory = Path(result_directory)
    config_path = Path(config_path)
    config = load_config(config_path)
    configured_grid_shape = (config["grid"]["size"], config["grid"]["size"])
    with h5py.File(result_directory / "phases.h5") as file:
        phases = file["phases"][:]
        phase_grid_shape = tuple(file["grid_shape"][:])

    if phase_grid_shape != configured_grid_shape:
        raise ValueError(f"Grid mismatch: config uses {configured_grid_shape}, phases use {phase_grid_shape}.")

    calibration_directory = result_directory / "calibration_data"
    calibration_direct = CalibrationResult.load(calibration_directory / "calibration_direct.h5")
    unitary, _ = polar(calibration_direct.transform.matrix)
    roi_fourier = fourier_roi(config)
    with h5py.File(calibration_directory / "calibration_fourier.h5") as file:
        recorded_roi = tuple(file["roi"][:])

    if recorded_roi != roi_fourier:
        raise ValueError(f"Fourier ROI mismatch: config uses {roi_fourier}, calibration uses {recorded_roi}.")

    prepared_orders = order_directories(result_directory)
    for order_directory in prepared_orders:
        with h5py.File(order_directory / "modes.h5") as file:
            mode_grid_shape = tuple(file["grid_shape"][:])
        if mode_grid_shape != phase_grid_shape:
            raise ValueError(f"Grid mismatch: phases use {phase_grid_shape}, modes use {mode_grid_shape}.")

    snapshot_config(config_path, result_directory)

    opening_mode = "w" if result_directory.name == "test" else "a"
    slm = slmcontrol.SLMDisplay(host=config["slm"]["host"])
    camera_direct = ImagingSourceCamera()
    camera_direct.set_exposure(config["direct_camera"]["exposure"])
    camera_fourier = XimeaCamera()

    fourier_camera = config["fourier_camera"]
    camera_fourier.set_exposure(fourier_camera["exposure"])
    # camera_fourier.camera.enable_aeag()
    # camera_fourier.camera.set_aeag_roi_width(fourier_camera["width"] // 2)
    # camera_fourier.camera.set_aeag_roi_height(fourier_camera["height"] // 2)
    # camera_fourier.camera.set_aeag_roi_offset_x(fourier_camera["offset_x"] + fourier_camera["width"] // 4)
    # camera_fourier.camera.set_aeag_roi_offset_y(fourier_camera["offset_y"] + fourier_camera["height"] // 4)
    # camera_fourier.camera.set_exp_priority(fourier_camera["exposure_priority"])
    # camera_fourier.camera.set_aeag_level(fourier_camera["aeag_level"])

    try:
        for order_directory in prepared_orders:
            with h5py.File(order_directory / "modes.h5") as file:
                modes = (file["coefficients"][:], file["basis"][:])

            num_modes = len(modes[0])
            num_sigmas, num_phases = phases.shape[:2]
            print(f"Capturing {order_directory.name}")
            print(f"Estimated time: {num_modes * (1 + num_phases * num_sigmas) * 0.3 / 60:.1f} min")
            capture_order(
                slm,
                camera_direct,
                camera_fourier,
                roi_fourier,
                modes,
                phases,
                order_directory,
                unitary,
                opening_mode,
                config,
            )
    finally:
        slm.close()
        camera_direct.close()
        camera_fourier.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture prepared linear-combination measurements.")
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    arguments = parser.parse_args()
    main(arguments.result_directory, arguments.config)
