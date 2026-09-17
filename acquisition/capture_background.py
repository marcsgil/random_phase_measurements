import argparse
from pathlib import Path

import h5py
import numpy as np
import slmcontrol
from acquisition.config import fourier_roi, load_config, snapshot_config


def average_background(capture, num_frames, **kwargs):
    image = capture(**kwargs)
    total = np.asarray(image, dtype=np.float64)
    for _ in range(num_frames - 1):
        total += capture(**kwargs)
    return np.ceil(total / num_frames).astype(image.dtype)


def main(result_directory, config_path):
    from cameras.ImagingSourceNew import ImagingSourceCamera
    from cameras.Ximea import XimeaCamera

    result_directory = Path(result_directory)
    result_directory.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_path)
    config = load_config(config_path)
    roi_fourier = fourier_roi(config)
    num_frames = config["background"]["num_frames"]
    phase_exposures = config["fourier_camera"]["phase_exposures"]
    if len(phase_exposures) != len(config["phases"]["sigmas"]):
        raise ValueError("phase_exposures must contain one exposure per sigma.")

    camera_direct = ImagingSourceCamera()
    camera_direct.set_exposure(config["direct_camera"]["exposure"])
    camera_fourier = XimeaCamera()
    fourier_camera = config["fourier_camera"]

    slm = slmcontrol.SLMDisplay(host=config["slm"]["host"])
    holo = np.zeros((slm.height, slm.width), dtype=np.uint8)
    slm.updateArray(holo)
    slm.close()

    try:
        print(f"Capture {num_frames} background frames with the optical input blocked.")
        image_direct = average_background(camera_direct.capture, num_frames)
        camera_fourier.set_exposure(fourier_camera["fourier_exposure"])
        image_fourier = average_background(
            camera_fourier.capture,
            num_frames,
            roi=roi_fourier,
        )
        images_phase_fourier = []
        for exposure in phase_exposures:
            camera_fourier.set_exposure(exposure)
            images_phase_fourier.append(
                average_background(camera_fourier.capture, num_frames, roi=roi_fourier)
            )
    finally:
        camera_direct.close()
        camera_fourier.close()

    with h5py.File(result_directory / "background.h5", "w") as file:
        file["images_direct"] = image_direct
        file["images_fourier"] = image_fourier
        file["images_phase_fourier"] = np.stack(images_phase_fourier)
        file["roi_fourier"] = roi_fourier
        file.attrs["fourier_exposure"] = fourier_camera["fourier_exposure"]
        file.attrs["phase_exposures"] = phase_exposures

    snapshot_config(config_path, result_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture direct and Fourier camera backgrounds.")
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    arguments = parser.parse_args()
    main(arguments.result_directory, arguments.config)
