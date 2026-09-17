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


def main(result_directory, config_path, num_frames=1000):
    from cameras.ImagingSourceNew import ImagingSourceCamera
    from cameras.Ximea import XimeaCamera

    result_directory = Path(result_directory)
    result_directory.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_path)
    config = load_config(config_path)
    roi_fourier = fourier_roi(config)

    camera_direct = ImagingSourceCamera()
    camera_direct.set_exposure(config["direct_camera"]["exposure"])
    camera_fourier = XimeaCamera()
    camera_fourier.set_exposure(config["fourier_camera"]["exposure"])

    slm = slmcontrol.SLMDisplay(host=config["slm"]["host"])
    holo = np.zeros((slm.height, slm.width), dtype=np.uint8)
    slm.updateArray(holo)
    slm.close()

    try:
        print(f"Capture {num_frames} background frames with the optical input blocked.")
        image_direct = average_background(camera_direct.capture, num_frames)
        image_fourier = average_background(
            camera_fourier.capture,
            num_frames,
            roi=roi_fourier,
        )
    finally:
        camera_direct.close()
        camera_fourier.close()

    with h5py.File(result_directory / "background.h5", "w") as file:
        file["images_direct"] = image_direct
        file["images_fourier"] = image_fourier
        file["roi_fourier"] = roi_fourier

    snapshot_config(config_path, result_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture direct and Fourier camera backgrounds.")
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--num-frames", type=int, default=1000)
    arguments = parser.parse_args()
    main(arguments.result_directory, arguments.config, arguments.num_frames)
