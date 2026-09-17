import argparse
from pathlib import Path

from acquisition import calibration, capture_background, capture_linear_combinations
from acquisition import generate_modes, generate_phase_masks
from preprocessing.match_phase_fourier_basis_to_experiment import prepare


def create_result_directory(result_directory):
    result_directory = Path(result_directory)
    if result_directory.exists():
        raise FileExistsError(f"Destination already exists: {result_directory}")
    result_directory.mkdir(parents=True)
    return result_directory


def main(result_directory, config_path=Path("config.toml")):
    result_directory = create_result_directory(result_directory)

    print("Calibrating cameras")
    calibration.main(result_directory, config_path)
    print("Generating phase masks")
    generate_phase_masks.main(result_directory, config_path)
    print("Generating modes")
    generate_modes.main(result_directory, config_path)
    print("Capturing background")
    capture_background.main(result_directory, config_path)
    print("Capturing images")
    capture_linear_combinations.main(result_directory, config_path)
    print("Preparing camera-grid theory")
    prepare(result_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a complete random-phase measurement acquisition.")
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    arguments = parser.parse_args()
    main(arguments.result_directory, arguments.config)
