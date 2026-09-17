from pathlib import Path
import shutil
import tomllib


DEFAULT_CONFIG_PATH = Path("config.toml")


def load_config(path=DEFAULT_CONFIG_PATH):
    path = Path(path)
    with path.open("rb") as file:
        return tomllib.load(file)


def fourier_roi(config):
    camera = config["fourier_camera"]
    return (
        camera["offset_y"],
        camera["offset_y"] + camera["height"],
        camera["offset_x"],
        camera["offset_x"] + camera["width"],
    )


def snapshot_config(config_path, result_directory):
    shutil.copy2(config_path, Path(result_directory) / "config.toml")
