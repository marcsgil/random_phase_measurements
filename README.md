# Random phase measurements

The repository is organized by workflow:

- `acquisition/`: calibration, image acquisition, and phase-screen generation;
- `preprocessing/`: theory-side preparation for the camera grid;
- `analysis/`: current single-image tomography and acquisition diagnostics;
- `three_image/`: three-image phase-retrieval algorithms;
- `sic/`: maintained SIC-POVM generation utilities and fiducial states;
- `common/`: shared Python field and image utilities;
- `archive/`: historical experiments and alternative solvers.

Run Python modules from the repository root, for example:

```bash
cp config_example.toml config.toml
python -m acquisition.calibration results/test
python -m acquisition.capture_background results/test
python -m acquisition.generate_phase_masks results/test
python -m acquisition.generate_modes results/test
python -m preprocessing.match_phase_fourier_basis_to_experiment results/test
python -m acquisition.capture_linear_combinations results/test
```

Each preparation, calibration, and capture snapshots the active `config.toml`
into the result directory.

The current Julia analysis entry point is:

```bash
julia --project=. analysis/single_image_tomography.jl
```

The scripts under `three_image/` and `archive/` are retained as references;
they use older data layouts or solver dependencies.
