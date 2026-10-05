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
uv run -m acquisition.run results/run_001
```

The one-shot command requires a destination that does not already exist. To
run individual stages instead:

```bash
uv run -m acquisition.calibration results/test
uv run -m acquisition.capture_background results/test
uv run -m acquisition.generate_phase_masks results/test
uv run -m acquisition.generate_modes results/test
uv run -m preprocessing.match_phase_fourier_basis_to_experiment results/test
uv run -m acquisition.capture_linear_combinations results/test
```

Each preparation, calibration, and capture snapshots the active `config.toml`
into the result directory.

The current Julia analysis entry point is:

```bash
julia --project=. analysis/single_image_tomography.jl
```

The scripts under `three_image/` and `archive/` are retained as references;
they use older data layouts or solver dependencies.
