# ASK 21 IA optimization code

This repository contains only the Python code and input data used for two parts of the Mathematics IA:

1. **Optimization under headwind and downdraft profiles**
2. **Layered airspeed strategy**

The model uses the IA's piecewise quadratic sink-rate function and optimizes horizontal range over the supported ASK 21 airspeed domain.

## Repository contents

### `headwind_downdraft/`

- `section_6_profiles.py` evaluates the seven constructed headwind/downdraft profiles used in the IA.
- `data/ask21_polar.csv` supplies the supported airspeed domain.
- `data/profiles.csv` supplies the profile values and starting altitude.

Run it from the repository root with:

```powershell
python headwind_downdraft/section_6_profiles.py
```

### `layered_airspeed/`

- `layered_air_strategy.py` optimizes the airspeed independently in each atmospheric layer and compares the result with the best single constant-speed strategy.
- `model_parameters.csv` contains the piecewise-model parameters.
- `layers.csv` contains the layer assumptions.
- `test_layered_air_strategy.py` contains verification tests.

Run the model and tests with:

```powershell
python layered_airspeed/layered_air_strategy.py
python -m unittest discover -s layered_airspeed -p "test_*.py" -v
```

## Important scope note

The headwind/downdraft profiles are constructed test conditions. The altitude allocation in `layered_airspeed/layers.csv` is illustrative rather than sourced flight data. The scripts verify and visualize the IA mathematics; they do not claim to prescribe exact real-flight speeds.

Generated figures, result tables, caches, document drafts, and unrelated exploratory code are intentionally excluded.
