# TransitAI Companion

TransitAI Companion is a Streamlit application for forecasting transit demand and operational pressure ahead of service time. It projects future trip volume, staffing needs, vehicle load, route demand, and service completion outlook from saved model artifacts, then presents the results in a planning-oriented interface with confidence bands, capacity warnings, and downloadable forecasts.

Developed and researched by Mohamed Gad.

## Overview

This project was built to help transit operators and operations teams answer practical planning questions before service begins:

- How many trips are likely to occur in a future service window?
- How many drivers and vehicles should be planned for that demand?
- When does the forecast suggest capacity pressure or operational intervention?
- How confident is the model at each forecasted point?

The current application runs in artifact-based deployment mode. It does not require the raw CSV dataset at runtime.

## Core capabilities

- Forecasts future trip demand for arbitrary future hourly timestamps
- Supports both date-range forecasting and custom timestamp forecasting
- Estimates required drivers, vehicles, and routes
- Predicts service completion outlook
- Computes confidence intervals and confidence scores
- Flags staffing and fleet shortfalls against user-entered available capacity
- Supports hourly and daily reporting views
- Exports the hourly forecast to CSV

## Intended use

This project is intended for:

- transit operations planning
- staffing and dispatch planning
- service readiness reviews
- operational scenario analysis
- model-driven research and prototyping

It is not presented as a production-certified decision system. Forecasts should be reviewed alongside operational judgment, local constraints, and current service conditions.

## Runtime model contract

The deployed app expects the following files in the project root:

- `rf_model.pkl`  
  Saved random forest model used for trip-demand forecasting

- `best_model.h5`  
  Saved Keras model used for drivers, vehicles, routes, and completion predictions

- `trips.pkl`  
  Optional saved trip-history artifact for restoring recent observed demand context in the UI

Additional development files in this repository:

- `code.ipynb`  
  Original notebook used during development, experimentation, and artifact generation

- `denver_trips_synth.csv`  
  Synthetic source dataset used during development and research. This file is not required by the deployed app runtime.

## Current deployment behavior

The deployed Streamlit app now loads saved model artifacts directly instead of rebuilding models from the raw CSV.

At runtime, the app:

1. loads the saved random forest demand model
2. builds calendar-based future features
3. predicts future trip demand
4. derives demand uncertainty from the random forest ensemble
5. runs the saved Keras model repeatedly with dropout enabled to estimate operational uncertainty
6. compares forecasted staffing and fleet needs against user-entered capacity

## Current limitations

- The saved deployment bundle does not include a dedicated OTP predictor, so on-time performance is not reported directly in artifact mode.
- `trips.pkl` in this repository may fail to deserialize cleanly under newer pandas versions. When that happens, the app still runs, but the recent-context chart falls back to model-generated baseline demand instead of observed historical demand.
- Saved operations-model accuracy comes from the original notebook's combined multi-output evaluation. Per-target holdout metrics for drivers, vehicles, routes, and completion were not persisted with the artifact bundle.
- Confidence scores reflect model agreement and interval tightness, not guaranteed real-world outcome probability.

## Repository structure

```text
.
|-- .streamlit/
|   `-- config.toml
|-- app.py
|-- forecasting.py
|-- requirements.txt
|-- rf_model.pkl
|-- best_model.h5
|-- trips.pkl
|-- code.ipynb
`-- denver_trips_synth.csv
```

## Installation

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the app

Start Streamlit from the project root:

```powershell
streamlit run app.py
```

Streamlit will print a local address such as:

```text
http://127.0.0.1:8501
```

## Using the app

Inside the UI, users can:

- choose a future date range or enter custom future timestamps
- switch between hourly and daily presentation
- set available drivers and vehicles
- apply a reserve staffing buffer
- define a completion target
- inspect risk windows, forecast confidence, and projected shortfalls
- export the hourly forecast results

## Inputs and outputs

### Inputs

- forecast range or custom timestamps
- available drivers
- available vehicles
- reserve buffer percentage
- completion target

### Outputs

- projected trips
- planned drivers
- planned vehicles
- projected routes
- predicted completion rate
- driver shortfall
- vehicle shortfall
- confidence score
- risk classification: `Stable`, `Watch`, `Intervene`
- downloadable hourly forecast CSV

## Technical notes

- `app.py` contains the Streamlit product surface and visualization logic.
- `forecasting.py` contains the inference-only backend for artifact loading and forecasting.
- `rf_model.pkl` is used for trip-demand prediction.
- `best_model.h5` is used for downstream operations prediction.
- `code.ipynb` remains useful for retraining and artifact regeneration, but it is no longer part of the deployed runtime path.

## Recommended artifact improvements

For a stronger deployment package, the next version should persist:

- trip model
- operations model
- feature metadata
- saved evaluation metrics
- recent observed history in a stable format such as Parquet

Recommended next steps:

- replace `trips.pkl` with `trips.parquet`
- save per-target evaluation metrics
- export a single versioned deployment bundle
- add a dedicated OTP model if on-time forecasting is required

## Research and authorship

This project was developed and researched by Mohamed Gad.

If this repository is used in derivative work, presentations, demos, or internal research references, keeping visible attribution to Mohamed Gad is recommended.

## License

This project is released under the MIT License.

See [LICENSE](LICENSE) for the full text.

## Disclaimer

This repository is provided for research, prototyping, and software development purposes. Forecast results should be interpreted as model estimates, not operational guarantees.
