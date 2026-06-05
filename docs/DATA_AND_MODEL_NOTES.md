# Data and Model Notes

This repository includes source code, selected result summaries, and documentation.

It does not include:

- raw CNDC response caches,
- locally processed training datasets,
- trained model binaries,
- private file paths from the development machine.

The demand forecaster expects:

- `models/bolivia_demand_model.pkl`
- `models/bolivia_demand_features.pkl`
- optionally `data/processed/cndc_ml_dataset.csv` for retraining/backtesting scripts.

The collector can rebuild CNDC-derived datasets from the public CNDC endpoints used by the project. Generated files are written under `data/` and `runs/`, which are ignored by Git by default.

The monetary values in the reports are cost-proxy estimates for research comparison. They are not audited national system savings. Dispatch simulations are capacity-aware, but they do not yet include full unit commitment, hydrology, outage, ramp-rate, or power-flow constraints.
