# Usage

Run commands from the repository root. See the [quick start](../README.md#quick-start)
for installation and [Environments](ENVIRONMENTS.md) for runtime versions.

## Result tables and figures

```sh
python scripts/prepare_results.py
python -m jupyterlab notebooks/
```

Run notebook cells from top to bottom to recompute tables from released
predictions and source measurements. Aggregation runs on a CPU using the
305 MB result archive. Reference values are available in folded tables.
The [experiment index](EXPERIMENTS.md) links all notebook pages.

To import a local result archive:

```sh
python scripts/prepare_results.py --archive /path/to/gw-results.zip
```

Preparation verifies archive and file identities, reuses matching files,
and stops on conflicting files. The [download guide](DOWNLOADS.md) lists
fixed versions, sizes, experiment IDs and offline preparation options.

## Checkpoint inference

Install the inference environment, prepare models, then obtain the corresponding
images following [Data preparation](../data/README.md).

```sh
python -m pip install -r requirements/cuda.txt
python scripts/prepare_assets.py --experiment s4_family \
  --package-index artifacts/model_packages.json --url-index artifacts/model_urls.json
python scripts/prepare_assets.py --experiment s4_family --check
```

Models are downloaded by experiment and verified before loading. Set
`RUN_INFERENCE = True` in the notebook's **Inference** section to generate
predictions from prepared checkpoints. `device='auto'` selects CUDA when
available, otherwise CPU. New predictions are written under `output/`.

For the ImageNet-initialized checkpoints, use the matched prediction
environment described in [Environments](ENVIRONMENTS.md).

To prepare models from local archive parts:

```sh
python scripts/prepare_assets.py --experiment s4_family \
  --package-index artifacts/model_packages.json --packages-dir /path/to/model-parts
```

## Training and fitting

Run one registered training configuration:

```sh
python scripts/train_classification.py --experiment random --run-id 010 --device auto
```

Other experiments provide training or fitting cells on their notebook pages.
The distribution-fitting page provides **Region selection** and **Refit**
entries for fixed training-image regions. These require the relevant provider
images or prepared pixel blocks. Parameter, seed, input-processing and model
identities are recorded in the machine-readable configurations.

## Repository layout

| Location | Content |
|---|---|
| `src/gw/` | Method, training, inference, metrics and fitting |
| `scripts/` | Data, asset and experiment entry points |
| `configs/` | Experiment settings and seeds |
| `data/` | Provider links, fixed splits and file identities |
| `notebooks/` | Saved, executable paper results |
| `results/paper/` | Paper reference tables |
| `artifacts/` | Download and verification catalogs |
| `tests/` | Integrity and reproducibility checks |

Paper figures and their numerical sources are documented in the
[figure mapping](SUBMITTED_FIGURES.md) and [source guide](NUMERICAL_SOURCES.md).
