# Generalized Warm-up

Code and paper notebooks for *Generalized warm-up for confidence calibration
of deep neural network classifiers*.

## Results

Open a notebook to view its saved tables and figures. Each page contains the
paper reference, reproduced results, and the corresponding executable steps.

| Experiment | Notebook |
|---|---|
| Random initialization | [Main results](notebooks/main_random.ipynb) |
| ImageNet initialization | [Main results](notebooks/main_pretrained.ipynb) |
| Calibration combinations | [Main results](notebooks/main_compatibility.ipynb) |
| Initialization bias | [Mechanism](notebooks/s2_initialization.ipynb) |
| Directional response | [Mechanism](notebooks/s2_direction.ipynb) |
| CIFAR-100 | [Three and ten seeds](notebooks/cifar100.ipynb) |
| Optical noise domains | [Results and paired interaction](notebooks/s3_noise.ipynb) |
| Homogeneous-region distributions | [Distribution fitting](notebooks/s4_fit.ipynb) |
| Warm-up constructions | [Distribution comparison](notebooks/s4_family.ipynb) |
| Warm-up duration | [Duration comparison](notebooks/s7_length.ipynb) |
| Paper figures | [Four submitted figures](notebooks/paper_figures.ipynb) |

The complete metric tables are included within the main pages. Settings and
sample splits are available in `configs/` and `data/manifests/`.

## Start

Use Python 3.12 or later. From the project root:

```sh
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -e '.[notebook]'
python scripts/prepare_results.py
python -m jupyterlab notebooks/
```

Run the notebook cells from top to bottom to recompute its tables from the
released predictions and source measurements. This step does not require a
GPU or the original image datasets. Reference values are available in folded tables.

The result archive uses the fixed download catalog in `artifacts/downloads.json`.
An already downloaded archive can be imported with
`python scripts/prepare_results.py --archive /path/to/gw-results.zip`.
Existing files are checked and reused; conflicting files are not replaced.

The result catalog points to an existing fixed Hugging Face commit. New experiment-specific model archives have been prepared locally; their URL map is intentionally empty until they are uploaded and a real fixed revision is verified. This candidate's
asset repository is private; anonymous download is not yet enabled. Until
public access is approved, download through your authorized browser session
and use the local archive/parts options below. Browser login is not shared
with the Python scripts.

To inspect the exact files and transfer size without downloading:

```sh
python scripts/plan_downloads.py --results-only
python scripts/plan_downloads.py --experiment s4_family
```

Saved-table recomputation needs only the 305 MB result archive, not model
weights. Model preparation downloads only archive parts containing the selected
experiment and imports only its models. Each new model part contains exactly one experiment. The plan reports full archive sizes, not just the selected checkpoint bytes.

## Checkpoint inference

Install the inference environment, prepare models, then obtain the corresponding
images following [Data preparation](data/README.md).

```sh
python -m pip install -r requirements/cuda.txt
python scripts/prepare_assets.py --experiment s4_family \
  --package-index artifacts/model_packages.json --packages-dir /path/to/model-parts
python scripts/prepare_assets.py --experiment s4_family --check
```

Models are downloaded by experiment and verified before loading. Set
`RUN_INFERENCE = True` in the notebook's **Inference** section to run its
prepared checkpoints. `device='auto'` selects CUDA when available, otherwise
CPU. New predictions are written under `output/` and do not replace the
released results. Use the Inference section to generate new model predictions.

For the ImageNet page, use the matched prediction environment described in
[Environments](docs/ENVIRONMENTS.md).

An offline model-parts directory is also supported:

```sh
python scripts/prepare_assets.py --experiment s4_family \
  --package-index artifacts/model_packages.json --packages-dir /path/to/model-parts
```

Details and experiment IDs are in [Downloads](docs/DOWNLOADS.md).

## Training and fitting

Run one registered training configuration:

```sh
python scripts/train_classification.py --experiment random --run-id 010 --device auto
```

Other experiments provide training or fitting cells on their notebook pages.
The distribution-fitting page provides **Region selection** and **Refit**
entries for the fixed training-image regions. These require the relevant
provider images or prepared pixel blocks. Parameter, seed, input-processing,
and model identities remain in the machine-readable configurations.

## Files

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

Model weights and the result archive are distributed separately from the
GitHub code. Original third-party datasets retain their providers' terms.

## Citation

Author and software metadata are in [CITATION.cff](CITATION.cff).

## Submission alignment update (19 October 2026)

The default paper gallery uses the four figures submitted on 18 October 2026. See [figure mapping](docs/SUBMITTED_FIGURES.md) and [numerical-source policy](docs/NUMERICAL_SOURCES.md). Old quantitative component IDs are preserved separately. New model parts are local candidates, not yet published; use the local `--packages-dir` workflow. No new anonymous download availability is claimed.
