# Generalized Warm-up

**Generalized warm-up for confidence calibration of deep neural network classifiers**

[![Models: Hugging Face](https://img.shields.io/badge/Models-Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/kanadelsy/generalized-warmup-assets)
[![Results: 305 MB](https://img.shields.io/badge/Results-305%20MB-2EA44F?style=flat-square)](docs/DOWNLOADS.md#results)
[![Notebooks: Jupyter](https://img.shields.io/badge/Notebooks-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)](docs/EXPERIMENTS.md)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](docs/ENVIRONMENTS.md)
[![Code license: MIT](https://img.shields.io/badge/Code-MIT-7500A8?style=flat-square)](LICENSE)

Shuai-Yu Li · Xin-Chao Wang · Yu-Ying Song · Si-Wei Chen

[Overview](#overview) · [Quick start](#quick-start) · [Results and models](#results-and-models) · [Citation](#citation) · [License](#license)

## Overview

Generalized warm-up prepares a classifier with uniform targets before supervised
learning. Its inputs combine a Gaussian base with a statistics-guided component
constructed from approximately homogeneous regions of training images.
This repository provides the method implementation, experiment configurations,
model checkpoints and executable notebooks for the paper.

![Generalized warm-up: fluctuation statistics, input construction and uniform-target warm-up.](assets/submitted/figure4.png)

*Method overview. Training-image fluctuation statistics guide input construction;
uniform-target warm-up precedes supervised learning on labelled data.*

## Quick start

Use **Python 3.12**. The following commands install the notebook environment,
download the result archive and open the paper notebooks:

```sh
git clone https://github.com/LeeShuaiyu/generalized-warmup-release.git
cd generalized-warmup-release
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[notebook]'
python scripts/prepare_results.py
python -m jupyterlab notebooks/
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate it
with `.venv\Scripts\activate` in Command Prompt.

Run notebook cells to recompute tables from the released predictions and source
measurements on a CPU. Start with the [random-initialization results](notebooks/main_random.ipynb),
[ImageNet-initialization results](notebooks/main_pretrained.ipynb), or
[paper figures](notebooks/paper_figures.ipynb).

See [all experiments](docs/EXPERIMENTS.md), [usage](docs/USAGE.md) and
[environments](docs/ENVIRONMENTS.md) for additional workflows.

## Results and models

The [Hugging Face asset repository](https://huggingface.co/kanadelsy/generalized-warmup-assets)
contains the **305 MB result archive** and **388 model checkpoints in 50 archives**.
Preparation commands use fixed download versions and verify file sizes and hashes.
Model downloads can be selected by experiment:

```sh
python scripts/plan_downloads.py --experiment s4_family
python scripts/prepare_assets.py --experiment s4_family \
  --package-index artifacts/model_packages.json --url-index artifacts/model_urls.json
```

For checkpoint inference, follow [Usage](docs/USAGE.md#checkpoint-inference)
and [Data preparation](data/README.md). The [download guide](docs/DOWNLOADS.md)
covers archive sizes, fixed versions and offline preparation.

## Citation

Software citation metadata are available in [CITATION.cff](CITATION.cff).

```bibtex
@software{li2026generalizedwarmup,
  author = {Li, Shuai-Yu and Wang, Xin-Chao and Song, Yu-Ying and Chen, Si-Wei},
  title = {Generalized warm-up for confidence calibration of deep neural network classifiers},
  year = {2026},
  url = {https://github.com/LeeShuaiyu/generalized-warmup-release}
}
```

## License

Original software and associated documentation use [MIT](LICENSE).
Contributor-owned model weights use [MIT](LICENSE-MODELS); original numerical
result data use [CC BY 4.0](LICENSE-RESULTS). See [license scope and attribution](LICENSES.md)
and [third-party terms](docs/THIRD_PARTY.md).
