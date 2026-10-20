# Environments

## Notebook results

Python 3.12; `python -m pip install -e '.[notebook]'`.
Result-table aggregation runs on a CPU. Launch Jupyter from the project root
after importing the result archive.

## Checkpoint inference

The CUDA inference environment is pinned in `requirements/cuda.txt`
(Python 3.12, PyTorch 2.3, CUDA 12.1). CPU can also be selected explicitly;
`device='auto'` uses CUDA when it is available.

Post-hoc calibration uses IEEE FP32 inference with cuDNN and matrix-multiply
TF32 disabled, batch size 64 and deterministic cuDNN. The same precision is
used for all six validation/test model pairs. `RUN_REFIT` reuses the displayed
model-derived predictions; `--source archived` explicitly selects archived
logits in `scripts/refit_calibration.py`.

The ImageNet-initialized SAR checkpoints have a separate recorded prediction
runtime, pinned in `requirements/replay-cuda.txt` (Python 3.10, PyTorch 2.11,
CUDA 12.8). Create a separate prediction environment:

```sh
python3.10 -m venv .venv-prediction
.venv-prediction/bin/python -m pip install -r requirements/replay-cuda.txt
export GW_INFERENCE_PYTHON="$PWD/.venv-prediction/bin/python"
# Start Jupyter from the Python 3.12 notebook environment:
python -m jupyterlab notebooks/
```

The ImageNet page invokes this interpreter for checkpoint inference, then
returns the numerical results to the notebook.
For a direct prediction-only call:

```sh
.venv-prediction/bin/python scripts/infer_pretrained.py --device auto --no-report
```

On Windows, use `.venv-prediction\Scripts\python.exe` and set
`$env:GW_INFERENCE_PYTHON` to its absolute path before launching Jupyter.
Dataset TIFF preparation additionally requires the `data` extra documented
in `data/README.md`. CUDA installation requires a compatible NVIDIA driver.
