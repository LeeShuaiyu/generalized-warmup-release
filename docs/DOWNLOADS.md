# Downloads

The original result ZIP is stored at the fixed Hugging Face revision.
Replacement model parts are separated by experiment and remain local until
a real uploaded revision is verified. No new model URL is inferred.
Preparation checks archive size/SHA256 and extracted file identities before
installation; existing conflicting files are not overwritten.

## Results

For saved tables, figures and refitting from released inputs, inspect or
prepare the result archive without downloading model weights:

```sh
python scripts/plan_downloads.py --results-only
python scripts/prepare_results.py
```

This candidate still references a private asset repository. For private
access, download the ZIP with your authorized browser session and run:

```sh
python scripts/prepare_results.py --archive /path/to/gw-results.zip
```

## Models

| Experiment ID | Checkpoints |
|---|---|
| `random` | Random-initialization comparisons |
| `pretrained` | ImageNet-initialization comparisons |
| `training_compat` | Calibration combinations |
| `cifar` | CIFAR-100 |
| `noise` | Optical noise domains |
| `s4_family` | Warm-up constructions |
| `s7_length` | Warm-up duration |

Choose an experiment rather than fetching all 33.93 GB:

```sh
python scripts/plan_downloads.py --experiment s4_family --include-results
python scripts/prepare_assets.py --experiment s4_family \
  --package-index artifacts/model_packages.json --url-index artifacts/model_urls.json
```

The plan reports selected-model bytes separately from required archive bytes.
Archive parts are separated by experiment; preparation extracts only the selected models. A valid cache is reused. Model files and archive caches
need additional disk space; preparation keeps an 8 GiB reserve.

For private access, obtain the listed parts in the authorized browser, keep
them together, and pass `--packages-dir /path/to/model-parts` instead of
`--url-index`. The Python scripts do not read browser cookies or create tokens.

## Public-access check

Once public access has been separately approved and enabled:

```sh
python scripts/check_public_downloads.py
```

This read-only gate uses anonymous requests and requires public repository
metadata plus exact size/SHA256 for every formal file. It does not change
repository visibility or claim a full download/restore test. Before approval,
denied access is expected, not evidence of another upload failure.

Provider images are obtained separately under [their terms](../data/README.md).
The model/result hosting does not imply a Zenodo DOI or a license grant.

New experiment-specific model parts are not uploaded yet. `artifacts/model_urls.json` is empty deliberately; use authorized local archives. Rebind links only after checking the actual uploaded revision and hashes.
