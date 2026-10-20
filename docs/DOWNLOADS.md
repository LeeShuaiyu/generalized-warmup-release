# Downloads

The result ZIP and all 50 experiment-specific model TAR archives are stored
in [the asset repository](https://huggingface.co/kanadelsy/generalized-warmup-assets/tree/b5655af5c6800f9b853fe961a2b0f41aaf5d0eaf)
at fixed revision `b5655af5c6800f9b853fe961a2b0f41aaf5d0eaf`.
Preparation checks archive size/SHA256 and extracted file identities before
installation. Matching files are reused; conflicting files stop preparation.

## Results

For saved tables, figures and refitting from released inputs, inspect or
prepare the result archive without downloading model weights:

```sh
python scripts/plan_downloads.py --results-only
python scripts/prepare_results.py
```

To import a local result archive:

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

Select the experiment to prepare:

```sh
python scripts/plan_downloads.py --experiment s4_family --include-results
python scripts/prepare_assets.py --experiment s4_family \
  --package-index artifacts/model_packages.json --url-index artifacts/model_urls.json
```

The plan reports selected-model bytes separately from required archive bytes.
Archive parts are separated by experiment; preparation extracts the selected
models and reuses a valid cache. Allow space for downloaded archives, extracted
models and a temporary copy of the largest file, plus an 8 GiB reserve.
The preparation commands support APFS, NTFS, Linux filesystems and exFAT.

For offline preparation, keep the listed parts together and pass
`--packages-dir /path/to/model-parts`.

## Public-access check

Verify anonymous access to the fixed version:

```sh
python scripts/check_public_downloads.py
```

This read-only check uses anonymous requests to verify public repository
metadata and the registered size/SHA256 for every archive.

Provider images are obtained separately under [their terms](../data/README.md).

## Asset licensing

Contributor-owned model weights use [MIT](../LICENSE-MODELS). Original
numerical result data use [CC BY 4.0](../LICENSE-RESULTS). File scope and
attribution are listed in [Licenses](../LICENSES.md).
Provider materials retain their applicable terms in [Third-party materials](THIRD_PARTY.md).
