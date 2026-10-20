# Downloads

The result ZIP and all 50 experiment-specific model TAR archives are stored
in [the asset repository](https://huggingface.co/kanadelsy/generalized-warmup-assets/tree/41026c835ed3bc2a2d293a5eb6c951ecb8b69079)
at fixed revision `41026c835ed3bc2a2d293a5eb6c951ecb8b69079`.
Preparation checks archive size/SHA256 and extracted file identities before
installation. Matching files are reused; conflicting files stop preparation.

## Results

For saved tables, figures and refitting from released inputs, inspect or
prepare the result archive without downloading model weights:

```sh
python scripts/plan_downloads.py --results-only
python scripts/prepare_results.py
```

For private access, download the ZIP with your authorized browser session:

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

For private access, obtain the listed parts in the authorized browser, keep
them together, and pass `--packages-dir /path/to/model-parts`.

## Public-access check

Once public access has been separately approved and enabled:

```sh
python scripts/check_public_downloads.py
```

This read-only check uses anonymous requests to verify public repository
metadata and the registered size/SHA256 for every archive.

Provider images are obtained separately under [their terms](../data/README.md).

## Asset licensing

Model and result archive licensing is pending author confirmation.
Provider materials retain their applicable terms in [Third-party materials](THIRD_PARTY.md).
