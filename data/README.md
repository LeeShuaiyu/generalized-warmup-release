# Data preparation

The fixed sample lists and splits are in `manifests/`. Processing uses these
registered splits. `files.json` records each processed file's SHA256.

## Sources

The paper cites the original providers:

- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [EuroSAT](https://github.com/phelber/EuroSAT)
- [OpenSARShip 2.0](https://opensar.sjtu.edu.cn/DataAndCodes.html)
- [Schwegmann SAR ship dataset](https://gitlab.com/sarmaps/sar-ship-dataset)
- [FUSAR-Ship](https://www.radars.ac.cn/web/data/getData?dataType=FUSAR)
- [OCTDL](https://doi.org/10.5281/zenodo.10839556)

Obtain third-party images under the provider's terms. The commands below can
convert extracted provider files or import an existing processed tree.

## Original files

Install the TIFF conversion dependencies with `pip install -e '.[data]'`.
Unpack the provider download, then supply the directory indicated below:

| Dataset ID | `--raw-dir` contents | Conversion |
|---|---|---|
| `sar` | `json/` containing `ship_positives.json`, `false_positives.json`, `true_negatives.json` | Native `windowfu` arrays saved as float32 NPY |
| `opensar` | `OpenSARShip2/` containing the original scene ZIP files | Registered GRD/SDV `Patch_Cal` TIFF members decoded to float32 NPY |
| `oct` | `OCTDL/` containing the class directories and original JPEG files | Byte-for-byte JPEG copy |

From the project root:

```sh
python scripts/prepare_original.py --dataset sar --raw-dir /path/to/json
python scripts/prepare_original.py --dataset opensar --raw-dir /path/to/OpenSARShip2
python scripts/prepare_original.py --dataset oct --raw-dir /path/to/OCTDL
```

These commands convert and install the registered files in one step. Existing
complete, verified inputs are reused without conversion or download. Check
installed inputs and list the required provider filenames without writing:

```sh
python scripts/prepare_original.py --dataset opensar --raw-dir /path/to/OpenSARShip2 --check
```

OCTDL also has a pinned public download from the provider's
[v2 Zenodo record](https://zenodo.org/records/10839556), under its CC BY 4.0
license. Download the original ZIP, verify its published size/checksum and
each registered image's SHA256, then prepare the inputs:

```sh
python scripts/prepare_original.py --dataset oct --download
# Or reuse a previously downloaded official ZIP:
python scripts/prepare_original.py --dataset oct --archive /path/to/OCTDL.zip
```

The archive is cached in `data/downloads/`; unpacking writes a new directory
under `output/provider_data/`. Only the registered JPEGs are copied; ZIP links
and unsafe paths are rejected. Other providers retain their own acquisition
procedures.

Each conversion creates `output/data_conversion/<run>/files/` and a conversion
record. All generated files must match the registered SHA256. Conversion uses
the fixed splits, creates new output directories and retains at least 8 GiB
of free space. Conversion-only and separate import remain available:

```sh
python scripts/convert_data.py --dataset sar --raw-dir /path/to/json
python scripts/prepare_data.py --dataset sar --from-dir output/data_conversion/<run>/files
```

Use the corresponding dataset ID for the other conversions. `--split test`
converts a registered split; `--limit 10` selects a diagnostic subset.
Inspect sample identities and group-disjoint splits with
`python scripts/convert_data.py --dataset sar --audit`.

OpenSAR conversion retains the TIFF-first decoder and its grayscale fallback;
the chosen decoder is recorded per sample. Percentile scaling and resizing
are applied during model-input preparation.

## Fixed processed inputs

For the random-initialization experiments and OCTDL Swin evaluation, supply
a directory with the following relative paths:

| Dataset ID | Path within the supplied directory |
|---|---|
| `sar` | `data/processed/classification/sar_dataset/arrays_window/<component>/<sample>.npy` |
| `opensar` | `data/processed/classification/opensarship2_grd3/arrays/<class>/<sample>.npy` |
| `oct` | `data/octdl/images/OCTDL/<class>/<sample>.jpg` |
| `fusar` | Original extracted class directories, e.g. `Cargo/CargoShip/<sample>.tiff` |

The complete filenames are in the corresponding manifest. From the project root:

```sh
python scripts/prepare_data.py --dataset sar --from-dir /path/to/sar-source
python scripts/prepare_data.py --dataset opensar --from-dir /path/to/opensar-source
python scripts/prepare_data.py --dataset oct --from-dir /path/to/oct-source
python scripts/prepare_data.py --dataset fusar --from-dir /path/to/FUSAR-Ship/release
```

Files are copied into `data/processed/<dataset>/` only after matching their
registered hashes. Existing matching files are reused; mismatches stop
preparation. Check the prepared evaluation inputs:

```sh
python scripts/prepare_data.py --dataset sar --split test --check
python scripts/prepare_data.py --dataset opensar --split test --check
python scripts/prepare_data.py --dataset oct --split test --check
```

`src/gw/classification_data.py` contains the exact model-input transformation:
SAR arrays use the recorded percentile scaling; stored OCTDL images are decoded
as grayscale. Resizing uses the corresponding quantization and bilinear rule.
The archived ImageNet training input transform is in `src/gw/imagenet_data.py`.

FUSAR training uses the fixed 393-image five-class subset in `fusar_fit.csv`
(188 train, 48 validation, 157 test), shared with the region-input registry.
The subset uses this study's fixed group-disjoint split. TIFFs are decoded as grayscale and resized by
the recorded model-input rule.

## Homogeneous regions

The fitting experiment uses training images at native resolution, before model
resizing. SAR selection scores non-overlapping 16×16 blocks using brightness,
variation, saturation and quadrant/sub-block consistency, retaining the lowest
valid score per image. OCT selection additionally checks the dark background's
distance from the image-derived tissue boundary and its surrounding context;
the lowest valid score is retained per patient group. Selection is based on these image statistics.

```sh
python scripts/select_regions.py --dataset sar
python scripts/select_regions.py --dataset opensar
python scripts/select_regions.py --dataset oct
python scripts/select_regions.py --dataset fusar --source-dir /path/to/FUSAR-Ship
```

The first three commands use the prepared data above. For FUSAR, supply the
extracted provider class directories (`Cargo/`, etc.); `fusar_fit.csv` fixes the
sample set and `region_files.json` checks the original image bytes.
`configs/region_selection.json` binds the datasets and `src/gw/region_selection.py`
contains the exact thresholds and tie rules. Each run writes a new bank,
coordinates, image audit and pixel-by-pixel comparison under `output/regions/`.
Refit a successfully rebuilt bank:

```sh
python scripts/refit_distributions.py --dataset sar --region-run sar=output/regions/<run>
```

The fitting notebook opens with saved results and region checks; `RUN_SELECTION`
rebuilds the regions, while `RUN_REFIT` repeats distribution estimation.
