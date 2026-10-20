# Third-party materials

The project's original software and associated software documentation use
the MIT License in `LICENSE`. Dependencies and incorporated third-party
materials retain their applicable terms. Model/result licensing is listed in
[Downloads](DOWNLOADS.md).

Dataset images are obtained under their providers' terms. Provider links and
input preparation are in `data/README.md`.
Dataset-specific source identities are retained in the sample manifests.

Training and model definitions use PyTorch and torchvision. ImageNet-initialized
models inherit torchvision's corresponding architecture and initialization;
the relevant settings remain in each run configuration. The model archives
contain trained checkpoints, including the recorded initialization.

Redistribution of provider images and image-derived pixel blocks follows the
applicable provider terms.
