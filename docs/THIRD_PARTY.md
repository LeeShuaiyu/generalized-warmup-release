# Third-party materials

Dataset images are obtained from the original providers, not covered by the
software license. Provider links and input preparation are in `data/README.md`.
Dataset-specific source identities are retained in the sample manifests.

Training and model definitions use PyTorch and torchvision. ImageNet-initialized
models inherit torchvision's corresponding architecture and initialization;
the relevant settings remain in each run configuration. The model archives
contain trained checkpoints, including the recorded initialization.

Python dependencies retain their own licenses. Redistribution of provider
images or image-derived pixel blocks requires the provider's applicable terms;
no permission is implied by this repository's code license.
