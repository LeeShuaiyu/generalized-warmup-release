"""Grayscale quantization used by the archived ImageNet training runner."""
import numpy as np
from .classification_data import ManifestDataset


class ImageDataset(ManifestDataset):
    def __init__(self, rows, split, *, image_size, data_root, limit=None, seed=0):
        # Read the native grayscale array before the source runner's explicit
        # uint8 round-trip, including when its dimensions already match.
        super().__init__(rows, split, image_size=None, data_root=data_root, limit=limit, seed=seed)
        self.target_size = image_size

    def __getitem__(self, index):
        from PIL import Image
        x, y = super().__getitem__(index)
        array = np.nan_to_num(x.numpy()[0], nan=0.0, posinf=1.0, neginf=0.0)
        image = Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).astype('uint8'), mode='L')
        if image.size != (self.target_size, self.target_size):
            image = image.resize((self.target_size, self.target_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return self.torch.from_numpy(array[None]).float(), y
