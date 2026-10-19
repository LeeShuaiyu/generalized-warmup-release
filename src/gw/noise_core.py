"""RGB noise-control data, sampling and model operations."""


from __future__ import annotations
import argparse
import hashlib
import math
import random
from typing import Any, Sequence
import numpy as np


def seed_all(seed: int, deterministic: bool) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic



def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little") % (2**63 - 1)



def stratified_split(labels: Sequence[int], seed: int, fractions: Sequence[float]) -> list[list[int]]:
    labels_array = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    result = [[] for _ in fractions]
    for cls in sorted(set(labels_array.tolist())):
        indices = np.flatnonzero(labels_array == cls)
        rng.shuffle(indices)
        start = 0
        for split_index, fraction in enumerate(fractions):
            end = len(indices) if split_index == len(fractions) - 1 else start + int(round(len(indices) * fraction))
            result[split_index].extend(indices[start:end].tolist())
            start = end
    for indices in result:
        rng.shuffle(indices)
    return result



def compute_train_stats(dataset: Any, indices: Sequence[int]) -> tuple[list[float], list[float]]:
    import torch
    from torchvision.transforms.functional import pil_to_tensor

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_square = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    for index in indices:
        image, _ = dataset[index]
        tensor = pil_to_tensor(image).to(torch.float64) / 255.0
        channel_sum += tensor.sum(dim=(1, 2))
        channel_square += tensor.square().sum(dim=(1, 2))
        pixel_count += tensor.shape[1] * tensor.shape[2]
    mean = channel_sum / pixel_count
    std = ((channel_square / pixel_count) - mean.square()).clamp_min(1e-12).sqrt()
    return mean.tolist(), std.tolist()



def standardized_loggamma(
    shape: Sequence[int], looks: float, generator: Any, device: Any = "cpu", dtype: Any = None
):
    import torch

    dtype = torch.float32 if dtype is None else dtype
    concentration = torch.full(tuple(shape), float(looks), device=device, dtype=dtype)
    gamma = torch._standard_gamma(concentration, generator=generator).clamp_min(1e-12) / float(looks)
    looks_tensor = torch.tensor(float(looks), device=device, dtype=dtype)
    mean = torch.digamma(looks_tensor) - torch.log(looks_tensor)
    variance = torch.polygamma(1, looks_tensor).clamp_min(1e-12)
    return (torch.log(gamma) - mean) / torch.sqrt(variance)



def target_noise_field(
    index: int,
    realization: int,
    seed: int,
    domain: str,
    looks: float,
    channels: int = 3,
    size: int = 64,
):
    import torch

    generator = torch.Generator().manual_seed(stable_seed("target_noise", seed, index, realization))
    shape = (channels, size, size)
    if domain == "clean":
        return None
    if domain == "gaussian_noisy":
        return torch.randn(shape, generator=generator)
    if domain == "loggamma_noisy":
        return standardized_loggamma(shape, looks, generator)
    raise ValueError(domain)



def add_target_noise(standardized_image: Any, field: Any, alpha: float):
    if field is None:
        return standardized_image
    return (standardized_image + float(alpha) * field) / math.sqrt(1.0 + float(alpha) ** 2)



def training_augmentation(image: Any):
    import torch
    from torchvision.transforms.functional import crop, hflip, pad

    image = pad(image, [4, 4, 4, 4], fill=0)
    top = int(torch.randint(0, 9, size=(1,)).item())
    left = int(torch.randint(0, 9, size=(1,)).item())
    image = crop(image, top, left, 64, 64)
    return hflip(image) if float(torch.rand(1).item()) < 0.5 else image



def worker_init_fn(worker_id: int) -> None:
    seed = (int(np.random.get_state()[1][0]) + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)



class EuroSATDistributionDataset:
    def __init__(
        self,
        base: Any,
        indices: Sequence[int],
        domain: str,
        mean: Sequence[float],
        std: Sequence[float],
        target_alpha: float,
        target_looks: float,
        seed: int,
        train: bool,
        noise_seed: int | None = None,
        realization: int = 0,
    ) -> None:
        import torch

        self.base = base
        self.indices = list(indices)
        self.domain = domain
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]
        self.target_alpha = float(target_alpha)
        self.target_looks = float(target_looks)
        self.seed = int(seed)
        self.noise_seed = int(seed if noise_seed is None else noise_seed)
        self.train = bool(train)
        self.realization = int(realization)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, item: int):
        from torchvision.transforms.functional import pil_to_tensor

        original_index = self.indices[item]
        image, label = self.base[original_index]
        realization = self.epoch if self.train else self.realization
        if self.train:
            image = training_augmentation(image)
        tensor = pil_to_tensor(image).to(dtype=self.mean.dtype) / 255.0
        tensor = (tensor - self.mean) / self.std
        field = target_noise_field(
            original_index,
            realization,
            self.noise_seed,
            self.domain,
            self.target_looks,
            size=tensor.shape[-1],
        )
        tensor = add_target_noise(tensor, field, self.target_alpha)
        return tensor, int(label), int(original_index)



def build_loaders(args: argparse.Namespace, context: dict[str, Any], domain: str, seed: int):
    import torch
    from torch.utils.data import DataLoader

    common = dict(
        base=context["base"], domain=domain, mean=context["mean"], std=context["std"],
        target_alpha=args.alpha, target_looks=args.target_looks,
    )
    train_set = EuroSATDistributionDataset(
        indices=context["train_indices"], train=True, seed=seed, noise_seed=seed, **common
    )
    val_set = EuroSATDistributionDataset(
        indices=context["val_indices"], train=False, realization=10_000,
        seed=seed, noise_seed=args.eval_noise_seed, **common
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, generator=generator, worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    return train_loader, val_loader



def build_model(num_classes: int):
    import torch.nn as nn
    from torchvision import models

    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model



def build_sgd_optimizer(model: Any, args: argparse.Namespace):
    import torch

    return torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )



def state_digest(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()



def make_warmup_batch(
    method: str,
    batch_size: int,
    num_classes: int,
    alpha: float,
    looks: float,
    generators: dict[str, Any],
    device: Any,
    return_audit: bool = False,
):
    import torch

    base = torch.randn((batch_size, 3, 64, 64), generator=generators["base"], device=device)
    probe = None
    if method == "gaussian":
        inputs = base
    elif method == "loggamma":
        probe = standardized_loggamma(
            base.shape, looks, generators["structure"], device=device, dtype=base.dtype
        )
        inputs = (base + float(alpha) * probe) / math.sqrt(1.0 + float(alpha) ** 2)
    else:
        raise ValueError(method)
    labels = torch.full(
        (batch_size, num_classes), 1.0 / float(num_classes), device=device, dtype=inputs.dtype
    )
    if return_audit:
        return inputs, labels, {"base": base, "probe": probe}
    return inputs, labels



def warmup_epoch(model: Any, method: str, steps: int, args: argparse.Namespace, num_classes: int, optimizer: Any, generators: dict[str, Any], device: Any) -> float:
    import torch.nn.functional as functional

    model.train()
    losses = []
    for _ in range(steps):
        inputs, targets = make_warmup_batch(
            method,
            args.batch_size,
            num_classes,
            args.alpha,
            args.warmup_looks,
            generators,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = -(targets * functional.log_softmax(logits, dim=1)).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))



def train_epoch(model: Any, loader: Any, optimizer: Any, device: Any, epoch: int) -> tuple[float, float]:
    import torch.nn.functional as functional

    loader.dataset.set_epoch(epoch)
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    for inputs, labels, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach()) * labels.numel()
        correct += int((logits.argmax(dim=1) == labels).sum())
        count += labels.numel()
    return loss_sum / count, correct / count



def collect_logits(model: Any, loader: Any, device: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    logits_parts, labels_parts, index_parts = [], [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels, indices in loader:
            logits_parts.append(model(inputs.to(device, non_blocking=True)).cpu().numpy().astype(np.float32))
            labels_parts.append(labels.numpy().astype(np.int64))
            index_parts.append(indices.numpy().astype(np.int64))
    return np.concatenate(logits_parts), np.concatenate(labels_parts), np.concatenate(index_parts)

