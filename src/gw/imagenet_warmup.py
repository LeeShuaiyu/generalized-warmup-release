"""Input and update operations used by the archived SAR ImageNet protocol."""

import numpy as np
from .classification_data import balanced_sample_weights


def make_loader(dataset, split, batch_size, seed, torch, *, balanced=False, num_workers=0, pin_memory=False):
    """Retain the original sampler and loader's distinct RNG behavior."""
    if split not in {'train', 'val', 'test', 'stats'}:
        raise ValueError('Unknown loader split')
    generator = torch.Generator().manual_seed(seed + {'stats': 101, 'train': 201, 'val': 301, 'test': 401}[split])
    options = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    if split == 'train' and balanced:
        sampler = torch.utils.data.WeightedRandomSampler(
            balanced_sample_weights(dataset.rows), len(dataset), replacement=True, generator=generator)
        # No DataLoader generator in this original branch: its base seed uses
        # the global stream, separately from the sampler's seeded stream.
        return torch.utils.data.DataLoader(dataset, sampler=sampler, **options)
    return torch.utils.data.DataLoader(dataset, shuffle=split == 'train', generator=generator, **options)


def supervised_epoch(model, loader, optimizer, device, torch, geometric_augmentation=False, label_smoothing=0.0):
    model.train()
    losses, correct, total = [], 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if geometric_augmentation:
            if bool(torch.rand((), device=x.device) < .5):
                x = torch.flip(x, dims=[-1])
            if bool(torch.rand((), device=x.device) < .5):
                x = torch.flip(x, dims=[-2])
            rotation = int(torch.randint(0, 4, (), device=x.device).item())
            if rotation:
                x = torch.rot90(x, rotation, dims=(-2, -1))
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y, label_smoothing=label_smoothing)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        correct += int((logits.argmax(1) == y).sum())
        total += len(y)
    if not total:
        raise ValueError('Empty training loader')
    return {'loss': float(np.mean(losses)), 'accuracy': correct / total, 'updates': len(losses)}


def sample(spec, shape, torch, device, mean, std, multiplier=1.0):
    if len(shape) != 4 or shape[1] != 1:
        raise ValueError('This protocol requires single-channel image batches')
    if spec.family not in {'isotropic_gaussian', 'sar_scaled_loggamma_mixture'}:
        raise ValueError('Unsupported ImageNet warm-up family')
    scale = std * multiplier
    x = torch.randn(*shape, device=device) * scale + mean
    if spec.family == 'sar_scaled_loggamma_mixture':
        concentration = torch.full_like(x, float(spec.shape))
        gamma = torch.distributions.Gamma(concentration, concentration).sample().clamp_min(1e-12)
        looks = torch.tensor(float(spec.shape), device=device, dtype=x.dtype)
        log_mean = torch.special.digamma(looks) - torch.log(looks)
        log_var = torch.polygamma(1, looks).clamp_min(1e-12)
        q = (torch.log(gamma) - log_mean) / torch.sqrt(log_var)
        structured = x + scale * float(spec.rho) * q
        # The archived runner generates the residual before this sample gate.
        mask = torch.rand((shape[0], 1, 1, 1), device=device) < float(spec.probability)
        x = torch.where(mask, structured, x)
    return x.clamp(0.0, 1.0)


def epoch(model, spec, steps, batch_size, image_size, num_classes, optimizer,
          device, torch, mean, std, multiplier=1.0):
    if steps < 1:
        raise ValueError('Warm-up requires at least one update')
    model.train()
    target = torch.full((batch_size, num_classes), 1.0 / num_classes, device=device)
    losses = []
    for _ in range(steps):
        x = sample(spec, (batch_size, 1, image_size, image_size), torch, device, mean, std, multiplier)
        optimizer.zero_grad(set_to_none=True)
        loss = -(target * torch.nn.functional.log_softmax(model(x), dim=1)).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {'loss': sum(losses) / len(losses), 'updates': steps}


def configure_determinism(torch):
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    for name, enabled in [('enable_flash_sdp', False), ('enable_mem_efficient_sdp', False),
                          ('enable_math_sdp', True)]:
        function = getattr(torch.backends.cuda, name, None)
        if function:
            function(enabled)
