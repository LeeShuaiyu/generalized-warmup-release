"""Uniform-target warm-up and supervised optimization stages."""
from .warmup import sample, uniform_soft_loss


def apply_geometric_augmentation(x, torch):
    if float(torch.rand(())) < 0.5:
        x = torch.flip(x, dims=(-1,))
    if float(torch.rand(())) < 0.5:
        x = torch.flip(x, dims=(-2,))
    k = int(torch.randint(0, 4, ()).item())
    return torch.rot90(x, k, dims=(-2, -1)) if k else x


def supervised_epoch(model, loader, optimizer, device, torch, geometric_augmentation=False, label_smoothing=0.0):
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError('label_smoothing must be in [0, 1)')
    model.train()
    total_loss = total_correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if geometric_augmentation:
            x = apply_geometric_augmentation(x, torch)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y, label_smoothing=label_smoothing)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())*len(y)
        total_correct += int((logits.argmax(1) == y).sum())
        total += len(y)
    if not total:
        raise ValueError('Empty training loader')
    return {'loss': total_loss/total, 'accuracy': total_correct/total}


def warmup_epoch(model, spec, steps, batch_size, image_size, num_classes, optimizer, device, torch, mean=0.0, std=1.0, input_channels=1):
    if steps < 1:
        raise ValueError('Warm-up requires at least one update')
    model.train()
    total = 0.0
    for _ in range(steps):
        x = sample(spec, (batch_size, input_channels, image_size, image_size), torch, device, mean, std)
        optimizer.zero_grad(set_to_none=True)
        loss = uniform_soft_loss(model(x), torch)
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
    return {'loss': total/steps, 'updates': steps}


def collect_logits(model, loader, device, torch):
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits.append(model(x.to(device)).float().cpu())
            labels.append(y.cpu())
    if not logits:
        raise ValueError('Empty evaluation loader')
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def estimate_image_stats(loader, device, torch):
    count = 0
    total, squares = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            count += x.numel()
            total += x.sum()
            squares += (x*x).sum()
    if not count:
        raise ValueError('Empty statistics loader')
    mean = total/count
    variance = (squares/count-mean*mean).clamp_min(1e-8)
    return float(mean.cpu()), float(torch.sqrt(variance).cpu())


def set_warmup_trainability(model, head_only):
    prefixes = ('fc.', 'classifier.', 'head.', 'net.fc.', 'net.classifier.', 'net.head.')
    names = []
    for name, param in model.named_parameters():
        param.requires_grad = not head_only or name.startswith(prefixes)
        if param.requires_grad:
            names.append(name)
    return names
