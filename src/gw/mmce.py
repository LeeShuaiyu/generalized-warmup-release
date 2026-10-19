"""Weighted MMCE with the original mean-CE convention and kernel bandwidth."""
import torch


def penalty(logits, labels):
    confidence, predicted = logits.softmax(1).max(1)
    correct = predicted.eq(labels)
    a, b = confidence[correct], confidence[~correct]
    if not a.numel() or not b.numel():
        return logits.sum() * 0
    kernel = lambda x, y: torch.exp(-(x[:, None] - y[None, :]).abs() / .4)
    aa = ((1-a[:, None]) * (1-a[None, :]) * kernel(a, a)).mean()
    bb = (b[:, None] * b[None, :] * kernel(b, b)).mean()
    ab = ((1-a[:, None]) * b[None, :] * kernel(a, b)).mean()
    return (aa + bb - 2*ab).clamp_min(1e-12).sqrt()


def epoch(model, loader, optimizer, device, torch, alpha=0., **kwargs):
    if kwargs.get('geometric_augmentation', False) or kwargs.get('label_smoothing', 0.) != 0.:
        raise ValueError('This MMCE protocol has no augmentation or label smoothing')
    if not 0 <= alpha:
        raise ValueError('MMCE strength must be nonnegative')
    model.train()
    total = correct = 0
    losses = ce_sum = mmce_sum = 0.
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        ce = torch.nn.functional.cross_entropy(logits, y)
        reg = penalty(logits, y) if alpha else logits.sum()*0
        loss = ce + alpha*reg
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite MMCE/CE')
        loss.backward()
        optimizer.step()
        n = len(y)
        total += n
        correct += int(logits.argmax(1).eq(y).sum())
        losses += float(loss.detach()) * n
        ce_sum += float(ce.detach()) * n
        mmce_sum += float(reg.detach()) * n
    if total == 0:
        raise ValueError('Empty training loader')
    return dict(loss=losses/total, accuracy=correct/total, ce=ce_sum/total, mmce=mmce_sum/total)
