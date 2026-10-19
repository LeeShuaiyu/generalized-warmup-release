"""Warm-up input generators and uniform-target loss."""
from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class WarmupSpec:
    name: str
    family: str
    probability: float = 0.0
    shape: float = 1.0
    rho: float = 0.0
    target: str = 'uniform_soft'


def parse_method(name, modality='sar'):
    if name in {'erm', 'ordinary'}:
        return WarmupSpec(name, 'none')
    if name in {'gaussian', 'gaussian_warmup'}:
        return WarmupSpec(name, 'isotropic_gaussian')
    patterns = [
        (r'loggamma_warmup_p(\d+)_l(\d+(?:\.\d+)?)_r(\d+)', 'sar_scaled_loggamma_mixture'),
        (r'addgamma_p(\d+)_l(\d+(?:\.\d+)?)_rho(\d+)', 'oct_standardized_addgamma_mixture'),
        (r'mulgamma_warmup_p(\d+)_l(\d+(?:\.\d+)?)_rho(\d+)', 'standardized_mulgamma_mixture'),
    ]
    for pattern, family in patterns:
        match = re.fullmatch(pattern, name)
        if match:
            return WarmupSpec(name, family, float(match[1])/100, float(match[2]), float(match[3])/100)
    match = re.fullmatch(r'laplace_warmup_p(\d+)_rho(\d+)', name)
    if match:
        return WarmupSpec(name, 'symmetric_laplace_mixture', float(match[1])/100, 1.0, float(match[2])/100)
    raise ValueError(f'Unknown warm-up method: {name!r}')


def energy_standardize(x, eps=1e-6):
    dims = tuple(range(1, x.ndim))
    return (x-x.mean(dim=dims, keepdim=True))/x.std(dim=dims, keepdim=True, unbiased=False).clamp_min(eps)


def _gamma_like(x, shape, torch):
    concentration = torch.full_like(x, float(shape))
    return torch.distributions.Gamma(concentration, concentration).sample().clamp_min(1e-12)


def sample(spec, shape, torch, device, mean=0.0, std=1.0):
    # Preserve the RNG order: Gaussian, sample-wise gate, then residual.
    z = torch.randn(*shape, device=device)
    if spec.family == 'isotropic_gaussian':
        return z*std+mean
    if spec.family == 'none':
        raise ValueError('No warm-up inputs for standard training')
    mask_shape = (shape[0],)+(1,)*(len(shape)-1)
    mask = torch.rand(mask_shape, device=device) < spec.probability
    if spec.family == 'sar_scaled_loggamma_mixture':
        x = z*std+mean
        g = _gamma_like(x, spec.shape, torch)
        looks = torch.tensor(spec.shape, device=device, dtype=x.dtype)
        q = (torch.log(g)-(torch.special.digamma(looks)-torch.log(looks)))/torch.sqrt(torch.polygamma(1, looks).clamp_min(1e-12))
        return torch.where(mask, x+std*spec.rho*q, x)
    if spec.family == 'oct_standardized_addgamma_mixture':
        z = energy_standardize(z)
        q = energy_standardize(_gamma_like(z, spec.shape, torch)).clamp(-6, 6)
        structured = (z+spec.rho*q)/math.sqrt(1+spec.rho**2)
        return energy_standardize(torch.where(mask, structured, z))
    if spec.family == 'symmetric_laplace_mixture':
        scale = torch.tensor(1/math.sqrt(2), device=device, dtype=z.dtype)
        q = torch.distributions.Laplace(torch.zeros((), device=device, dtype=z.dtype), scale).sample(shape)
        return torch.where(mask, z*std+mean+std*spec.rho*q, z*std+mean)
    if spec.family == 'standardized_mulgamma_mixture':
        z = energy_standardize(z)
        g = _gamma_like(z, spec.shape, torch)
        structured = energy_standardize(z*((1-spec.rho)+spec.rho*g))
        return energy_standardize(torch.where(mask, structured, z))
    raise ValueError(f'Unsupported warm-up family: {spec.family}')


def uniform_soft_loss(logits, torch):
    return -torch.nn.functional.log_softmax(logits, dim=1).mean(dim=1).mean()
