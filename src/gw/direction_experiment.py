#!/usr/bin/env python3
"""Controlled real-optimization test of directional residual persistence.

This is deliberately separate from the closed-form residual simulation.  It
uses synthetic, Haar-rotated 8-D binary data, but both stages update an
actual classifier by mini-batch softmax cross-entropy SGD:

* ``u`` is the semantic direction.
* ``w`` is label-invariant and well covered by supervised training.
* ``v`` is label-invariant and has configurable supervised coverage.
* warm-up has equal step/batch budgets and either targets ``v``, targets
  ``w``, covers both isotropically, or receives null inputs (no warm-up).

The primary response is the centered-logit directional derivative

    r_t(a)^2 = E_x || P J_z(x; theta_t) a ||_2^2,

where P centers the two logits.  For the linear model this is exact and
independent of x; for the MLP the exact tanh Jacobian is averaged over a fixed
reference set.  No directional decay law is prescribed: r changes only
through the CE/SGD parameter updates.

Outputs are tidy CSV plus JSON metadata.  This script intentionally produces
no final figures so plotting decisions can be made after the frozen experiment
has been inspected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np


METHODS = ("targeted_v", "wrong_w", "isotropic_wv", "no_warmup")
DIRECTIONS = ("u_semantic", "w_strong_nuisance", "v_weak_nuisance")


class RNGRole(IntEnum):
    """Stable, explicit namespaces for every stochastic role in the experiment."""

    HAAR_BASIS = 1
    WARMUP_DATA = 2
    SUPERVISED_DATA = 3
    EVALUATION_DATA = 4
    MODEL_INITIALIZATION = 5
    BOOTSTRAP_G1 = 101
    BOOTSTRAP_G2 = 102
    BOOTSTRAP_G3_MEAN = 103
    BOOTSTRAP_G3_MEDIAN = 104
    BOOTSTRAP_G4_SLOPE = 105
    BOOTSTRAP_G4_RETENTION = 106
    BOOTSTRAP_ANGLE_MEDIAN = 107


RNG_SUITE_IDS = {
    "primary": 1,
    "coverage_horizon": 2,
    "angle_dose": 3,
    "iid_robustness": 4,
}
RNG_MODE_IDS = {"smoke": 1, "formal": 2}
RNG_ARCHITECTURE_IDS = {"shared": 0, "linear": 1, "mlp_tanh": 2}
RNG_GLOBAL_SEED_INDEX = 0xFFFFFFFF
RNG_NAMESPACE_VERSION = "hierarchical-seedsequence-v1"


@dataclass(frozen=True)
class ExperimentConfig:
    protocol_version: str = "real-ce-directional-residual-v3-hierarchical-rng"
    suite_name: str = "primary"
    rng_mode: str = "formal"
    seed_base: int = 20270810
    seeds: int = 100
    mlp_seeds: int = 30
    input_dim: int = 8
    classes: int = 2
    semantic_mean: float = 1.0
    semantic_std: float = 1.0
    train_var_w: float = 1.0
    coverage_v_grid: tuple[float, ...] = (0.10,)
    primary_coverage_v: float = 0.10
    reference_var_w: float = 1.0
    reference_var_v: float = 1.0
    shifted_var_v: float = 4.0
    warmup_q_high: float = 4.0
    warmup_q_low: float = 0.25
    warmup_steps: int = 500
    supervised_horizons: tuple[int, ...] = (0, 10, 25, 50, 100, 250, 500, 1000)
    primary_horizon: int = 1000
    batch_size: int = 2048
    eval_size: int = 8192
    linear_init_scale: float = 0.05
    linear_lr: float = 0.01
    mlp_hidden: int = 32
    mlp_lr: float = 0.001
    mlp_batch_size: int = 512
    mlp_warmup_steps: int = 1000
    mlp_supervised_horizons: tuple[int, ...] = (0, 10, 25, 50, 100, 250, 500, 1000, 2000)
    mlp_primary_horizon: int = 2000
    mlp_probe_size: int = 4096
    trace_stride: int = 25
    architectures: tuple[str, ...] = ("linear", "mlp_tanh")
    methods: tuple[str, ...] = METHODS
    log_epsilon: float = 1e-12
    bootstrap_repetitions: int = 10_000
    intervention_alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0)
    strict_finite_batch_label_invariance: bool = True

    @property
    def max_supervised_steps(self) -> int:
        return max(self.supervised_horizons)

    def seeds_for(self, architecture: str) -> int:
        return self.seeds if architecture == "linear" else self.mlp_seeds

    def batch_size_for(self, architecture: str) -> int:
        return self.batch_size if architecture == "linear" else self.mlp_batch_size

    def warmup_steps_for(self, architecture: str) -> int:
        return self.warmup_steps if architecture == "linear" else self.mlp_warmup_steps

    def horizons_for(self, architecture: str) -> tuple[int, ...]:
        return self.supervised_horizons if architecture == "linear" else self.mlp_supervised_horizons

    def primary_horizon_for(self, architecture: str) -> int:
        return self.primary_horizon if architecture == "linear" else self.mlp_primary_horizon


def smoke_config() -> ExperimentConfig:
    return replace(
        ExperimentConfig(),
        rng_mode="smoke",
        seed_base=20260810,
        seeds=4,
        mlp_seeds=3,
        coverage_v_grid=(0.10,),
        primary_coverage_v=0.10,
        warmup_steps=50,
        supervised_horizons=(0, 10, 25, 50, 100),
        primary_horizon=100,
        batch_size=128,
        eval_size=512,
        mlp_hidden=16,
        mlp_batch_size=128,
        mlp_warmup_steps=50,
        mlp_supervised_horizons=(0, 10, 25, 50, 100),
        mlp_primary_horizon=100,
        trace_stride=10,
        bootstrap_repetitions=1_000,
    )


def suite_config(mode: str, suite: str) -> ExperimentConfig:
    cfg = smoke_config() if mode == "smoke" else ExperimentConfig()
    if suite == "primary":
        return cfg
    if suite == "coverage_horizon":
        if mode == "smoke":
            return replace(
                cfg,
                suite_name="coverage_horizon",
                architectures=("linear",),
                seeds=3,
                coverage_v_grid=(0.0, 0.10, 1.0),
                supervised_horizons=(0, 2, 5, 10, 25),
                primary_horizon=10,
            )
        return replace(
            cfg,
            suite_name="coverage_horizon",
            architectures=("linear",),
            seeds=50,
            coverage_v_grid=(0.0, 0.03, 0.10, 0.30, 1.0),
            supervised_horizons=(0, 50, 100, 250, 500, 1000, 2000, 5000),
            primary_horizon=1000,
        )
    if suite == "angle_dose":
        angle_methods = (
            "targeted_v",
            "angle_22.5",
            "angle_45",
            "angle_67.5",
            "wrong_w",
        )
        return replace(
            cfg,
            suite_name="angle_dose",
            architectures=("linear",),
            seeds=3 if mode == "smoke" else 50,
            coverage_v_grid=(cfg.primary_coverage_v,),
            methods=angle_methods,
        )
    if suite == "iid_robustness":
        return replace(
            cfg,
            suite_name="iid_robustness",
            architectures=("linear",),
            seeds=3 if mode == "smoke" else 30,
            coverage_v_grid=(cfg.primary_coverage_v,),
            strict_finite_batch_label_invariance=False,
        )
    raise ValueError(suite)


def rng_entropy_tuple(
    cfg: ExperimentConfig,
    seed: int,
    role: RNGRole,
    architecture: str,
    substream: int = 0,
) -> tuple[int, int, int, int, int, int, int]:
    """Return a collision-resistant, fully labelled SeedSequence namespace."""
    if cfg.suite_name not in RNG_SUITE_IDS:
        raise ValueError(f"Unknown RNG suite namespace: {cfg.suite_name}")
    if architecture not in RNG_ARCHITECTURE_IDS:
        raise ValueError(f"Unknown RNG architecture namespace: {architecture}")
    if cfg.rng_mode not in RNG_MODE_IDS:
        raise ValueError(f"Unknown RNG mode namespace: {cfg.rng_mode}")
    values = (
        int(cfg.seed_base),
        int(RNG_MODE_IDS[cfg.rng_mode]),
        int(seed),
        int(role),
        int(RNG_SUITE_IDS[cfg.suite_name]),
        int(RNG_ARCHITECTURE_IDS[architecture]),
        int(substream),
    )
    if any(value < 0 for value in values):
        raise ValueError(f"SeedSequence entropy values must be non-negative: {values}")
    return values


def hierarchical_rng(
    cfg: ExperimentConfig,
    seed: int,
    role: RNGRole,
    architecture: str,
    substream: int = 0,
) -> np.random.Generator:
    entropy = rng_entropy_tuple(cfg, seed, role, architecture, substream)
    return np.random.default_rng(np.random.SeedSequence(entropy))


def rng_stream_specs(cfg: ExperimentConfig, mode: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    data_roles = (
        RNGRole.HAAR_BASIS,
        RNGRole.WARMUP_DATA,
        RNGRole.SUPERVISED_DATA,
        RNGRole.EVALUATION_DATA,
        RNGRole.MODEL_INITIALIZATION,
    )
    for architecture in cfg.architectures:
        for seed in range(cfg.seeds_for(architecture)):
            for role in data_roles:
                specs.append(
                    {
                        "mode": mode,
                        "suite": cfg.suite_name,
                        "architecture": architecture,
                        "seed": seed,
                        "role": role.name,
                        "role_id": int(role),
                        "substream": 0,
                    }
                )
        for role in (RNGRole.BOOTSTRAP_G1, RNGRole.BOOTSTRAP_G2):
            specs.append(
                {
                    "mode": mode,
                    "suite": cfg.suite_name,
                    "architecture": architecture,
                    "seed": RNG_GLOBAL_SEED_INDEX,
                    "role": role.name,
                    "role_id": int(role),
                    "substream": 0,
                }
            )
    if "linear" in cfg.architectures:
        for role in (
            RNGRole.BOOTSTRAP_G3_MEAN,
            RNGRole.BOOTSTRAP_G3_MEDIAN,
            RNGRole.BOOTSTRAP_G4_SLOPE,
            RNGRole.BOOTSTRAP_G4_RETENTION,
        ):
            specs.append(
                {
                    "mode": mode,
                    "suite": cfg.suite_name,
                    "architecture": "linear",
                    "seed": RNG_GLOBAL_SEED_INDEX,
                    "role": role.name,
                    "role_id": int(role),
                    "substream": 0,
                }
            )
    if cfg.suite_name == "angle_dose":
        specs.append(
            {
                "mode": mode,
                "suite": cfg.suite_name,
                "architecture": "linear",
                "seed": RNG_GLOBAL_SEED_INDEX,
                "role": RNGRole.BOOTSTRAP_ANGLE_MEDIAN.name,
                "role_id": int(RNGRole.BOOTSTRAP_ANGLE_MEDIAN),
                "substream": 0,
            }
        )
    return specs


def audit_rng_namespaces() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enumerate smoke/formal streams and hard-fail on entropy or state collisions."""
    rows: list[dict[str, Any]] = []
    entropy_seen: dict[tuple[int, ...], str] = {}
    state_seen: dict[tuple[int, ...], str] = {}
    for mode in ("smoke", "formal"):
        for suite in ("primary", "coverage_horizon", "angle_dose", "iid_robustness"):
            cfg = suite_config(mode, suite)
            for spec in rng_stream_specs(cfg, mode):
                role = RNGRole(int(spec["role_id"]))
                entropy = rng_entropy_tuple(
                    cfg,
                    int(spec["seed"]),
                    role,
                    str(spec["architecture"]),
                    int(spec["substream"]),
                )
                sequence = np.random.SeedSequence(entropy)
                state = tuple(int(value) for value in sequence.generate_state(8, dtype=np.uint32))
                label = (
                    f"{mode}:{suite}:{spec['architecture']}:seed={spec['seed']}:"
                    f"{spec['role']}:substream={spec['substream']}"
                )
                if entropy in entropy_seen:
                    raise RuntimeError(
                        f"Duplicate RNG entropy tuple for {label} and {entropy_seen[entropy]}: {entropy}"
                    )
                if state in state_seen:
                    raise RuntimeError(
                        f"Duplicate generated RNG state for {label} and {state_seen[state]}"
                    )
                entropy_seen[entropy] = label
                state_seen[state] = label
                rows.append(
                    {
                        **spec,
                        "seed_base": cfg.seed_base,
                        "mode_id": RNG_MODE_IDS[cfg.rng_mode],
                        "suite_id": RNG_SUITE_IDS[cfg.suite_name],
                        "architecture_id": RNG_ARCHITECTURE_IDS[str(spec["architecture"])],
                        "canonical_key": label,
                        "entropy_tuple": ":".join(str(value) for value in entropy),
                        "generated_state_u32": ":".join(f"{value:08x}" for value in state),
                    }
                )
    payload = "\n".join(
        f"{row['entropy_tuple']}|{row['generated_state_u32']}" for row in rows
    ).encode("utf-8")
    smoke_entropy = {row["entropy_tuple"] for row in rows if row["mode"] == "smoke"}
    formal_entropy = {row["entropy_tuple"] for row in rows if row["mode"] == "formal"}
    smoke_states = {row["generated_state_u32"] for row in rows if row["mode"] == "smoke"}
    formal_states = {row["generated_state_u32"] for row in rows if row["mode"] == "formal"}
    if smoke_entropy & formal_entropy or smoke_states & formal_states:
        raise RuntimeError("Smoke/formal RNG namespaces overlap")
    summary = {
        "namespace_version": RNG_NAMESPACE_VERSION,
        "schema": "SeedSequence([seed_base, mode_id, seed, role_id, suite_id, architecture_id, substream])",
        "stream_count": len(rows),
        "unique_entropy_tuple_count": len(entropy_seen),
        "unique_generated_state_count": len(state_seen),
        "entropy_collision_count": 0,
        "generated_state_collision_count": 0,
        "smoke_formal_entropy_intersection_count": len(smoke_entropy & formal_entropy),
        "smoke_formal_state_intersection_count": len(smoke_states & formal_states),
        "registry_sha256": hashlib.sha256(payload).hexdigest(),
        "modes": ["smoke", "formal"],
        "suites": list(RNG_SUITE_IDS),
    }
    return rows, summary


class Model(Protocol):
    def copy(self) -> "Model": ...

    def step(self, x: np.ndarray, targets: np.ndarray, lr: float) -> float: ...

    def logits(self, x: np.ndarray) -> np.ndarray: ...

    def direction_r2(self, x_ref: np.ndarray, direction: np.ndarray) -> float: ...

    def reset_optimizer(self) -> None: ...


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-np.mean(np.sum(targets * log_probs, axis=1)))


def one_hot(y: np.ndarray) -> np.ndarray:
    out = np.zeros((len(y), 2), dtype=np.float64)
    out[np.arange(len(y)), y] = 1.0
    return out


class LinearSoftmax:
    def __init__(self, rng: np.random.Generator, input_dim: int, init_scale: float) -> None:
        self.w = rng.normal(0.0, init_scale, size=(input_dim, 2))

    def copy(self) -> "LinearSoftmax":
        other = object.__new__(LinearSoftmax)
        other.w = self.w.copy()
        return other

    def logits(self, x: np.ndarray) -> np.ndarray:
        return np.einsum("...d,dk->...k", x, self.w, optimize=True)

    def step(self, x: np.ndarray, targets: np.ndarray, lr: float) -> float:
        logits = self.logits(x)
        probs = stable_softmax(logits)
        diff = (probs - targets) / len(x)
        self.w -= lr * np.einsum("...d,...k->dk", x, diff, optimize=True)
        return cross_entropy(logits, targets)

    def reset_optimizer(self) -> None:
        # Vanilla SGD has no optimizer state.
        return None

    def direction_r2(self, x_ref: np.ndarray, direction: np.ndarray) -> float:
        del x_ref
        response = np.einsum("d,dk->k", direction, self.w, optimize=True)
        centered = response - response.mean()
        return float(centered @ centered)


class TanhMLP:
    def __init__(self, rng: np.random.Generator, input_dim: int, hidden: int) -> None:
        # Xavier scales are fixed before outcome inspection.
        self.w1 = rng.normal(0.0, math.sqrt(2.0 / (input_dim + hidden)), size=(input_dim, hidden))
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.w2 = rng.normal(0.0, math.sqrt(2.0 / (hidden + 2)), size=(hidden, 2))
        self.b2 = np.zeros(2, dtype=np.float64)
        self.reset_optimizer()

    def copy(self) -> "TanhMLP":
        other = object.__new__(TanhMLP)
        other.w1 = self.w1.copy()
        other.b1 = self.b1.copy()
        other.w2 = self.w2.copy()
        other.b2 = self.b2.copy()
        other._adam_m = {key: value.copy() for key, value in self._adam_m.items()}
        other._adam_v = {key: value.copy() for key, value in self._adam_v.items()}
        other._adam_t = self._adam_t
        return other

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.tanh(np.einsum("...d,dh->...h", x, self.w1, optimize=True) + self.b1)
        logits = np.einsum("...h,hk->...k", h, self.w2, optimize=True) + self.b2
        return h, logits

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)[1]

    def step(self, x: np.ndarray, targets: np.ndarray, lr: float) -> float:
        h, logits = self._forward(x)
        probs = stable_softmax(logits)
        dlogits = (probs - targets) / len(x)
        grad_w2 = np.einsum("...h,...k->hk", h, dlogits, optimize=True)
        grad_b2 = dlogits.sum(axis=0)
        dh = np.einsum("...k,hk->...h", dlogits, self.w2, optimize=True)
        dpre = dh * (1.0 - h * h)
        grad_w1 = np.einsum("...d,...h->dh", x, dpre, optimize=True)
        grad_b1 = dpre.sum(axis=0)
        gradients = {"w1": grad_w1, "b1": grad_b1, "w2": grad_w2, "b2": grad_b2}
        self._adam_t += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for name, gradient in gradients.items():
            self._adam_m[name] = beta1 * self._adam_m[name] + (1.0 - beta1) * gradient
            self._adam_v[name] = beta2 * self._adam_v[name] + (1.0 - beta2) * (gradient * gradient)
            m_hat = self._adam_m[name] / (1.0 - beta1**self._adam_t)
            v_hat = self._adam_v[name] / (1.0 - beta2**self._adam_t)
            parameter = getattr(self, name)
            parameter -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        return cross_entropy(logits, targets)

    def reset_optimizer(self) -> None:
        self._adam_m = {
            "w1": np.zeros_like(self.w1),
            "b1": np.zeros_like(self.b1),
            "w2": np.zeros_like(self.w2),
            "b2": np.zeros_like(self.b2),
        }
        self._adam_v = {key: np.zeros_like(value) for key, value in self._adam_m.items()}
        self._adam_t = 0

    def direction_r2(self, x_ref: np.ndarray, direction: np.ndarray) -> float:
        h = np.tanh(np.einsum("...d,dh->...h", x_ref, self.w1, optimize=True) + self.b1)
        preactivation_direction = np.einsum("d,dh->h", direction, self.w1, optimize=True)
        dh = (1.0 - h * h) * preactivation_direction[None, :]
        response = np.einsum("...h,hk->...k", dh, self.w2, optimize=True)
        centered = response - response.mean(axis=1, keepdims=True)
        return float(np.mean(np.sum(centered * centered, axis=1)))


def make_haar_rotation(rng: np.random.Generator, input_dim: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(input_dim, input_dim)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q = q * signs[None, :]
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1.0
    return q


def model_factory(name: str, rng: np.random.Generator, cfg: ExperimentConfig) -> Model:
    if name == "linear":
        return LinearSoftmax(rng, cfg.input_dim, cfg.linear_init_scale)
    if name == "mlp_tanh":
        return TanhMLP(rng, cfg.input_dim, cfg.mlp_hidden)
    raise ValueError(f"Unknown architecture: {name}")


def model_lr(name: str, cfg: ExperimentConfig) -> float:
    return cfg.linear_lr if name == "linear" else cfg.mlp_lr


def supervised_from_base(
    y: np.ndarray,
    eps_semantic: np.ndarray,
    z_w: np.ndarray,
    z_v: np.ndarray,
    z_aux: np.ndarray,
    basis: np.ndarray,
    coverage_v: float,
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    sign = 2.0 * y.astype(np.float64) - 1.0
    coords = np.concatenate(
        [
            (cfg.semantic_mean * sign + cfg.semantic_std * eps_semantic)[..., None],
            (math.sqrt(cfg.train_var_w) * z_w)[..., None],
            (math.sqrt(coverage_v) * z_v)[..., None],
            z_aux,
        ],
        axis=-1,
    )
    rotated = np.einsum("...j,ij->...i", coords, basis, optimize=True)
    return rotated, one_hot(y.reshape(-1)).reshape(*y.shape, 2)


def reference_from_base(
    y: np.ndarray,
    eps_semantic: np.ndarray,
    z_w: np.ndarray,
    z_v: np.ndarray,
    z_aux: np.ndarray,
    basis: np.ndarray,
    var_v: float,
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    sign = 2.0 * y.astype(np.float64) - 1.0
    coords = np.column_stack(
        [
            cfg.semantic_mean * sign + cfg.semantic_std * eps_semantic,
            math.sqrt(cfg.reference_var_w) * z_w,
            math.sqrt(var_v) * z_v,
            z_aux,
        ]
    )
    rotated = np.einsum("...j,ij->...i", coords, basis, optimize=True)
    return rotated, one_hot(y)


def warmup_batch(
    method: str,
    base_z: np.ndarray,
    basis: np.ndarray,
    cfg: ExperimentConfig,
) -> np.ndarray:
    angle_deg: float | None = None
    if method == "targeted_v":
        angle_deg = 0.0
    elif method == "wrong_w":
        angle_deg = 90.0
    elif method.startswith("angle_"):
        angle_deg = float(method.removeprefix("angle_"))
    if method == "isotropic_wv":
        q_w = q_v = 0.5 * (cfg.warmup_q_high + cfg.warmup_q_low)
    elif method == "no_warmup":
        return np.zeros((len(base_z), cfg.input_dim), dtype=np.float64)
    elif angle_deg is not None:
        theta = math.radians(angle_deg)
        coords = np.zeros((len(base_z), cfg.input_dim), dtype=np.float64)
        high_component = math.sqrt(cfg.warmup_q_high) * base_z[:, 0]
        low_component = math.sqrt(cfg.warmup_q_low) * base_z[:, 1]
        coords[:, 1] = math.sin(theta) * high_component + math.cos(theta) * low_component
        coords[:, 2] = math.cos(theta) * high_component - math.sin(theta) * low_component
        return np.einsum("...j,ij->...i", coords, basis, optimize=True)
    else:
        raise ValueError(method)
    coords = np.zeros((len(base_z), cfg.input_dim), dtype=np.float64)
    coords[:, 1] = math.sqrt(q_w) * base_z[:, 0]
    coords[:, 2] = math.sqrt(q_v) * base_z[:, 1]
    return np.einsum("...j,ij->...i", coords, basis, optimize=True)


def balanced_binary_targets(
    rng: np.random.Generator, steps: int, batch_size: int
) -> np.ndarray:
    if batch_size % 2 != 0:
        raise ValueError("Balanced hard-label warm-up requires an even batch size")
    labels = np.empty((steps, batch_size), dtype=np.int64)
    template = np.repeat(np.arange(2, dtype=np.int64), batch_size // 2)
    for step in range(steps):
        labels[step] = rng.permutation(template)
    return one_hot(labels.reshape(-1)).reshape(steps, batch_size, 2)


def strict_label_invariant_batches(
    rng: np.random.Generator,
    steps: int,
    batch_size: int,
    nuisance_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate finite batches with exactly identical nuisance draws per class.

    Every quartet contains labels (0, 0, 1, 1) and nuisance draws
    (z, -z, z, -z).  Thus each class has the same empirical nuisance
    distribution, not merely independence in expectation.  Rows are shuffled
    with a shared permutation before use.
    """
    if batch_size % 4 != 0:
        raise ValueError("Strict label-invariant construction requires batch_size divisible by 4")
    groups = batch_size // 4
    z_base = rng.normal(size=(steps, groups, nuisance_dim))
    z = np.stack([z_base, -z_base, z_base, -z_base], axis=2).reshape(
        steps, batch_size, nuisance_dim
    )
    y = np.tile(np.asarray([0, 0, 1, 1], dtype=np.int64), groups)
    labels = np.broadcast_to(y, (steps, batch_size)).copy()
    semantic_noise = rng.normal(size=(steps, batch_size))
    for step in range(steps):
        permutation = rng.permutation(batch_size)
        labels[step] = labels[step, permutation]
        semantic_noise[step] = semantic_noise[step, permutation]
        z[step] = z[step, permutation]
    return labels, semantic_noise, z


def iid_batches(
    rng: np.random.Generator,
    steps: int,
    batch_size: int,
    nuisance_dim: int,
    *,
    balanced_labels: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if balanced_labels:
        if batch_size % 2 != 0:
            raise ValueError("Balanced IID batches require an even batch size")
        template = np.repeat(np.arange(2, dtype=np.int64), batch_size // 2)
        labels = np.stack([rng.permutation(template) for _ in range(steps)], axis=0)
    else:
        labels = rng.integers(0, 2, size=(steps, batch_size))
    semantic_noise = rng.normal(size=(steps, batch_size))
    nuisance = rng.normal(size=(steps, batch_size, nuisance_dim))
    return labels, semantic_noise, nuisance


def strict_label_invariant_evaluation(
    rng: np.random.Generator,
    sample_size: int,
    nuisance_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, semantic_noise, z = strict_label_invariant_batches(
        rng, 1, sample_size, nuisance_dim
    )
    return y[0], semantic_noise[0], z[0]


def whiten_warmup_batches_exactly(z: np.ndarray) -> np.ndarray:
    """Whiten each finite 2-D warm-up batch to zero mean and identity covariance."""
    if z.ndim != 3 or z.shape[-1] != 2:
        raise ValueError(f"Expected [steps,batch,2] warm-up array, got {z.shape}")
    whitened = np.empty_like(z, dtype=np.float64)
    for step in range(len(z)):
        centered = z[step] - z[step].mean(axis=0, keepdims=True)
        covariance = np.einsum("ni,nj->ij", centered, centered, optimize=True) / len(centered)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if float(eigenvalues.min()) <= 1e-12:
            raise RuntimeError(f"Degenerate warm-up covariance at step {step}: {eigenvalues}")
        inverse_sqrt = np.einsum(
            "ia,a,ja->ij", eigenvectors, 1.0 / np.sqrt(eigenvalues), eigenvectors, optimize=True
        )
        whitened[step] = np.einsum("ni,ij->nj", centered, inverse_sqrt, optimize=True)
    return whitened


def matched_spectrum_audit(
    warm_z: np.ndarray,
    basis: np.ndarray,
    cfg: ExperimentConfig,
) -> dict[str, float | int]:
    endpoint_errors: list[float] = []
    for step in sorted({0, len(warm_z) - 1}):
        endpoint_errors.append(
            float(
                np.max(
                    np.abs(
                        warmup_batch("targeted_v", warm_z[step], basis, cfg)
                        - warmup_batch("angle_0", warm_z[step], basis, cfg)
                    )
                )
            )
        )
        endpoint_errors.append(
            float(
                np.max(
                    np.abs(
                        warmup_batch("wrong_w", warm_z[step], basis, cfg)
                        - warmup_batch("angle_90", warm_z[step], basis, cfg)
                    )
                )
            )
        )
    z_covariances = np.einsum(
        "sni,snj->sij", warm_z, warm_z, optimize=True
    ) / warm_z.shape[1]
    whiten_error = float(np.max(np.abs(z_covariances - np.eye(2)[None, :, :])))
    low = math.sqrt(cfg.warmup_q_low)
    high = math.sqrt(cfg.warmup_q_high)
    # Matrices map common base columns [z0,z1] into [w,v].  These are
    # exactly the theta=0 and theta=90 endpoints used by warmup_batch.
    target_map = np.asarray([[0.0, low], [high, 0.0]])
    wrong_map = np.asarray([[high, 0.0], [0.0, -low]])
    target_covariances = np.einsum(
        "ai,sij,bj->sab", target_map, z_covariances, target_map, optimize=True
    )
    wrong_covariances = np.einsum(
        "ai,sij,bj->sab", wrong_map, z_covariances, wrong_map, optimize=True
    )
    target_eigenvalues = np.linalg.eigvalsh(target_covariances)
    wrong_eigenvalues = np.linalg.eigvalsh(wrong_covariances)
    eigen_error = float(np.max(np.abs(target_eigenvalues - wrong_eigenvalues)))
    target_traces = np.trace(target_covariances, axis1=1, axis2=2)
    wrong_traces = np.trace(wrong_covariances, axis1=1, axis2=2)
    trace_error = float(np.max(np.abs(target_traces - wrong_traces)))
    target_determinants = np.linalg.det(target_covariances)
    wrong_determinants = np.linalg.det(wrong_covariances)
    determinant_error = float(np.max(np.abs(target_determinants - wrong_determinants)))
    result: dict[str, float | int] = {
        "checked_batch_count": len(warm_z),
        "max_whitened_covariance_abs_error": whiten_error,
        "max_target_wrong_eigenvalue_abs_difference": eigen_error,
        "max_target_wrong_trace_abs_difference": trace_error,
        "max_target_wrong_determinant_abs_difference": determinant_error,
        "max_named_vs_angle_endpoint_input_abs_difference": max(endpoint_errors),
    }
    if max(
        whiten_error,
        eigen_error,
        trace_error,
        determinant_error,
        max(endpoint_errors),
    ) >= 1e-10:
        raise RuntimeError(f"Matched-spectrum audit failed: {result}")
    return result


def finite_batch_invariance_audit(
    labels: np.ndarray, nuisance: np.ndarray
) -> dict[str, float | int]:
    if labels.ndim == 1:
        labels = labels[None, :]
        nuisance = nuisance[None, :, :]
    batch_indices = sorted({0, len(labels) - 1})
    mean_errors: list[float] = []
    second_moment_errors: list[float] = []
    balance_errors: list[int] = []
    for step in batch_indices:
        class_values = [nuisance[step, labels[step] == klass] for klass in (0, 1)]
        mean_errors.append(float(np.max(np.abs(class_values[0].mean(0) - class_values[1].mean(0)))))
        second0 = np.einsum("ni,nj->ij", class_values[0], class_values[0]) / len(class_values[0])
        second1 = np.einsum("ni,nj->ij", class_values[1], class_values[1]) / len(class_values[1])
        second_moment_errors.append(float(np.max(np.abs(second0 - second1))))
        balance_errors.append(abs(len(class_values[0]) - len(class_values[1])))
    return {
        "checked_batch_count": len(batch_indices),
        "max_abs_class_mean_difference": max(mean_errors),
        "max_abs_class_second_moment_difference": max(second_moment_errors),
        "max_class_count_difference": max(balance_errors),
    }


def intervene_linear_v_state(
    base: LinearSoftmax, v: np.ndarray, alpha: float
) -> tuple[LinearSoftmax, float, float]:
    """Scale only the centered v response, leaving orthogonal input states fixed."""
    model = base.copy()
    response = v @ model.w
    centered = response - response.mean()
    before = float(centered @ centered)
    model.w += np.outer(v, (alpha - 1.0) * centered)
    after_response = v @ model.w
    after_centered = after_response - after_response.mean()
    after = float(after_centered @ after_centered)
    return model, before, after


def prediction_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    probs = stable_softmax(logits)
    y = targets.argmax(axis=1)
    nll = cross_entropy(logits, targets)
    accuracy = float(np.mean(probs.argmax(axis=1) == y))
    brier = float(np.mean(np.sum((probs - targets) ** 2, axis=1)))
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(np.float64)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1.0 else confidence <= hi)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(confidence[mask].mean() - correct[mask].mean()))
    return {"nll": nll, "accuracy": accuracy, "brier": brier, "ece_15": ece}


def append_state_rows(
    *,
    model: Model,
    architecture: str,
    seed: int,
    coverage_v: float,
    method: str,
    stage: str,
    supervised_step: int,
    global_step: int,
    basis: np.ndarray,
    x_ref: np.ndarray,
    evaluation_sets: dict[str, tuple[np.ndarray, np.ndarray]],
    residual_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    response_rows: list[dict[str, Any]] | None = None,
) -> None:
    for idx, name in enumerate(DIRECTIONS):
        r2 = max(0.0, model.direction_r2(x_ref, basis[:, idx]))
        residual_rows.append(
            {
                "architecture": architecture,
                "seed": seed,
                "coverage_v": coverage_v,
                "method": method,
                "stage": stage,
                "supervised_step": supervised_step,
                "global_step": global_step,
                "direction": name,
                "r": math.sqrt(r2),
                "r2": r2,
            }
        )
        if (
            response_rows is not None
            and isinstance(model, LinearSoftmax)
            and name == "v_weak_nuisance"
        ):
            response = basis[:, idx] @ model.w
            centered = response - response.mean()
            for component, value in enumerate(centered):
                response_rows.append(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "coverage_v": coverage_v,
                        "method": method,
                        "stage": stage,
                        "supervised_step": supervised_step,
                        "global_step": global_step,
                        "direction": name,
                        "logit_component": component,
                        "centered_response": float(value),
                    }
                )
    for split, (x_eval, y_eval) in evaluation_sets.items():
        metric_rows.append(
            {
                "architecture": architecture,
                "seed": seed,
                "coverage_v": coverage_v,
                "method": method,
                "stage": stage,
                "supervised_step": supervised_step,
                "global_step": global_step,
                "evaluation_split": split,
                **prediction_metrics(model.logits(x_eval), y_eval),
            }
        )


def append_linear_dense_residual_rows(
    *,
    model: Model,
    seed: int,
    coverage_v: float,
    method: str,
    phase: str,
    phase_step: int,
    global_step: int,
    basis: np.ndarray,
    rows: list[dict[str, Any]],
) -> None:
    if not isinstance(model, LinearSoftmax):
        return
    for idx, direction in enumerate(DIRECTIONS):
        r2 = max(0.0, model.direction_r2(np.empty((0, 0)), basis[:, idx]))
        rows.append(
            {
                "architecture": "linear",
                "seed": seed,
                "coverage_v": coverage_v,
                "method": method,
                "phase": phase,
                "phase_step": phase_step,
                "global_step": global_step,
                "direction": direction,
                "r": math.sqrt(r2),
                "R": r2,
            }
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def correlation(x: Iterable[float], y: Iterable[float], *, rank: bool = False) -> float:
    xa = np.asarray(list(x), dtype=np.float64)
    ya = np.asarray(list(y), dtype=np.float64)
    if rank:
        xa, ya = average_ranks(xa), average_ranks(ya)
    if len(xa) < 2 or np.std(xa) == 0.0 or np.std(ya) == 0.0:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), float(values[0])
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def paired_bootstrap_mean_ci(
    values: np.ndarray, repetitions: int, rng: np.random.Generator
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    means = np.empty(repetitions, dtype=np.float64)
    chunk = 1_000
    written = 0
    while written < repetitions:
        take = min(chunk, repetitions - written)
        indices = rng.integers(0, len(values), size=(take, len(values)))
        means[written : written + take] = values[indices].mean(axis=1)
        written += take
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_bootstrap_median_ci(
    values: np.ndarray, repetitions: int, rng: np.random.Generator
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    medians = np.empty(repetitions, dtype=np.float64)
    chunk = 1_000
    written = 0
    while written < repetitions:
        take = min(chunk, repetitions - written)
        indices = rng.integers(0, len(values), size=(take, len(values)))
        medians[written : written + take] = np.median(values[indices], axis=1)
        written += take
    low, high = np.percentile(medians, [2.5, 97.5])
    return float(low), float(high)


def derive_vector_retention(
    response_rows: list[dict[str, Any]], cfg: ExperimentConfig
) -> list[dict[str, Any]]:
    lookup: dict[tuple[Any, ...], np.ndarray] = {}
    groups: dict[tuple[Any, ...], dict[int, float]] = {}
    for row in response_rows:
        key = (
            row["architecture"],
            int(row["seed"]),
            float(row["coverage_v"]),
            row["method"],
            row["stage"],
            int(row["supervised_step"]),
        )
        groups.setdefault(key, {})[int(row["logit_component"])] = float(row["centered_response"])
    for key, components in groups.items():
        lookup[key] = np.asarray([components[idx] for idx in sorted(components)], dtype=np.float64)

    rows: list[dict[str, Any]] = []
    if "linear" not in cfg.architectures:
        return rows
    for seed in range(cfg.seeds):
        for coverage_v in cfg.coverage_v_grid:
            target_init = lookup[("linear", seed, coverage_v, "targeted_v", "post_warmup", 0)]
            wrong_init = lookup[("linear", seed, coverage_v, "wrong_w", "post_warmup", 0)]
            delta_init = wrong_init - target_init
            norm_init = float(np.linalg.norm(delta_init))
            for horizon in cfg.supervised_horizons:
                if horizon == 0:
                    target_t, wrong_t = target_init, wrong_init
                else:
                    target_t = lookup[("linear", seed, coverage_v, "targeted_v", "supervised", horizon)]
                    wrong_t = lookup[("linear", seed, coverage_v, "wrong_w", "supervised", horizon)]
                delta_t = wrong_t - target_t
                norm_t = float(np.linalg.norm(delta_t))
                rows.append(
                    {
                        "architecture": "linear",
                        "seed": seed,
                        "coverage_v": coverage_v,
                        "supervised_horizon": horizon,
                        "delta_init_vector_norm": norm_init,
                        "delta_t_vector_norm": norm_t,
                        "vector_difference_retention_ratio": norm_t / max(norm_init, cfg.log_epsilon),
                    }
                )
    return rows


def derive_intervention_slopes(
    intervention_rows: list[dict[str, Any]], cfg: ExperimentConfig
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not intervention_rows:
        return rows
    for seed in range(cfg.seeds):
        for horizon in cfg.supervised_horizons:
            sub = sorted(
                (
                    row
                    for row in intervention_rows
                    if int(row["seed"]) == seed
                    and int(row["supervised_horizon"]) == horizon
                    and float(row["alpha"]) > 0.0
                ),
                key=lambda row: float(row["alpha"]),
            )
            x = np.asarray([row["log_R_init_eps"] for row in sub], dtype=np.float64)
            y = np.asarray([row["log_R_t_eps"] for row in sub], dtype=np.float64)
            slope, intercept = np.polyfit(x, y, 1)
            adjacent_wins = int(np.sum(np.diff(y) > 0.0))
            adjacent_total = max(0, len(y) - 1)
            rows.append(
                {
                    "seed": seed,
                    "coverage_v": cfg.primary_coverage_v,
                    "supervised_horizon": horizon,
                    "log_R_t_on_log_R_init_slope": float(slope),
                    "intercept": float(intercept),
                    "pearson": correlation(x, y),
                    "spearman": correlation(x, y, rank=True),
                    "strict_order_non_decreasing": int(np.all(np.diff(y) >= -1e-12)),
                    "adjacent_order_pairs_won": adjacent_wins,
                    "adjacent_order_pairs_total": adjacent_total,
                    "adjacent_order_win_fraction": adjacent_wins / max(1, adjacent_total),
                    "alpha_count": len(sub),
                }
            )
    return rows


def derive_intervention_vector_retention(
    intervention_rows: list[dict[str, Any]], cfg: ExperimentConfig
) -> list[dict[str, Any]]:
    lookup = {
        (int(row["seed"]), float(row["alpha"]), int(row["supervised_horizon"])): np.asarray(
            [row["centered_response_0"], row["centered_response_1"]], dtype=np.float64
        )
        for row in intervention_rows
    }
    rows: list[dict[str, Any]] = []
    if not lookup:
        return rows
    for seed in range(cfg.seeds):
        init_high = lookup[(seed, 2.0, 0)]
        init_low = lookup[(seed, 0.25, 0)]
        init_difference_norm = float(np.linalg.norm(init_high - init_low))
        for horizon in cfg.supervised_horizons:
            high_t = lookup[(seed, 2.0, horizon)]
            low_t = lookup[(seed, 0.25, horizon)]
            t_difference_norm = float(np.linalg.norm(high_t - low_t))
            rows.append(
                {
                    "seed": seed,
                    "coverage_v": cfg.primary_coverage_v,
                    "alpha_high": 2.0,
                    "alpha_low": 0.25,
                    "supervised_horizon": horizon,
                    "init_response_difference_norm": init_difference_norm,
                    "t_response_difference_norm": t_difference_norm,
                    "difference_retention_ratio": t_difference_norm
                    / max(init_difference_norm, cfg.log_epsilon),
                }
            )
    return rows


def method_angle_degrees(method: str) -> float:
    mapping = {
        "targeted_v": 0.0,
        "angle_22.5": 22.5,
        "angle_45": 45.0,
        "angle_67.5": 67.5,
        "wrong_w": 90.0,
    }
    return mapping[method]


def derive_angle_dose_seed_rows(
    supervised_pairs: list[dict[str, Any]], cfg: ExperimentConfig
) -> list[dict[str, Any]]:
    required = {"targeted_v", "angle_22.5", "angle_45", "angle_67.5", "wrong_w"}
    if not required.issubset(set(cfg.methods)):
        return []
    rows: list[dict[str, Any]] = []
    horizon = cfg.primary_horizon
    for seed in range(cfg.seeds):
        sub = sorted(
            (
                row
                for row in supervised_pairs
                if row["architecture"] == "linear"
                and int(row["seed"]) == seed
                and float(row["coverage_v"]) == cfg.primary_coverage_v
                and int(row["supervised_horizon"]) == horizon
                and row["method"] in required
            ),
            key=lambda row: method_angle_degrees(row["method"]),
        )
        angles = [method_angle_degrees(row["method"]) for row in sub]
        log_init = [row["log_R_init_eps"] for row in sub]
        log_t = [row["log_R_t_eps"] for row in sub]
        rows.append(
            {
                "seed": seed,
                "coverage_v": cfg.primary_coverage_v,
                "supervised_horizon": horizon,
                "angle_count": len(sub),
                "spearman_angle_vs_log_R_init": correlation(angles, log_init, rank=True),
                "spearman_angle_vs_log_R_t": correlation(angles, log_t, rank=True),
                "spearman_log_R_init_vs_log_R_t": correlation(log_init, log_t, rank=True),
            }
        )
    return rows


def derive_pair_tables(
    residual_rows: list[dict[str, Any]], cfg: ExperimentConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    v_rows = [r for r in residual_rows if r["direction"] == "v_weak_nuisance"]
    lookup = {
        (
            r["architecture"],
            int(r["seed"]),
            float(r["coverage_v"]),
            r["method"],
            r["stage"],
            int(r["supervised_step"]),
        ): float(r["r"])
        for r in v_rows
    }
    supervised_pairs: list[dict[str, Any]] = []
    paired_effects: list[dict[str, Any]] = []
    for architecture in cfg.architectures:
        for seed in range(cfg.seeds_for(architecture)):
            for coverage_v in cfg.coverage_v_grid:
                for method in cfg.methods:
                    r_random = lookup[(architecture, seed, coverage_v, method, "random_init", -1)]
                    r_init = lookup[(architecture, seed, coverage_v, method, "post_warmup", 0)]
                    for horizon in cfg.horizons_for(architecture):
                        if horizon == 0:
                            r_t = r_init
                        else:
                            r_t = lookup[(architecture, seed, coverage_v, method, "supervised", horizon)]
                        supervised_pairs.append(
                            {
                                "architecture": architecture,
                                "seed": seed,
                                "coverage_v": coverage_v,
                                "method": method,
                                "supervised_horizon": horizon,
                                "r_random_v": r_random,
                                "r_init_v": r_init,
                                "r_t_v": r_t,
                                "R_random_v": r_random * r_random,
                                "R_init_v": r_init * r_init,
                                "R_t_v": r_t * r_t,
                                "log_R_init_eps": math.log(r_init * r_init + cfg.log_epsilon),
                                "log_R_t_eps": math.log(r_t * r_t + cfg.log_epsilon),
                                "retention_r_t_over_r_init": r_t / max(r_init, 1e-15),
                            }
                        )
                for baseline in (method for method in cfg.methods if method != "targeted_v"):
                    target_init = lookup[(architecture, seed, coverage_v, "targeted_v", "post_warmup", 0)]
                    base_init = lookup[(architecture, seed, coverage_v, baseline, "post_warmup", 0)]
                    for horizon in cfg.horizons_for(architecture):
                        target_t = target_init if horizon == 0 else lookup[(architecture, seed, coverage_v, "targeted_v", "supervised", horizon)]
                        base_t = base_init if horizon == 0 else lookup[(architecture, seed, coverage_v, baseline, "supervised", horizon)]
                        paired_effects.append(
                            {
                                "architecture": architecture,
                                "seed": seed,
                                "coverage_v": coverage_v,
                                "baseline": baseline,
                                "supervised_horizon": horizon,
                                "baseline_minus_targeted_r_init": base_init - target_init,
                                "baseline_minus_targeted_r_t": base_t - target_t,
                                "targeted_minus_baseline_log_R_init": math.log(
                                    target_init * target_init + cfg.log_epsilon
                                )
                                - math.log(base_init * base_init + cfg.log_epsilon),
                                "targeted_minus_baseline_log_R_t": math.log(
                                    target_t * target_t + cfg.log_epsilon
                                )
                                - math.log(base_t * base_t + cfg.log_epsilon),
                                "norm_difference_retention_ratio": (base_t - target_t)
                                / max(base_init - target_init, cfg.log_epsilon),
                                "targeted_lower_at_init": int(target_init < base_init),
                                "targeted_lower_at_t": int(target_t < base_t),
                                "ordering_persists": int((target_init < base_init) and (target_t < base_t)),
                            }
                        )

    association_rows: list[dict[str, Any]] = []
    for architecture in cfg.architectures:
        for coverage_v in cfg.coverage_v_grid:
            for horizon in cfg.horizons_for(architecture):
                sub = [
                    r
                    for r in supervised_pairs
                    if r["architecture"] == architecture
                    and float(r["coverage_v"]) == coverage_v
                    and int(r["supervised_horizon"]) == horizon
                ]
                association_rows.append(
                    {
                        "architecture": architecture,
                        "coverage_v": coverage_v,
                        "supervised_horizon": horizon,
                        "n_seed_method_points": len(sub),
                        "pearson_r_init_vs_r_t": correlation(
                            (r["r_init_v"] for r in sub), (r["r_t_v"] for r in sub)
                        ),
                        "spearman_r_init_vs_r_t": correlation(
                            (r["r_init_v"] for r in sub), (r["r_t_v"] for r in sub), rank=True
                        ),
                    }
                )
    return supervised_pairs, paired_effects, association_rows


def aggregate_summary(
    supervised_pairs: list[dict[str, Any]],
    paired_effects: list[dict[str, Any]],
    association_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    vector_retention_rows: list[dict[str, Any]],
    intervention_slope_rows: list[dict[str, Any]],
    intervention_retention_rows: list[dict[str, Any]],
    angle_dose_rows: list[dict[str, Any]],
    cfg: ExperimentConfig,
) -> dict[str, Any]:
    architecture_summaries: dict[str, Any] = {}
    for architecture in cfg.architectures:
        primary_horizon = cfg.primary_horizon_for(architecture)
        metric_initial = [
            row
            for row in metric_rows
            if row["architecture"] == architecture
            and float(row["coverage_v"]) == cfg.primary_coverage_v
            and row["evaluation_split"] == "train_like"
            and row["stage"] == "post_warmup"
        ]
        metric_final = [
            row
            for row in metric_rows
            if row["architecture"] == architecture
            and float(row["coverage_v"]) == cfg.primary_coverage_v
            and row["evaluation_split"] == "train_like"
            and row["stage"] == "supervised"
            and int(row["supervised_step"]) == primary_horizon
        ]
        initial_nll = float(np.mean([row["nll"] for row in metric_initial]))
        final_nll = float(np.mean([row["nll"] for row in metric_final]))
        final_accuracy = float(np.mean([row["accuracy"] for row in metric_final]))
        primary_effect = [
            r
            for r in paired_effects
            if r["architecture"] == architecture
            and float(r["coverage_v"]) == cfg.primary_coverage_v
            and r["baseline"] == "wrong_w"
            and int(r["supervised_horizon"]) == primary_horizon
        ]
        delta_init = np.asarray([r["baseline_minus_targeted_r_init"] for r in primary_effect], dtype=float)
        delta_t = np.asarray([r["baseline_minus_targeted_r_t"] for r in primary_effect], dtype=float)
        log_delta_init = np.asarray(
            [r["targeted_minus_baseline_log_R_init"] for r in primary_effect], dtype=float
        )
        log_delta_t = np.asarray(
            [r["targeted_minus_baseline_log_R_t"] for r in primary_effect], dtype=float
        )
        log_init_ci = paired_bootstrap_mean_ci(
            log_delta_init,
            cfg.bootstrap_repetitions,
            hierarchical_rng(
                cfg,
                RNG_GLOBAL_SEED_INDEX,
                RNGRole.BOOTSTRAP_G1,
                architecture,
            ),
        )
        log_t_ci = paired_bootstrap_mean_ci(
            log_delta_t,
            cfg.bootstrap_repetitions,
            hierarchical_rng(
                cfg,
                RNG_GLOBAL_SEED_INDEX,
                RNGRole.BOOTSTRAP_G2,
                architecture,
            ),
        )
        assoc = next(
            r
            for r in association_rows
            if r["architecture"] == architecture
            and float(r["coverage_v"]) == cfg.primary_coverage_v
            and int(r["supervised_horizon"]) == primary_horizon
        )
        architecture_summaries[architecture] = {
            "primary_comparison": "wrong_w minus targeted_v",
            "primary_coverage_v": cfg.primary_coverage_v,
            "primary_horizon": primary_horizon,
            "G0_initial_mean_train_like_nll": initial_nll,
            "G0_final_mean_train_like_nll": final_nll,
            "G0_final_mean_train_like_accuracy": final_accuracy,
            "G0_pass": bool(final_nll < initial_nll and final_accuracy >= 0.75),
            "mean_delta_r_init": float(delta_init.mean()),
            "delta_r_init_seed_percentile_interval": percentile_interval(delta_init),
            "mean_delta_r_t": float(delta_t.mean()),
            "delta_r_t_seed_percentile_interval": percentile_interval(delta_t),
            "targeted_lower_r_init_seed_count": int(sum(r["targeted_lower_at_init"] for r in primary_effect)),
            "targeted_lower_r_t_seed_count": int(sum(r["targeted_lower_at_t"] for r in primary_effect)),
            "seed_count": len(primary_effect),
            "pearson_r_init_vs_r_t_all_methods": assoc["pearson_r_init_vs_r_t"],
            "spearman_r_init_vs_r_t_all_methods": assoc["spearman_r_init_vs_r_t"],
            "hypothesis_direction_supported_by_means": bool(delta_init.mean() > 0.0 and delta_t.mean() > 0.0),
            "G1_targeted_minus_wrong_mean_log_R_init": float(log_delta_init.mean()),
            "G1_paired_bootstrap_ci95": log_init_ci,
            "G1_targeted_lower_win_fraction": float(
                np.mean([r["targeted_lower_at_init"] for r in primary_effect])
            ),
            "G1_pass": bool(
                log_init_ci[1] < 0.0
                and np.mean([r["targeted_lower_at_init"] for r in primary_effect]) >= 0.80
            ),
            "G2_targeted_minus_wrong_mean_log_R_t": float(log_delta_t.mean()),
            "G2_paired_bootstrap_ci95": log_t_ci,
            "G2_targeted_lower_win_fraction": float(
                np.mean([r["targeted_lower_at_t"] for r in primary_effect])
            ),
            "G2_pass": bool(
                log_t_ci[1] < 0.0
                and np.mean([r["targeted_lower_at_t"] for r in primary_effect]) >= 0.80
            ),
        }

    g3_rows = [
        row
        for row in vector_retention_rows
        if float(row["coverage_v"]) == cfg.primary_coverage_v
        and int(row["supervised_horizon"]) == cfg.primary_horizon
    ]
    g3_values = np.asarray(
        [row["vector_difference_retention_ratio"] for row in g3_rows], dtype=np.float64
    )
    g3_ci = paired_bootstrap_mean_ci(
        g3_values,
        cfg.bootstrap_repetitions,
        hierarchical_rng(
            cfg, RNG_GLOBAL_SEED_INDEX, RNGRole.BOOTSTRAP_G3_MEAN, "linear"
        ),
    ) if len(g3_values) else (float("nan"), float("nan"))
    g3_median = float(np.median(g3_values)) if len(g3_values) else float("nan")
    g3_median_ci = paired_bootstrap_median_ci(
        g3_values,
        cfg.bootstrap_repetitions,
        hierarchical_rng(
            cfg, RNG_GLOBAL_SEED_INDEX, RNGRole.BOOTSTRAP_G3_MEDIAN, "linear"
        ),
    ) if len(g3_values) else (float("nan"), float("nan"))

    g4_rows = [
        row
        for row in intervention_slope_rows
        if int(row["supervised_horizon"]) == cfg.primary_horizon
    ]
    g4_values = np.asarray(
        [row["log_R_t_on_log_R_init_slope"] for row in g4_rows], dtype=np.float64
    )
    g4_ci = paired_bootstrap_mean_ci(
        g4_values,
        cfg.bootstrap_repetitions,
        hierarchical_rng(
            cfg, RNG_GLOBAL_SEED_INDEX, RNGRole.BOOTSTRAP_G4_SLOPE, "linear"
        ),
    ) if len(g4_values) else (float("nan"), float("nan"))
    g4_adjacent_wins = int(sum(row["adjacent_order_pairs_won"] for row in g4_rows))
    g4_adjacent_total = int(sum(row["adjacent_order_pairs_total"] for row in g4_rows))
    g4_adjacent_fraction = g4_adjacent_wins / max(1, g4_adjacent_total)
    g4_retention_primary = [
        row
        for row in intervention_retention_rows
        if int(row["supervised_horizon"]) == cfg.primary_horizon
    ]
    g4_retention_values = np.asarray(
        [row["difference_retention_ratio"] for row in g4_retention_primary], dtype=np.float64
    )
    g4_retention_ci = paired_bootstrap_mean_ci(
        g4_retention_values,
        cfg.bootstrap_repetitions,
        hierarchical_rng(
            cfg, RNG_GLOBAL_SEED_INDEX, RNGRole.BOOTSTRAP_G4_RETENTION, "linear"
        ),
    ) if len(g4_retention_values) else (float("nan"), float("nan"))

    angle_spearman = np.asarray(
        [row["spearman_log_R_init_vs_log_R_t"] for row in angle_dose_rows],
        dtype=np.float64,
    )
    angle_median = float(np.median(angle_spearman)) if len(angle_spearman) else None
    angle_median_ci = paired_bootstrap_median_ci(
        angle_spearman,
        cfg.bootstrap_repetitions,
        hierarchical_rng(
            cfg, RNG_GLOBAL_SEED_INDEX, RNGRole.BOOTSTRAP_ANGLE_MEDIAN, "linear"
        ),
    ) if len(angle_spearman) else (None, None)
    return {
        "interpretation_guardrail": (
            "A positive endpoint supports finite-horizon persistence under actual CE/SGD "
            "within this controlled task."
        ),
        "architectures": architecture_summaries,
        "linear_vector_retention_G3": {
            "mean_ratio": float(g3_values.mean()) if len(g3_values) else float("nan"),
            "paired_bootstrap_ci95": g3_ci,
            "median_ratio": g3_median,
            "paired_bootstrap_median_ci95": g3_median_ci,
            "threshold": 0.05,
            "median_threshold": 0.10,
            "pass": bool(g3_ci[0] > 0.05 and g3_median > 0.10),
            "seed_count": len(g3_values),
        },
        "linear_state_intervention_G4": {
            "mean_log_log_slope": float(g4_values.mean()) if len(g4_values) else float("nan"),
            "paired_bootstrap_ci95": g4_ci,
            "threshold": 0.0,
            "adjacent_order_pairs_won": g4_adjacent_wins,
            "adjacent_order_pairs_total": g4_adjacent_total,
            "adjacent_order_win_fraction": g4_adjacent_fraction,
            "alpha_2_vs_0.25_mean_response_difference_retention": float(
                g4_retention_values.mean()
            ) if len(g4_retention_values) else float("nan"),
            "alpha_2_vs_0.25_bootstrap_ci95": g4_retention_ci,
            "response_difference_retention_threshold": 0.05,
            "pass": bool(
                g4_ci[0] > 0.0
                and g4_adjacent_fraction >= 0.80
                and g4_retention_ci[0] > 0.05
            ),
            "seed_count": len(g4_values),
        },
        "angle_dose_gate": {
            "median_seed_spearman_log_R_init_vs_log_R_t": angle_median,
            "paired_bootstrap_median_ci95": angle_median_ci,
            "median_threshold": 0.60,
            "ci_lower_threshold": 0.30,
            "pass": bool(angle_median > 0.60 and angle_median_ci[0] > 0.30)
            if angle_median is not None
            else None,
            "seed_count": len(angle_spearman),
            "not_applicable": not bool(len(angle_spearman)),
        },
        "row_counts": {
            "supervised_pairs": len(supervised_pairs),
            "paired_effects": len(paired_effects),
            "association": len(association_rows),
            "vector_retention": len(vector_retention_rows),
            "intervention_slopes": len(intervention_slope_rows),
            "intervention_vector_retention": len(intervention_retention_rows),
            "angle_dose_seed_rows": len(angle_dose_rows),
        },
    }


def run_experiment(cfg: ExperimentConfig, out: Path, mode: str) -> dict[str, Any]:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is non-empty; choose a new path: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    rng_audit_rows, rng_audit_summary = audit_rng_namespaces()

    residual_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    optimization_rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    invariance_rows: list[dict[str, Any]] = []
    spectrum_audit_rows: list[dict[str, Any]] = []
    dense_linear_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []

    for architecture in cfg.architectures:
        architecture_seeds = cfg.seeds_for(architecture)
        batch_size = cfg.batch_size_for(architecture)
        warmup_steps = cfg.warmup_steps_for(architecture)
        horizons = cfg.horizons_for(architecture)
        max_supervised_steps = max(horizons)
        lr = model_lr(architecture, cfg)
        for seed in range(architecture_seeds):
            rng_basis = hierarchical_rng(
                cfg, seed, RNGRole.HAAR_BASIS, architecture
            )
            basis = make_haar_rotation(rng_basis, cfg.input_dim)
            gram = np.einsum("ia,ib->ab", basis, basis, optimize=True)
            haar_error = float(np.max(np.abs(gram - np.eye(cfg.input_dim))))
            for direction_idx in range(cfg.input_dim):
                direction_name = DIRECTIONS[direction_idx] if direction_idx < 3 else f"aux_nuisance_{direction_idx - 2}"
                for coordinate in range(cfg.input_dim):
                    basis_rows.append(
                        {
                            "architecture": architecture,
                            "seed": seed,
                            "direction": direction_name,
                            "ambient_coordinate": coordinate,
                            "value": float(basis[coordinate, direction_idx]),
                        }
                    )

            rng_warm = hierarchical_rng(
                cfg, seed, RNGRole.WARMUP_DATA, architecture
            )
            batch_generator = (
                strict_label_invariant_batches
                if cfg.strict_finite_batch_label_invariance
                else iid_batches
            )
            if cfg.strict_finite_batch_label_invariance:
                y_warm, _, warm_z = batch_generator(
                    rng_warm, warmup_steps, batch_size, 2
                )
            else:
                y_warm, _, warm_z = iid_batches(
                    rng_warm, warmup_steps, batch_size, 2, balanced_labels=True
                )
            warm_z = whiten_warmup_batches_exactly(warm_z)
            warm_targets = one_hot(y_warm.reshape(-1)).reshape(warmup_steps, batch_size, 2)
            spectrum_audit_rows.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    **matched_spectrum_audit(warm_z, basis, cfg),
                }
            )
            rng_train = hierarchical_rng(
                cfg, seed, RNGRole.SUPERVISED_DATA, architecture
            )
            if cfg.strict_finite_batch_label_invariance:
                y_train, eps_sem_train, z_train_nuisance = strict_label_invariant_batches(
                    rng_train, max_supervised_steps, batch_size, cfg.input_dim - 1
                )
            else:
                y_train, eps_sem_train, z_train_nuisance = iid_batches(
                    rng_train, max_supervised_steps, batch_size, cfg.input_dim - 1
                )
            z_w_train = z_train_nuisance[:, :, 0]
            z_v_train = z_train_nuisance[:, :, 1]
            z_aux_train = z_train_nuisance[:, :, 2:]
            invariance_rows.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "phase": "supervised",
                    "haar_orthogonality_max_abs_error": haar_error,
                    **finite_batch_invariance_audit(y_train, z_train_nuisance),
                }
            )
            invariance_rows.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "phase": "warmup",
                    "haar_orthogonality_max_abs_error": haar_error,
                    **finite_batch_invariance_audit(y_warm, warm_z),
                }
            )

            rng_eval = hierarchical_rng(
                cfg, seed, RNGRole.EVALUATION_DATA, architecture
            )
            if cfg.strict_finite_batch_label_invariance:
                y_eval, eps_sem_eval, z_eval_nuisance = strict_label_invariant_evaluation(
                    rng_eval, cfg.eval_size, cfg.input_dim - 1
                )
            else:
                y_eval_batch, eps_eval_batch, z_eval_batch = iid_batches(
                    rng_eval, 1, cfg.eval_size, cfg.input_dim - 1
                )
                y_eval, eps_sem_eval, z_eval_nuisance = (
                    y_eval_batch[0],
                    eps_eval_batch[0],
                    z_eval_batch[0],
                )
            z_w_eval = z_eval_nuisance[:, 0]
            z_v_eval = z_eval_nuisance[:, 1]
            z_aux_eval = z_eval_nuisance[:, 2:]
            x_ref, y_ref = reference_from_base(
                y_eval,
                eps_sem_eval,
                z_w_eval,
                z_v_eval,
                z_aux_eval,
                basis,
                cfg.reference_var_v,
                cfg,
            )
            x_shift, y_shift = reference_from_base(
                y_eval,
                eps_sem_eval,
                z_w_eval,
                z_v_eval,
                z_aux_eval,
                basis,
                cfg.shifted_var_v,
                cfg,
            )
            rng_init = hierarchical_rng(
                cfg, seed, RNGRole.MODEL_INITIALIZATION, architecture
            )
            initial_model = model_factory(architecture, rng_init, cfg)
            x_residual_probe = (
                x_ref if architecture == "linear" else x_ref[: cfg.mlp_probe_size]
            )

            for coverage_v in cfg.coverage_v_grid:
                x_train, y_train_soft = supervised_from_base(
                    y_train,
                    eps_sem_train,
                    z_w_train,
                    z_v_train,
                    z_aux_train,
                    basis,
                    coverage_v,
                    cfg,
                )
                x_train_eval, y_train_eval = reference_from_base(
                    y_eval,
                    eps_sem_eval,
                    z_w_eval,
                    z_v_eval,
                    z_aux_eval,
                    basis,
                    coverage_v,
                    cfg,
                )
                eval_sets = {
                    "train_like": (x_train_eval, y_train_eval),
                    "reference_unit_v": (x_ref, y_ref),
                    "shifted_v4": (x_shift, y_shift),
                }

                for method in cfg.methods:
                    model = initial_model.copy()
                    append_linear_dense_residual_rows(
                        model=model,
                        seed=seed,
                        coverage_v=coverage_v,
                        method=method,
                        phase="random_init",
                        phase_step=0,
                        global_step=0,
                        basis=basis,
                        rows=dense_linear_rows,
                    )
                    append_state_rows(
                        model=model,
                        architecture=architecture,
                        seed=seed,
                        coverage_v=coverage_v,
                        method=method,
                        stage="random_init",
                        supervised_step=-1,
                        global_step=0,
                        basis=basis,
                        x_ref=x_residual_probe,
                        evaluation_sets=eval_sets,
                        residual_rows=residual_rows,
                        metric_rows=metric_rows,
                        response_rows=response_rows,
                    )

                    for step in range(warmup_steps):
                        x_warm = warmup_batch(method, warm_z[step], basis, cfg)
                        if method == "no_warmup":
                            # An explicit no-op avoids Adam amplifying round-off-only
                            # gradients from an otherwise exact zero-input control.
                            loss = math.log(2.0)
                        else:
                            loss = model.step(x_warm, warm_targets[step], lr)
                        if (step + 1) % cfg.trace_stride == 0 or step + 1 == warmup_steps:
                            optimization_rows.append(
                                {
                                    "architecture": architecture,
                                    "seed": seed,
                                    "coverage_v": coverage_v,
                                    "method": method,
                                    "phase": "warmup_balanced_hard_label_ce",
                                    "phase_step": step + 1,
                                    "loss": loss,
                                }
                            )
                            append_linear_dense_residual_rows(
                                model=model,
                                seed=seed,
                                coverage_v=coverage_v,
                                method=method,
                                phase="warmup",
                                phase_step=step + 1,
                                global_step=step + 1,
                                basis=basis,
                                rows=dense_linear_rows,
                            )

                    append_state_rows(
                        model=model,
                        architecture=architecture,
                        seed=seed,
                        coverage_v=coverage_v,
                        method=method,
                        stage="post_warmup",
                        supervised_step=0,
                        global_step=warmup_steps,
                        basis=basis,
                        x_ref=x_residual_probe,
                        evaluation_sets=eval_sets,
                        residual_rows=residual_rows,
                        metric_rows=metric_rows,
                        response_rows=response_rows,
                    )

                    # The MLP formal protocol resets Adam moments between stages.
                    model.reset_optimizer()
                    checkpoints = set(horizons) - {0}
                    for step in range(1, max_supervised_steps + 1):
                        loss = model.step(x_train[step - 1], y_train_soft[step - 1], lr)
                        if step % cfg.trace_stride == 0 or step in checkpoints:
                            optimization_rows.append(
                                {
                                    "architecture": architecture,
                                    "seed": seed,
                                    "coverage_v": coverage_v,
                                    "method": method,
                                    "phase": "supervised_hard_label_ce",
                                    "phase_step": step,
                                    "loss": loss,
                                }
                            )
                        if step % cfg.trace_stride == 0 or step == max_supervised_steps:
                            append_linear_dense_residual_rows(
                                model=model,
                                seed=seed,
                                coverage_v=coverage_v,
                                method=method,
                                phase="supervised",
                                phase_step=step,
                                global_step=warmup_steps + step,
                                basis=basis,
                                rows=dense_linear_rows,
                            )
                        if step in checkpoints:
                            append_state_rows(
                                model=model,
                                architecture=architecture,
                                seed=seed,
                                coverage_v=coverage_v,
                                method=method,
                                stage="supervised",
                                supervised_step=step,
                                global_step=warmup_steps + step,
                                basis=basis,
                                x_ref=x_residual_probe,
                                evaluation_sets=eval_sets,
                                residual_rows=residual_rows,
                                metric_rows=metric_rows,
                                response_rows=response_rows,
                            )

                # A separate state intervention tests r_init -> r_T causally,
                # without relying on the warm-up construction.
                if architecture == "linear" and coverage_v == cfg.primary_coverage_v:
                    assert isinstance(initial_model, LinearSoftmax)
                    v = basis[:, 2]
                    base_orthogonal_responses = {
                        idx: np.einsum("d,dk->k", basis[:, idx], initial_model.w, optimize=True)
                        for idx in range(cfg.input_dim)
                        if idx != 2
                    }
                    for alpha in cfg.intervention_alphas:
                        model, base_r2, r2_init = intervene_linear_v_state(initial_model, v, alpha)
                        init_response = np.einsum("d,dk->k", v, model.w, optimize=True)
                        init_centered = init_response - init_response.mean()
                        orthogonal_change = max(
                            float(
                                np.max(
                                    np.abs(
                                        np.einsum(
                                            "d,dk->k", basis[:, idx], model.w, optimize=True
                                        )
                                        - base_response
                                    )
                                )
                            )
                            for idx, base_response in base_orthogonal_responses.items()
                        )
                        scale_error = abs(r2_init - alpha * alpha * base_r2)
                        intervention_rows.append(
                            {
                                "seed": seed,
                                "coverage_v": coverage_v,
                                "alpha": alpha,
                                "supervised_horizon": 0,
                                "R_init_v": r2_init,
                                "R_t_v": r2_init,
                                "r_init_v": math.sqrt(max(0.0, r2_init)),
                                "r_t_v": math.sqrt(max(0.0, r2_init)),
                                "log_R_init_eps": math.log(r2_init + cfg.log_epsilon),
                                "log_R_t_eps": math.log(r2_init + cfg.log_epsilon),
                                "orthogonal_response_max_abs_change": orthogonal_change,
                                "R_scale_absolute_error": scale_error,
                                "centered_response_0": float(init_centered[0]),
                                "centered_response_1": float(init_centered[1]),
                            }
                        )
                        checkpoints = set(cfg.supervised_horizons) - {0}
                        for step in range(1, max(cfg.supervised_horizons) + 1):
                            model.step(x_train[step - 1], y_train_soft[step - 1], cfg.linear_lr)
                            if step in checkpoints:
                                r2_t = model.direction_r2(x_ref, v)
                                t_response = np.einsum("d,dk->k", v, model.w, optimize=True)
                                t_centered = t_response - t_response.mean()
                                intervention_rows.append(
                                    {
                                        "seed": seed,
                                        "coverage_v": coverage_v,
                                        "alpha": alpha,
                                        "supervised_horizon": step,
                                        "R_init_v": r2_init,
                                        "R_t_v": r2_t,
                                        "r_init_v": math.sqrt(max(0.0, r2_init)),
                                        "r_t_v": math.sqrt(max(0.0, r2_t)),
                                        "log_R_init_eps": math.log(r2_init + cfg.log_epsilon),
                                        "log_R_t_eps": math.log(r2_t + cfg.log_epsilon),
                                        "orthogonal_response_max_abs_change": orthogonal_change,
                                        "R_scale_absolute_error": scale_error,
                                        "centered_response_0": float(t_centered[0]),
                                        "centered_response_1": float(t_centered[1]),
                                    }
                                )

    supervised_pairs, paired_effects, association_rows = derive_pair_tables(residual_rows, cfg)
    vector_retention_rows = derive_vector_retention(response_rows, cfg)
    intervention_slope_rows = derive_intervention_slopes(intervention_rows, cfg)
    intervention_retention_rows = derive_intervention_vector_retention(intervention_rows, cfg)
    angle_dose_rows = derive_angle_dose_seed_rows(supervised_pairs, cfg)
    summary = aggregate_summary(
        supervised_pairs,
        paired_effects,
        association_rows,
        metric_rows,
        vector_retention_rows,
        intervention_slope_rows,
        intervention_retention_rows,
        angle_dose_rows,
        cfg,
    )
    summary["rng_namespace_audit"] = rng_audit_summary

    write_csv(out / "tables" / "directional_residual_trajectory.csv", residual_rows)
    write_csv(out / "tables" / "predictive_metrics.csv", metric_rows)
    write_csv(out / "tables" / "optimization_trace.csv", optimization_rows)
    write_csv(out / "tables" / "rng_stream_audit.csv", rng_audit_rows)
    write_csv(out / "tables" / "random_rotation_bases.csv", basis_rows)
    write_csv(out / "tables" / "finite_batch_label_invariance_audit.csv", invariance_rows)
    write_csv(out / "tables" / "warmup_matched_spectrum_audit.csv", spectrum_audit_rows)
    if dense_linear_rows:
        write_csv(out / "tables" / "linear_dense_directional_trajectory.csv", dense_linear_rows)
    if response_rows:
        write_csv(out / "tables" / "linear_v_centered_response_vectors.csv", response_rows)
    write_csv(out / "tables" / "r_init_to_r_t_pairs.csv", supervised_pairs)
    write_csv(out / "tables" / "paired_method_effects.csv", paired_effects)
    write_csv(out / "tables" / "r_init_r_t_association.csv", association_rows)
    if vector_retention_rows:
        write_csv(out / "tables" / "linear_vector_difference_retention.csv", vector_retention_rows)
    if intervention_rows:
        write_csv(out / "tables" / "linear_state_intervention_trajectory.csv", intervention_rows)
        write_csv(out / "tables" / "linear_state_intervention_seed_slopes.csv", intervention_slope_rows)
        write_csv(
            out / "tables" / "linear_state_intervention_vector_retention.csv",
            intervention_retention_rows,
        )
    if angle_dose_rows:
        write_csv(out / "tables" / "angle_dose_seed_spearman.csv", angle_dose_rows)

    protocol = {
        "mode": mode,
        "config": asdict(cfg),
        "rng_namespace": rng_audit_summary,
        "hypotheses_frozen_before_formal_outcome_inspection": [
            "At weak v coverage, targeted_v yields lower r_init(v) than wrong_w under matched budgets.",
            "After identical finite-horizon supervised CE/SGD, targeted_v retains lower r_T(v) than wrong_w.",
            "Across matched initial states, lower r_init(v) predicts lower r_T(v).",
            "The targeted-v advantage diminishes as supervised v coverage or training horizon increases.",
            "A direct linear state intervention that changes only centered v response yields positive log R_init to log R_T slopes.",
        ],
        "primary_endpoint": {
            "comparison": "wrong_w minus targeted_v",
            "coverage_v": cfg.primary_coverage_v,
            "linear_horizon": cfg.primary_horizon,
            "mlp_horizon": cfg.mlp_primary_horizon,
            "quantity": "directional centered-logit Jacobian norm r(v)",
        },
        "warmup_budget_note": (
            "All active warm-up groups use the same number of iterations and batch size. The no_warmup "
            "control records the same scheduled iterations as explicit parameter no-ops; this avoids Adam "
            "amplifying floating-point round-off from a nominal zero-gradient dummy batch."
        ),
        "finite_sample_label_invariance": (
            "Every supervised and warm-up batch uses quartets (y,z)=(0,z),(0,-z),(1,z),(1,-z), "
            "then a shared shuffle; nuisance empirical distributions are exactly identical across labels."
            if cfg.strict_finite_batch_label_invariance
            else (
                "Secondary IID robustness suite: supervised and evaluation batches are ordinary IID draws; "
                "warm-up labels remain exactly balanced and its 2-D Gaussian base is whitened within each "
                "batch to preserve the matched-spectrum control."
            )
        ),
        "seed_namespace": {
            "active_seed_base": cfg.seed_base,
            "formal_seed_base": 20270810,
            "smoke_seed_base": 20260810,
            "separation_reason": (
                "v3 uses explicit disjoint hierarchical SeedSequence namespaces across mode, suite, seed, "
                "role, architecture, and substream; prior additive-offset formal results are superseded."
            ),
        },
        "finite_batch_matched_spectrum": (
            "Each 2-D warm-up base batch is centered and whitened to identity empirical covariance before "
            "the frozen eigenvalues are applied. Targeted_v and wrong_w therefore have identical per-batch "
            "empirical covariance eigenvalues, trace, and determinant up to a mandatory 1e-10 audit tolerance."
        ),
        "randomness_pairing": (
            "Within seed, all methods share the exact model initialization, random rotation, warm-up base draws, "
            "supervised labels/examples, and evaluation samples."
        ),
        "angle_dose_mapping_degrees_from_v_toward_w": {
            "targeted_v": 0.0,
            "angle_22.5": 22.5,
            "angle_45": 45.0,
            "angle_67.5": 67.5,
            "wrong_w": 90.0,
        },
    }
    json_dump(out / "protocol.json", protocol)
    json_dump(out / "summary.json", summary)
    return summary


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument(
        "--suite",
        choices=("primary", "coverage_horizon", "angle_dose", "iid_robustness"),
        default="primary",
        help="Frozen primary, boundary scan, angle dose, or secondary IID robustness suite.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Must be a new or empty directory; existing results are never overwritten.",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=("linear", "mlp_tanh"),
        default=None,
        help="Optional subset, recorded in protocol.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = suite_config(args.mode, args.suite)
    if args.architectures:
        cfg = replace(cfg, architectures=tuple(args.architectures))
    summary = run_experiment(cfg, args.output_dir.resolve(), f"{args.mode}:{args.suite}")
    script_path = Path(__file__).resolve()
    manifest = {
        "script": str(script_path),
        "script_sha256": sha256(script_path),
        "python": sys.version,
        "numpy": np.__version__,
        "output_dir": str(args.output_dir.resolve()),
        "rng_namespace_version": RNG_NAMESPACE_VERSION,
        "rng_registry_sha256": summary["rng_namespace_audit"]["registry_sha256"],
        "summary": summary,
    }
    json_dump(args.output_dir.resolve() / "manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
