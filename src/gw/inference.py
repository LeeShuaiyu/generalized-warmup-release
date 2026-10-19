"""Checkpoint inference using the fixed paper split and preprocessing."""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import shutil

import numpy as np

from .presentation import sha256
from .data_assets import verify_inputs
from .assets import relative_path


def split_hash(rows):
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["sample_id"]):
        stable = {key: value for key, value in row.items() if not key.endswith("_path") and key != "path"}
        digest.update(json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def preprocess_array(path, image_size):
    from PIL import Image
    array = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        array = np.zeros_like(array)
    else:
        low = float(np.percentile(array[finite], 1))
        high = float(np.percentile(array[finite], 99))
        array = np.zeros_like(array) if high <= low else np.clip((array - low) / (high - low), 0, 1).astype(np.float32)
    if array.ndim == 3:
        array = array.mean(axis=2)
    if array.ndim != 2:
        raise ValueError(f"Expected grayscale array: {path}")
    if array.shape != (image_size, image_size):
        image = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8), mode="L")
        array = np.asarray(image.resize((image_size, image_size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.
    return array


def preflight_experiment(root, config, data_root=None, checkpoint_root=None, output_root=None):
    """Check fixed assets and space without importing or executing a model."""
    root = Path(root).resolve()
    data_root = Path(data_root) if data_root else root / "data" / "processed" / "opensar"
    checkpoint_root = Path(checkpoint_root) if checkpoint_root else root / "artifacts" / "checkpoints" / config["experiment_id"]
    with (root / "data" / "manifests" / "opensarship2.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if split_hash(rows) != config["manifest_sha256"]:
        raise ValueError("Fixed data split checksum mismatch")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate dataset samples")
    selected = [row for row in rows if row["split"] == config["evaluation_split"]]
    if len(selected) != config["evaluation_samples"]:
        raise ValueError("Unexpected evaluation sample count")
    asset_manifest = json.loads((root / "artifacts" / "manifest.json").read_text())
    jobs = []
    for condition in config["conditions"]:
        for seed in config["seeds"]:
            filename = f"{condition['id']}_{seed}.pt"
            path = checkpoint_root / filename
            key = f"{config['experiment_id']}/{condition['id']}/{seed}"
            if not path.exists():
                raise FileNotFoundError(f"Required model asset: {path}")
            if sha256(path) != asset_manifest["checkpoints"][key]["sha256"]:
                raise ValueError(f"Checkpoint checksum mismatch: {key}")
            jobs.append((condition, seed, path))
    checked = verify_inputs(root, 'opensar', rows, config['evaluation_split'], data_root)
    base = Path(output_root) if output_root else root / 'output' / relative_path(config['experiment_id'])
    if output_root is None and not base.resolve().is_relative_to(root):
        raise ValueError('Inference output resolves outside the project')
    ancestor = base
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < 8 * 1024**3 + len(jobs) * len(selected) * (config['num_classes'] * 4 + 8):
        raise RuntimeError('Keep at least 8 GiB available after inference')
    return data_root, selected, jobs, checked, base


def run_experiment(root, config, data_root=None, checkpoint_root=None, output_root=None, device="auto"):
    if device not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('Unknown device')
    root = Path(root).resolve()
    data_root, selected, jobs, checked, base = preflight_experiment(
        root, config, data_root, checkpoint_root, output_root)
    import torch
    from torch import nn
    from torchvision.models import densenet121, resnet18
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device is unavailable")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    class Images(torch.utils.data.Dataset):
        def __len__(self):
            return len(selected)

        def __getitem__(self, index):
            row = selected[index]
            if row.get("data_type") != "npy_float32":
                raise ValueError("This experiment requires the fixed float32 image arrays")
            array = preprocess_array(data_root / row["data_path"], config["image_size"])
            image = torch.from_numpy(array[None]).float()
            normalization = config.get("normalization")
            if normalization:
                image = (image - normalization["mean"]) / normalization["std"]
            return image, int(row["class_id"])

    loader = torch.utils.data.DataLoader(Images(), batch_size=config["batch_size"], shuffle=False, num_workers=0)
    output = base / datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for condition, seed, checkpoint in jobs:
        if config["architecture"] == "densenet121_gray":
            model = densenet121(weights=None)
            old = model.features.conv0
            model.features.conv0 = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
                                            padding=old.padding, bias=old.bias is not None)
            model.classifier = nn.Linear(model.classifier.in_features, config["num_classes"])
        elif config["architecture"] == "resnet18_gray":
            model = resnet18(weights=None)
            old = model.conv1
            model.conv1 = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
                                   padding=old.padding, bias=old.bias is not None)
            model.fc = nn.Linear(model.fc.in_features, config["num_classes"])
        else:
            raise ValueError(f"Unsupported architecture: {config['architecture']}")
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        model = model.to(device).eval()
        logits_parts, label_parts = [], []
        with torch.no_grad():
            for images, labels in loader:
                logits_parts.append(model(images.to(device)).detach().cpu().numpy())
                label_parts.append(labels.numpy())
        filename = f"{condition['id']}_{seed}.npz"
        np.savez_compressed(output / filename, logits=np.concatenate(logits_parts), labels=np.concatenate(label_parts))
        records.append({"condition": condition["id"], "seed": seed, "device": device,
                        "checkpoint_sha256": sha256(checkpoint), "predictions_sha256": sha256(output / filename)})
        print(f"Inference {len(records)}/{len(jobs)} · {condition['label']} · seed {seed}", flush=True)
        del model
    (output / "run.json").write_text(json.dumps({"experiment_id": config["experiment_id"], "split": config["evaluation_split"],
                                                "samples": len(selected), "input_checks": checked,
                                                "records": records}, indent=2) + "\n")
    return output


def run_family(root, config, **kwargs):
    return run_experiment(root, config, **kwargs)
