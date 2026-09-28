import hashlib
import json
from pathlib import Path

import torch

from .data import axiom_training_example_to_json

def dataset_fingerprint(examples):
    digest = hashlib.sha256()
    for example in examples:
        record = axiom_training_example_to_json(example)
        digest.update(json.dumps(record, sort_keys = True).encode())
        digest.update(b"\n")

    return digest.hexdigest()

def check_training_target(output, resume):
    path = Path(output) / "model.pt"
    if path.exists() and not resume:
        raise ValueError("checkpoint already exists; use --resume or a new --data-dir")

    if resume and not path.is_file():
        raise FileNotFoundError(f"cannot resume: checkpoint not found: {path}")

def snapshot_training(optimizer, random, fingerprint):
    devices = {
        parameter.device.index
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.device.type == "cuda"
    }

    return {
        "optimizer": optimizer.state_dict(),
        "shuffle_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": [torch.cuda.get_rng_state(index) for index in sorted(devices)],
        "dataset_fingerprint": fingerprint,
    }

def restore_training(path, model, optimizer, random, config, fingerprint):
    checkpoint = torch.load(path, map_location = "cpu", weights_only = True)
    state = checkpoint.get("training_state")
    if state is None:
        raise ValueError("this checkpoint has no optimizer/RNG state and cannot resume")

    mutable = {"epochs", "device", "num_workers", "log_every"}
    previous = checkpoint["training_config"]
    differences = [
        name
        for name in previous.keys() | config.keys()
        if name not in mutable and previous.get(name) != config.get(name)
    ]

    if differences or state["dataset_fingerprint"] != fingerprint:
        raise ValueError(f"resume requires the same data and configuration: {differences}")

    epoch = checkpoint["epoch"]
    if config["epochs"] <= epoch:
        raise ValueError(f"checkpoint already reached epoch {epoch}; increase --epochs")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["shuffle_rng"])
    torch.set_rng_state(state["torch_rng"])
    device = next(model.parameters()).device
    if state["cuda_rng"] and device.type == "cuda":
        old_device = str(previous.get("device", "cuda"))
        old_index = torch.device(old_device).index if old_device.startswith("cuda") else 0
        index = (old_index or 0) if len(state["cuda_rng"]) > 1 else 0
        torch.cuda.set_rng_state(state["cuda_rng"][index], device = device)

    return epoch + 1
