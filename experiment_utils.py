"""Small shared utilities for reproducible runs and evaluation preflight."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import torch

from sampling import sigma_tag, validate_sampling


def write_json(path, value):
    """Replace a complete JSON document atomically; never publish NaN scores."""
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


@contextmanager
def record_run(path):
    """Keep setup failures and interrupts from masquerading as running jobs."""
    path = Path(path)
    started = datetime.now(timezone.utc).isoformat()
    initial = {"status": "running", "started_at_utc": started}
    write_json(path, initial)
    try:
        yield initial
    except BaseException as error:
        state = json.loads(path.read_text())
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=f"{type(error).__name__}: {error}", started_at_utc=started,
                     finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(path, state)
        raise
    else:
        state = json.loads(path.read_text())
        state.update(status="completed", started_at_utc=started)
        state.setdefault("finished_at_utc", datetime.now(timezone.utc).isoformat())
        write_json(path, state)


def configure_determinism():
    # Set before the first cuBLAS operation; keep an explicitly supplied value.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_evaluation_sweep(sweep_dir, root, n=10000):
    """Reject partial/mismatched inputs before fitting or loading an evaluator."""
    sweep_dir, root = Path(sweep_dir), Path(root)
    sweep = json.loads((sweep_dir / "summary.json").read_text())
    if sweep.get("status") != "completed":
        raise ValueError("Sweep must be completed before evaluation")
    config, conditions = sweep["config"], sweep["results"]
    if config["num_samples"] != n:
        raise ValueError(f"Local metric protocol requires {n} generated images per condition")
    if (len(set(config["seeds"])) != len(config["seeds"])
            or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in config["seeds"])
            or len(set(config["sigmas"])) != len(config["sigmas"])):
        raise ValueError("Sweep seeds and scales must be valid and unique")
    expected = {(seed, sigma) for seed in config["seeds"] for sigma in config["sigmas"]}
    actual = [(row["seed"], row["sigma"]) for row in conditions]
    if not expected or set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Sweep conditions are incomplete or duplicated")
    checkpoint = root / sweep["checkpoint"]
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != sweep["checkpoint_sha256"]:
        raise ValueError("Generator checkpoint hash mismatch")
    names = {f"{i}.png" for i in range(n)}
    for condition in conditions:
        validate_sampling(n, config["batch_size"], condition["sigma"])
        if condition["accepted_images"] != n:
            raise ValueError(f"Expected {n} samples in every condition")
        if condition["checkpoint_sha256"] != sweep["checkpoint_sha256"]:
            raise ValueError("Conditions use different generator checkpoints")
        directory = sweep_dir / f"seed-{condition['seed']}" / f"sigma-{sigma_tag(condition['sigma'])}" / "samples"
        if not directory.is_dir() or {p.name for p in directory.iterdir()} != names:
            raise ValueError(f"Missing or incorrectly numbered PNGs: {directory}")
    return sweep


def evaluation_conditions(*, root, sweep_dir=None, manifest=None, n=10000):
    """Normalize generic manifests or legacy sweeps without assuming a method.

    Generic manifest paths are relative to the manifest file, not the shell's
    working directory. Keep arbitrary method configuration as provenance.
    """
    if (sweep_dir is None) == (manifest is None):
        raise ValueError("Provide exactly one evaluation manifest or sweep")
    if sweep_dir is not None:
        sweep_dir = Path(sweep_dir).resolve()
        sweep = validate_evaluation_sweep(sweep_dir, root, n)
        return [{"run_id": row["run_id"], "method": row.get("method", sweep.get("method", "unspecified")),
                 "seed": row["seed"], "sigma": row["sigma"],
                 "config": row.get("config", sweep["config"]),
                 "checkpoint": str((Path(root) / sweep["checkpoint"]).resolve()),
                 "checkpoint_sha256": sweep["checkpoint_sha256"],
                 "samples": str(sweep_dir / f"seed-{row['seed']}" /
                                f"sigma-{sigma_tag(row['sigma'])}" / "samples")}
                for row in sweep["results"]]

    manifest = Path(manifest).resolve()
    rows = json.loads(manifest.read_text())["conditions"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must contain a nonempty conditions list")
    conditions, seen = [], set()
    expected_names = {f"{i}.png" for i in range(n)}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every condition must be an object")
        run_id = row["run_id"]
        if not isinstance(run_id, str) or not run_id.strip() or run_id in seen:
            raise ValueError("Condition run IDs must be nonempty unique strings")
        if not isinstance(row["method"], str) or not row["method"].strip():
            raise ValueError("Every condition must declare its method")
        if not isinstance(row["config"], dict):
            raise ValueError("Every condition must declare a configuration object")
        samples = (manifest.parent / row["samples"]).resolve()
        checkpoint = (manifest.parent / row["checkpoint"]).resolve()
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != row["checkpoint_sha256"]:
            raise ValueError(f"Generator checkpoint hash mismatch: {run_id}")
        if not samples.is_dir() or {p.name for p in samples.iterdir()} != expected_names:
            raise ValueError(f"Expected exactly {n} numbered PNGs: {samples}")
        conditions.append({**row, "samples": str(samples), "checkpoint": str(checkpoint)})
        seen.add(run_id)
    return conditions
