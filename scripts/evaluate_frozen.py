"""Evaluate declared sample sets with the frozen MNIST64/k3 protocol."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torchvision.datasets import MNIST

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_utils import configure_determinism, record_run, write_json
from experiment_utils import evaluation_conditions
from local_metrics import MNISTFeatures, frechet_distance, knn_radii, precision_recall
from scripts.evaluate_local import digest, extract, read_pngs


def load_frozen_evaluator(archive, data_path):
    """Verify immutable artifacts and rebuild the exact real reference features."""
    archive = Path(archive)
    protocol = json.loads((archive / "summary.json").read_text())
    if protocol["protocol"] != "local-MNIST64-k3-v1" or protocol["official"]:
        raise ValueError("Unexpected evaluator protocol")
    if digest(archive / "extractor.pth") != protocol["extractor"]["checkpoint_sha256"]:
        raise ValueError("Extractor checkpoint hash mismatch")
    if digest(archive / "split-indices.npz") != protocol["data"]["split_sha256"]:
        raise ValueError("Data split hash mismatch")
    if digest(ROOT / "local_metrics.py") != protocol["source_sha256"]["local_metrics.py"]:
        raise ValueError("Metric implementation differs from the frozen protocol")
    dataset = MNIST(str(data_path), train=True, download=False)
    for name, expected in protocol["data"]["raw_sha256"].items():
        if digest(Path(dataset.raw_folder) / name) != expected:
            raise ValueError(f"Dataset hash mismatch: {name}")
    with np.load(archive / "split-indices.npz") as split:
        indices = torch.from_numpy(split["real_reference"].copy())
    model = MNISTFeatures()
    model.load_state_dict(torch.load(archive / "extractor.pth", map_location="cpu", weights_only=True))
    model.eval().requires_grad_(False)
    images = dataset.data[indices].unsqueeze(1).float().div(255).sub(0.5).div(0.5)
    features, _, _ = extract(model, images)
    return model, features, protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--sweep", type=Path, help="Legacy paired latent-scaling sweep.")
    inputs.add_argument("--manifest", type=Path, help="Method-independent sample/checkpoint manifest.")
    parser.add_argument("--evaluator", type=Path, default=ROOT / "experiments/local-metrics-mnist64-k3")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()
    if min(args.threads, args.chunk_size) < 1:
        parser.error("threads and chunk-size must be positive")
    try:
        conditions = evaluation_conditions(root=ROOT, manifest=args.manifest,
                                           sweep_dir=args.sweep)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    with record_run(run / "summary.json") as run_state:
        start = time.perf_counter()
        torch.set_num_threads(args.threads)
        configure_determinism()
        model, reference, protocol = load_frozen_evaluator(args.evaluator, args.data_path)
        radii = knn_radii(reference, 3, args.chunk_size)
        source_hashes = {}
        for name in ["scripts/evaluate_frozen.py", "scripts/evaluate_local.py", "local_metrics.py", "sampling.py", "experiment_utils.py"]:
            target = run / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
            source_hashes[name] = digest(target)
        summary = {
            "protocol": protocol["protocol"], "official": False, "status": "running",
            "started_at_utc": run_state["started_at_utc"],
            "input_manifest": str(args.manifest.resolve()) if args.manifest else None,
            "input_sweep": str(args.sweep.resolve()) if args.sweep else None,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": source_hashes,
            "evaluator": str(args.evaluator), "extractor": protocol["extractor"],
            "data": protocol["data"], "metric": {**protocol["metric"], "chunk_size": args.chunk_size},
            "inherited_controls": protocol["controls"],
            "reference_features_sha256": hashlib.sha256(reference.numpy().tobytes()).hexdigest(),
            "environment": {"device": "cpu", "torch": torch.__version__, "numpy": np.__version__, "threads": args.threads},
            "extractor_retrained": False, "results": [],
        }
        write_json(run / "summary.json", summary)
        for condition in conditions:
            condition_start = time.perf_counter()
            images, image_hash = read_pngs(Path(condition["samples"]))
            features, predictions, confidence = extract(model, images)
            record = {
                **condition,
                "FD_MNIST": frechet_distance(reference.numpy(), features.numpy()),
                **precision_recall(reference, features, k=3, chunk_size=args.chunk_size, real_radii=radii),
                "image_bytes_sha256": image_hash,
                "predicted_class_counts": torch.bincount(predictions, minlength=10).tolist(),
                "mean_classifier_confidence": confidence.mean().item(),
                "evaluation_seconds": time.perf_counter() - condition_start,
            }
            summary["results"].append(record)
            write_json(run / "summary.json", summary)
            print(f"{condition['run_id']}: FD-MNIST={record['FD_MNIST']:.4f}, "
                  f"precision={record['precision']:.4%}, recall={record['recall']:.4%}", flush=True)
        summary["status"] = "completed"
        summary["total_seconds"] = time.perf_counter() - start
        if digest(args.evaluator / "extractor.pth") != protocol["extractor"]["checkpoint_sha256"]:
            raise RuntimeError("Frozen extractor file changed")
        write_json(run / "summary.json", summary)


if __name__ == "__main__":
    main()
