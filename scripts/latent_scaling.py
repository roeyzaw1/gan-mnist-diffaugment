"""Run paired, inference-only latent-scale sweeps on a frozen checkpoint."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from PIL import Image, ImageDraw
import torch
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_utils import configure_determinism, record_run, write_json
from sampling import (default_device, export_pngs, iter_candidates,
                      load_seeded_generator, sigma_tag, validate_sampling)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return os.path.relpath(path, ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/baseline-seed0-e10/G.pth")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.5, 0.7, 0.85, 1.0, 1.15, 1.3])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--ledger", type=Path, default=ROOT / "experiments/latent-scaling-ledger.jsonl")
    args = parser.parse_args()
    try:
        for sigma in args.sigmas:
            validate_sampling(args.num_samples, args.batch_size, sigma)
        if args.num_samples < 1 or args.threads < 1:
            raise ValueError("num-samples and threads must be positive")
        if 1.0 not in args.sigmas or len(set(args.sigmas)) != len(args.sigmas):
            raise ValueError("sigmas must be unique and include the sigma=1 reference")
        if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**63 for seed in args.seeds):
            raise ValueError("seeds must be unique integers in [0, 2**63)")
    except ValueError as error:
        parser.error(str(error))
    checkpoint = args.checkpoint.resolve()
    if checkpoint.is_dir():
        checkpoint = checkpoint / "G.pth"
    if args.ledger.resolve() == checkpoint:
        parser.error("ledger must not point to the generator checkpoint")
    checkpoint_hash = sha256(checkpoint)
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    with record_run(run / "summary.json") as run_state:
        device = default_device()
        torch.set_num_threads(args.threads)
        configure_determinism()
        source_hashes = {}
        for name in ["sampling.py", "generate.py", "model.py", "experiment_utils.py", "scripts/latent_scaling.py", "requirements.txt"]:
            target = run / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
            source_hashes[name] = sha256(target)
        packages = {name: importlib.metadata.version(name) for name in
                    ["torch", "torchvision", "numpy", "pillow", "scipy", "scikit-learn", "tqdm"]}
        cpu = platform.processor()
        if Path("/proc/cpuinfo").exists():
            cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                        if line.startswith("model name")), cpu)
        summary = {
            "run_id": run.name, "method": "I1-latent-scaling", "status": "running",
            "started_at_utc": run_state["started_at_utc"],
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip(),
            "source_sha256": source_hashes, "checkpoint": relative(checkpoint),
            "checkpoint_sha256": checkpoint_hash, "packages": packages, "python": sys.version,
            "hardware": {"device": str(device), "cpu": cpu, "platform": platform.platform(),
                         "logical_cpus": os.cpu_count(), "torch_threads": args.threads,
                         "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
            "config": {"sigmas": args.sigmas, "seeds": args.seeds, "num_samples": args.num_samples,
                       "batch_size": args.batch_size, "deterministic_algorithms": True,
                       "prior": "z = sigma * eps; eps ~ N(0,I), dimension 100",
                       "rng": "private CPU RNG seeded before generator initialization; full-batch draws",
                       "export": "starter PNG RGB 28x28; negative raw values clipped, no rescaling",
                       "training": "none; same frozen G for every condition"},
            "timing_protocol": "Per-condition load + weight snapshot + noise hashing/raw diagnostics + inference + all PNG writes; "
                               "excludes Python imports, source recording, post-export validation and grids. Same protocol for sigma=1. "
                               "Not directly comparable to subprocess-based baseline timings.",
            "official_metrics": {"FID": None, "precision": None, "recall": None,
                                 "status": "not yet evaluated: official evaluator unavailable"},
            "results": [],
        }
        write_json(run / "summary.json", summary)
        total_start = time.perf_counter()
        for seed in args.seeds:
            reference_noise_hash = None
            comparison = Image.new("RGB", (482, 54 * len(args.sigmas)), "#202020")
            draw = ImageDraw.Draw(comparison)
            for row, sigma in enumerate(args.sigmas):
                condition = run / f"seed-{seed}" / f"sigma-{sigma_tag(sigma)}"
                condition.mkdir(parents=True, exist_ok=False)
                started = time.perf_counter()
                model, noise_rng = load_seeded_generator(checkpoint, seed, device)
                before = {key: value.detach().clone() for key, value in model.state_dict().items()}
                noise_hash = hashlib.sha256()
                raw_sum = negative = pixels = 0
                def recorded_batches():
                    nonlocal raw_sum, negative, pixels
                    for batch in iter_candidates(model, args.num_samples, args.batch_size, sigma, noise_rng):
                        noise_hash.update(batch.eps.numpy().tobytes())
                        raw_sum += batch.images.cpu().double().sum().item()
                        negative += (batch.images < 0).sum().item()
                        pixels += batch.images.numel()
                        yield batch
                count = export_pngs(recorded_batches(), condition / "samples")
                generation_seconds = time.perf_counter() - started
                if any(not torch.equal(value, before[key]) for key, value in model.state_dict().items()):
                    raise RuntimeError("Inference changed generator weights")
                digest = noise_hash.hexdigest()
                if reference_noise_hash is None:
                    reference_noise_hash = digest
                if digest != reference_noise_hash:
                    raise RuntimeError("Base noise differs between sigma conditions")

                samples = condition / "samples"
                if count != args.num_samples or {p.name for p in samples.iterdir()} != {
                        f"{i}.png" for i in range(args.num_samples)}:
                    raise RuntimeError("Incorrect sample count or filenames")
                grid = []
                for i in range(args.num_samples):
                    with Image.open(samples / f"{i}.png") as image:
                        image.load()
                        if image.format != "PNG" or image.mode != "RGB" or image.size != (28, 28):
                            raise RuntimeError(f"Invalid image: {i}.png")
                        if i < 64:
                            grid.append(to_tensor(image))
                        if i < 16:
                            comparison.paste(image, (i * 30, row * 54 + 22))
                save_image(torch.stack(grid), condition / "sample-grid.png", nrow=8)
                draw.text((2, row * 54 + 3), f"seed={seed}, sigma={sigma:g}; same latent indices 0-15", fill="white")
                if sha256(checkpoint) != checkpoint_hash or sha256(ROOT / "model.py") != source_hashes["model.py"]:
                    raise RuntimeError("Checkpoint or generator source changed")
                record = {
                    "run_id": f"{run.name}-seed{seed}-sigma{sigma_tag(sigma)}", "method": summary["method"],
                    "seed": seed, "sigma": sigma, "commit": summary["commit"],
                    "checkpoint": relative(checkpoint), "checkpoint_sha256": checkpoint_hash,
                    "config": {**summary["config"], "sigma": sigma, "seed": seed},
                    "base_noise_sha256": digest, "candidate_index_range": [0, args.num_samples - 1],
                    "training_seconds": 0, "generation_seconds": generation_seconds,
                    "accepted_images": count, "acceptance_rate": 1.0,
                    "candidates_generated": ((count + args.batch_size - 1) // args.batch_size) * args.batch_size,
                    "metadata": relative(condition / "metadata.json"),
                    "official_metrics": summary["official_metrics"],
                    "diagnostics": {"raw_mean": raw_sum / pixels, "negative_pixel_fraction": negative / pixels},
                    "validation": {"all_png_rgb_28x28": True, "exact_count": True,
                                   "paired_noise": True, "generator_weights_unchanged": True,
                                   "checkpoint_unchanged": True},
                }
                write_json(condition / "metadata.json", record)
                args.ledger.parent.mkdir(parents=True, exist_ok=True)
                with args.ledger.open("a") as ledger:
                    ledger.write(json.dumps(record, sort_keys=True) + "\n")
                summary["results"].append(record)
                write_json(run / "summary.json", summary)
                print(f"seed={seed} sigma={sigma:g}: {count} images in {generation_seconds:.3f}s; pairing OK", flush=True)
            comparison.save(run / f"comparison-seed{seed}.png")
        summary["status"] = "completed"
        summary["total_seconds"] = time.perf_counter() - total_start
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(run / "summary.json", summary)


if __name__ == "__main__":
    main()
