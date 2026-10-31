"""Reproduce the starter GAN in an isolated, recorded run directory."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_utils import configure_determinism, record_run, write_json
from model import Discriminator, Generator
from utils import D_train, G_train, save_models


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generation-seed", type=int, default=1)
    parser.add_argument("--generation-batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.generation_batch_size, args.threads) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs, batch sizes, threads and learning rate must be positive")
    if any(not 0 <= seed < 2**32 for seed in [args.seed, args.generation_seed]):
        parser.error("training and generation seeds must be in [0, 2**32) for NumPy compatibility")
    configure_determinism()
    exporter_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if args.device != "auto" and args.device != exporter_device:
        parser.error(f"The unmodified starter exporter selects {exporter_device}; use --device auto")
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    with record_run(run / "metadata.json") as run_state:
        checkpoints = run / "checkpoints"
        checkpoints.mkdir()
        start = time.perf_counter()
        device_name = args.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        device = torch.device(device_name)
        torch.set_num_threads(args.threads)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        source_paths = [".gitignore", "README.md", "requirements.txt", "model.py", "train.py",
                        "generate.py", "sampling.py", "experiment_utils.py", "utils.py", "scripts/baseline.py", "scripts/train.sh", "scripts/generate.sh"]
        source_hashes = {}
        for relative in source_paths:
            source = ROOT / relative
            target = run / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            source_hashes[relative] = sha256(source)
        (run / "source" / "working-tree.patch").write_text(command("git", "diff", "HEAD"))
        (run / "environment.txt").write_text(command(sys.executable, "-m", "pip", "--no-cache-dir", "freeze") + "\n")
        (run / "torch-config.txt").write_text(torch.__config__.show())
        cpu_model = platform.processor()
        if Path("/proc/cpuinfo").exists():
            cpu_model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                              if line.startswith("model name")), cpu_model)
        metadata = {
            "run_id": run.name, "method": "B0-starter", "status": "running",
            "started_at_utc": run_state["started_at_utc"],
            "commit": command("git", "rev-parse", "HEAD"),
            "git_status": command("git", "status", "--short"),
            "source_sha256": source_hashes,
            "original_checkpoint_sha256": sha256(ROOT / "checkpoints/G.pth"),
            "invocation": sys.argv,
            "hardware": {"platform": platform.platform(), "cpu": cpu_model, "logical_cpus": os.cpu_count(),
                         "device": str(device), "torch_threads": args.threads,
                         "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
            "python": sys.version,
            "config": {"epochs": args.epochs, "batch_size": args.batch_size, "lr_G": args.lr, "lr_D": args.lr,
                       "optimizer": "Adam", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0,
                       "seed": args.seed, "generation_seed": args.generation_seed,
                       "generation_batch_size": args.generation_batch_size, "num_workers": 0,
                       "pin_memory": False, "drop_last": False, "data_path": str(args.data_path.resolve()),
                       "prior": "N(0,I), latent dimension 100", "latent_scale": 1.0,
                       "loss_D": "BCE(D(real),1) + BCE(D(G(z)),0)", "loss_G": "BCE(D(G(z)),1)",
                       "preprocessing": "ToTensor; Normalize(mean=0.5, std=0.5); flatten to 784",
                       "export": "starter-compatible generate.py: save_image without rescaling; negative values clipped",
                       "generation_rng": "seed before runpy executes generate.py, including model initialization"},
            "determinism": {"algorithms": True, "cudnn_benchmark": False, "cudnn_deterministic": True,
                            "cross_hardware_bitwise_reproducibility": False},
            "generator": {"layers": [100, 256, 512, 1024, 784], "hidden_activation": "LeakyReLU(0.2)",
                          "output_activation": "tanh", "parameters": 1486352},
            "discriminator": {"layers": [784, 1024, 512, 256, 1], "hidden_activation": "LeakyReLU(0.2)",
                              "output_activation": "sigmoid", "equal_real_fake_priors": True,
                              "calibration": "not evaluated"},
            "protocol_differences_from_train_py": [
                "Explicit seeds and deterministic settings; CPU thread count recorded.",
                "DataLoader num_workers=0 and pin_memory=False for this local run.",
                "Unused test loader omitted; all training examples retained.",
                "Save final G and D at any epoch count in an isolated directory; record losses and metadata."],
            "official_metrics": {"status": "not yet evaluated: official evaluator unavailable",
                                 "FID": None, "precision": None, "recall": None},
        }
        write_json(run / "metadata.json", metadata)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_start = time.perf_counter()
        dataset = datasets.MNIST(root=str(args.data_path.resolve()), train=True, transform=transform, download=True)
        raw = Path(dataset.raw_folder)
        metadata["dataset"] = {"name": "MNIST", "split": "train", "examples": len(dataset),
                               "raw_sha256": {p.name: sha256(p) for p in sorted(raw.iterdir()) if p.is_file()}}
        metadata["dataset_setup_seconds"] = time.perf_counter() - dataset_start
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=0, pin_memory=False)
        G, D = Generator(784).to(device), Discriminator(784).to(device)
        initial = {k: v.detach().clone() for k, v in G.state_dict().items()}
        assert sum(p.numel() for p in G.parameters()) == 1486352
        criterion = torch.nn.BCELoss()
        g_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr)
        d_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr)
        steps = 0
        def synchronize():
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
        synchronize()
        training_start = time.perf_counter()
        with (run / "training.jsonl").open("w") as log:
            for epoch in range(1, args.epochs + 1):
                epoch_start = time.perf_counter()
                d_total = g_total = examples = 0
                for x, _ in loader:
                    x = x.view(-1, 784).to(device)
                    d_loss = D_train(x, G, D, d_optimizer, criterion, device)
                    g_loss = G_train(x, G, D, g_optimizer, criterion, device)
                    if not np.isfinite([d_loss, g_loss]).all():
                        raise RuntimeError("Non-finite training loss")
                    d_total += d_loss * len(x)
                    g_total += g_loss * len(x)
                    examples += len(x)
                    steps += 1
                synchronize()
                record = {"epoch": epoch, "steps_total": steps, "examples": examples,
                          "loss_D": d_total / examples, "loss_G": g_total / examples,
                          "seconds": time.perf_counter() - epoch_start}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
        metadata["training_seconds"] = time.perf_counter() - training_start
        metadata["discriminator_steps"] = metadata["generator_steps"] = steps
        assert any(not torch.equal(v, initial[k]) for k, v in G.state_dict().items()), "G was not trained"
        save_models(G, D, str(checkpoints))
        checkpoint_hashes = {p.name: sha256(p) for p in checkpoints.iterdir()}
        metadata["checkpoint_sha256"] = checkpoint_hashes
        write_json(checkpoints / "metadata.json", metadata)

        # Execute the starter-compatible exporter in the run directory, with a seeded RNG.
        generation_code = (
            "import random, runpy, sys, numpy as np, torch; "
            f"sys.path.insert(0, {str(ROOT)!r}); "
            f"random.seed({args.generation_seed}); np.random.seed({args.generation_seed}); "
            f"torch.manual_seed({args.generation_seed}); torch.set_num_threads({args.threads}); "
            "torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False; "
            "torch.backends.cudnn.deterministic=True; "
            f"sys.argv=['generate.py', '--batch_size', '{args.generation_batch_size}']; "
            f"runpy.run_path({str(ROOT / 'generate.py')!r}, run_name='__main__')"
        )
        # generate.py selects its own device; forbid a misleading device record.
        generation_start = time.perf_counter()
        with (run / "generation.log").open("w") as log:
            subprocess.run([sys.executable, "-B", "-c", generation_code], cwd=run,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        metadata["generation_seconds"] = time.perf_counter() - generation_start
        metadata["generation"] = {"output_images": 10000, "rejection_acceptance_rate": 1.0,
                                  "candidate_images": ((10000 + args.generation_batch_size - 1)
                                                       // args.generation_batch_size) * args.generation_batch_size,
                                  "batches": (10000 + args.generation_batch_size - 1) // args.generation_batch_size}
        samples = run / "samples"
        assert {p.name for p in samples.iterdir()} == {f"{i}.png" for i in range(10000)}
        grid = []
        for i in range(10000):
            with Image.open(samples / f"{i}.png") as image:
                image.load()
                assert image.size == (28, 28) and image.mode == "RGB" and image.format == "PNG"
                if i < 64:
                    grid.append(transforms.ToTensor()(image))
        save_image(torch.stack(grid), run / "sample-grid.png", nrow=8)
        assert all(sha256(checkpoints / name) == digest for name, digest in checkpoint_hashes.items())
        assert sha256(ROOT / "checkpoints/G.pth") == metadata["original_checkpoint_sha256"]
        assert sha256(ROOT / "model.py") == source_hashes["model.py"]
        metadata["validation"] = {"sample_count": 10000, "all_png_rgb_28x28": True,
                                  "generator_weights_trained": True, "checkpoint_files_unchanged_after_generation": True,
                                  "original_checkpoint_preserved": True, "generator_source_unchanged": True}
        metadata["status"] = "completed"
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["total_seconds"] = time.perf_counter() - start
        write_json(run / "metadata.json", metadata)
        write_json(checkpoints / "metadata.json", metadata)
        ledger = ROOT / "experiments/ledger.jsonl"
        ledger.parent.mkdir(exist_ok=True)
        with ledger.open("a") as file:
            file.write(json.dumps({"run_id": run.name, "method": metadata["method"], "seed": args.seed,
                                   "commit": metadata["commit"], "config": metadata["config"],
                                   "checkpoint": os.path.relpath(checkpoints / "G.pth", ROOT),
                                   "metadata": os.path.relpath(run / "metadata.json", ROOT),
                                   "training_seconds": metadata["training_seconds"],
                                   "generation_seconds": metadata["generation_seconds"],
                                   "accepted_images": 10000, "acceptance_rate": 1.0,
                                   "official_metrics": metadata["official_metrics"]}, sort_keys=True) + "\n")
        print(f"Completed baseline: {run}", flush=True)


if __name__ == "__main__":
    main()
