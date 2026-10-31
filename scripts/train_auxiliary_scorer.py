"""Fit and independently calibrate a logistic scorer for one frozen G/sigma."""

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auxiliary_scorer import (AuxiliaryScorer, PREPROCESSING, calibrated_logits,
                              fit_sigmoid_calibration, real_splits, score_normalized)
from discriminator_diagnostics import calibration_diagnostics, starter_export_normalized
from experiment_utils import configure_determinism, record_run, write_json
from sampling import generate_candidates
from scripts.evaluate_local import digest

FAKE_SEEDS = {"train": 10001, "validation": 10002, "calibration": 10003, "diagnostic": 10004}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/baseline-seed0-e10/G.pth")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data")
    parser.add_argument("--evaluator", type=Path, default=ROOT / "experiments/local-metrics-mnist64-k3")
    parser.add_argument("--sigma", type=float, default=1.)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--split-seed", type=int, default=3028)
    parser.add_argument("--lr", type=float, default=.0002)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--generation-batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if (min(args.epochs, args.batch_size, args.generation_batch_size, args.threads) < 1
            or not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.sigma) or args.sigma <= 0
            or any(not 0 <= seed < 2**63 for seed in [args.seed, args.split_seed])):
        parser.error("Require positive finite sigma/lr, positive counts and valid seeds")
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    with record_run(run / "summary.json") as initial:
        start = time.perf_counter()
        torch.set_num_threads(args.threads)
        configure_determinism()
        checkpoint = args.checkpoint.resolve()
        generator_hash = digest(checkpoint)
        protocol = json.loads((args.evaluator / "summary.json").read_text())
        split_path = args.evaluator / "split-indices.npz"
        if digest(split_path) != protocol["data"]["split_sha256"]:
            raise ValueError("Frozen reference split hash mismatch")
        with np.load(split_path) as saved:
            excluded = torch.from_numpy(np.concatenate([saved["real_reference"], saved["real_control"]]))
        splits = real_splits(excluded, seed=args.split_seed)
        np.savez(run / "split-indices.npz", **{k: v.numpy() for k, v in splits.items()},
                 excluded_evaluator_indices=excluded.numpy())
        dataset = MNIST(str(args.data_path), train=True, download=False)
        if len(dataset) != 60000:
            raise ValueError("Expected the 60,000-image MNIST train split")
        for name, expected in protocol["data"]["raw_sha256"].items():
            if digest(Path(dataset.raw_folder) / name) != expected:
                raise ValueError(f"MNIST hash mismatch: {name}")
        source_hashes = {}
        for name in ["auxiliary_scorer.py", "discriminator_diagnostics.py", "sampling.py", "model.py",
                     "experiment_utils.py", "scripts/train_auxiliary_scorer.py", "requirements.txt"]:
            destination = run / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
            source_hashes[name] = digest(destination)
        summary = {
            **initial, "run_id": run.name, "method": "I2-auxiliary-logistic-scorer", "is_DRS": False,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip(),
            "source_sha256": source_hashes, "generator_checkpoint": str(checkpoint),
            "generator_sha256": generator_hash, "sigma": args.sigma, "preprocessing": PREPROCESSING,
            "architecture": "784->256->128->1, LeakyReLU(0.2), raw logits; fresh initialization",
            "config": {"epochs": args.epochs, "seed": args.seed, "split_seed": args.split_seed,
                       "optimizer": "Adam", "lr": args.lr, "betas": [.9, .999], "weight_decay": 0,
                       "loss": "BCEWithLogitsLoss; equal real/fake totals", "batch_size": args.batch_size,
                       "generation_batch_size": args.generation_batch_size, "fake_seeds": FAKE_SEEDS,
                       "split_sizes_per_class": {k: len(v) for k, v in splits.items()},
                       "checkpoint_selection": "lowest balanced validation log loss; no calibration/diagnostic tuning",
                       "calibration": "positive affine logits fitted only on calibration subset",
                       "sigma": args.sigma, "generator_training": "none; frozen checkpoint"},
            "data": {"source": "MNIST train only; no test images or labels used",
                     "evaluator_split_sha256": digest(split_path),
                     "split_sha256": digest(run / "split-indices.npz"),
                     "raw_sha256": protocol["data"]["raw_sha256"],
                     "auxiliary_real_splits_disjoint": True, "evaluator_reference_control_excluded": True,
                     "GAN_saw_all_real_images": True},
            "hardware": {"device": "cpu", "threads": args.threads, "platform": platform.platform()},
            "packages": {name: importlib.metadata.version(name) for name in
                         ["torch", "torchvision", "numpy", "scipy", "scikit-learn", "pillow"]},
            "training": [], "official_metrics": {"FID": None, "precision": None, "recall": None},
        }
        write_json(run / "summary.json", summary)
        preparation_start = time.perf_counter()
        real, fake = {}, {}
        for name, indices in splits.items():
            real[name] = dataset.data[indices].unsqueeze(1).float().div(255).sub(.5).div(.5)
            raw = generate_candidates(len(indices), FAKE_SEEDS[name], checkpoint, sigma=args.sigma,
                                      batch_size=args.generation_batch_size, device="cpu")
            fake[name] = starter_export_normalized(raw)
        del dataset, raw
        summary["preparation_seconds"] = time.perf_counter() - preparation_start
        torch.manual_seed(args.seed)
        model = AuxiliaryScorer()
        initial_weights = {key: value.clone() for key, value in model.state_dict().items()}
        train_x = torch.cat([real["train"], fake["train"]])
        train_y = torch.cat([torch.ones(len(real["train"])), torch.zeros(len(fake["train"]))])
        loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed), num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_loss, selected_epoch = math.inf, None
        training_start = time.perf_counter()
        with (run / "training.jsonl").open("w") as log:
            for epoch in range(1, args.epochs + 1):
                epoch_start = time.perf_counter()
                model.train()
                loss_sum = 0.
                for images, labels in loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(images), labels)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite auxiliary training loss")
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item() * len(images)
                val_real = score_normalized(model, real["validation"], args.batch_size)
                val_fake = score_normalized(model, fake["validation"], args.batch_size)
                validation = calibration_diagnostics(val_real, val_fake)
                record = {"epoch": epoch, "training_log_loss": loss_sum / len(train_x),
                          "validation_log_loss": validation["log_loss"], "validation_auc": validation["roc_auc"],
                          "seconds": time.perf_counter() - epoch_start}
                summary["training"].append(record)
                if validation["log_loss"] < best_loss:
                    best_loss, selected_epoch = validation["log_loss"], epoch
                    torch.save(model.state_dict(), run / "scorer.pth")
                log.write(json.dumps(record) + "\n")
                log.flush()
                write_json(run / "summary.json", summary)
                print(f"Epoch {epoch}: train loss={record['training_log_loss']:.5f}, "
                      f"validation loss={record['validation_log_loss']:.5f}, AUC={record['validation_auc']:.5f}", flush=True)
        summary["training_seconds"] = time.perf_counter() - training_start
        model.load_state_dict(torch.load(run / "scorer.pth", map_location="cpu", weights_only=True))
        model.eval().requires_grad_(False)
        if all(torch.equal(value, initial_weights[key]) for key, value in model.state_dict().items()):
            raise RuntimeError("Auxiliary scorer weights did not train")
        scores = {}
        for name in ["calibration", "diagnostic"]:
            scores[f"real_{name}"] = score_normalized(model, real[name], args.batch_size)
            scores[f"fake_{name}"] = score_normalized(model, fake[name], args.batch_size)
        calibration = fit_sigmoid_calibration(scores["real_calibration"].numpy(), scores["fake_calibration"].numpy())
        diagnostic = {
            "uncalibrated": calibration_diagnostics(scores["real_diagnostic"], scores["fake_diagnostic"]),
            "calibrated": calibration_diagnostics(calibrated_logits(scores["real_diagnostic"], calibration),
                                                   calibrated_logits(scores["fake_diagnostic"], calibration)),
        }
        np.savez_compressed(run / "held-out-logits.npz", **{k: v.numpy() for k, v in scores.items()})
        summary.update(selected_epoch=selected_epoch, validation_log_loss=best_loss,
                       scorer_sha256=digest(run / "scorer.pth"), calibration=calibration,
                       diagnostic=diagnostic, parameters=sum(p.numel() for p in model.parameters()))
        if digest(checkpoint) != generator_hash or any(digest(ROOT / name) != expected for name, expected in source_hashes.items()):
            raise RuntimeError("Generator checkpoint or source changed during scorer training")
        metadata = {"status": "completed", "architecture": summary["architecture"],
                    "sigma": args.sigma, "generator_sha256": generator_hash,
                    "preprocessing": PREPROCESSING, "scorer_sha256": summary["scorer_sha256"],
                    "split_sha256": summary["data"]["split_sha256"],
                    "calibration": calibration, "selected_epoch": selected_epoch,
                    "training_seed": args.seed, "fake_seeds": FAKE_SEEDS,
                    "generation_batch_size": args.generation_batch_size, "source_sha256": source_hashes}
        write_json(run / "scorer-metadata.json", metadata)
        summary["validation"] = {"generator_checkpoint_unchanged": True, "source_unchanged": True,
                                 "real_splits_disjoint": True, "reference_and_control_excluded": True,
                                 "fake_seeds_disjoint": True, "calibration_diagnostic_not_used_for_epoch_selection": True}
        summary["status"] = "completed"
        summary["total_seconds"] = time.perf_counter() - start
        write_json(run / "summary.json", summary)
        print(f"Diagnostic before/after calibration: "
              f"{ {key: {m: diagnostic[key][m] for m in ['roc_auc','brier','log_loss','ece']} for key in diagnostic} }", flush=True)
        print(f"Saved auxiliary scorer: {run}", flush=True)


if __name__ == "__main__":
    main()
