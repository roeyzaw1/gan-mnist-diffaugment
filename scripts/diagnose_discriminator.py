"""Compare frozen baseline-D top-k with random selection at sigma=1; not DRS."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import spearmanr
import torch
from torchvision.datasets import MNIST

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from discriminator_diagnostics import (calibration_diagnostics, discriminator_logits,
                                       load_discriminator, score_summary, select_indices,
                                       starter_export_normalized)
from experiment_utils import configure_determinism, record_run, write_json
from sampling import CandidateBatch, export_pngs, iter_candidates, load_seeded_generator
from scripts.evaluate_local import digest


def unchanged(model, before):
    return all(torch.equal(value.cpu(), before[key]) for key, value in model.state_dict().items())


def candidate_pool(checkpoint, seed, count, batch_size):
    model, rng = load_seeded_generator(checkpoint, seed, torch.device("cpu"))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    images, noise_hash = [], hashlib.sha256()
    for batch in iter_candidates(model, count, batch_size, sigma=1., noise_generator=rng):
        images.append(batch.images)
        noise_hash.update(batch.eps.numpy().tobytes())
    if not unchanged(model, before):
        raise RuntimeError("Generation changed G weights")
    return torch.cat(images), noise_hash.hexdigest()


def export_selection(images, indices, directory, batch_size):
    def batches():
        for start in range(0, len(indices), batch_size):
            yield CandidateBatch(start, images[indices[start:start + batch_size]], torch.empty(0, 100))
    return export_pngs(batches(), directory)


def comparison_grid(conditions, output):
    # First 16 selected candidate IDs, in original candidate order, in every row.
    canvas = Image.new("RGB", (1148, 82 * len(conditions)), "#202020")
    draw = ImageDraw.Draw(canvas)
    for row, condition in enumerate(conditions):
        draw.text((8, row * 82 + 14), f"seed={condition['seed']}, {condition['selection']}", fill="white")
        for i in range(16):
            with Image.open(Path(condition["samples"]) / f"{i}.png") as image:
                canvas.paste(image.resize((56, 56), Image.Resampling.NEAREST), (220 + i * 58, row * 82 + 12))
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints/baseline-seed0-e10")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data")
    parser.add_argument("--evaluator", type=Path, default=ROOT / "experiments/local-metrics-mnist64-k3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--calibration-seed", type=int, default=91)
    parser.add_argument("--pool-size", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.pool_size <= 10000 or min(args.batch_size, args.threads) < 1:
        parser.error("pool-size must exceed 10000; batch-size and threads must be positive")
    if (len(set(args.seeds)) != len(args.seeds) or args.calibration_seed in args.seeds
            or any(not 0 <= seed < 2**63 for seed in [*args.seeds, args.calibration_seed])):
        parser.error("Use unique valid seeds in [0,2**63); calibration seed must be separate")
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    with record_run(run / "summary.json") as initial:
        start = time.perf_counter()
        torch.set_num_threads(args.threads)
        configure_determinism()
        g_path, d_path = args.checkpoint_dir / "G.pth", args.checkpoint_dir / "D.pth"
        g_hash, d_hash = digest(g_path), digest(d_path)
        training = json.loads((args.checkpoint_dir / "metadata.json").read_text())
        if (training["checkpoint_sha256"]["G.pth"] != g_hash
                or training["checkpoint_sha256"]["D.pth"] != d_hash):
            raise ValueError("G/D do not match the checkpoint-pair metadata")
        if (not training["discriminator"]["equal_real_fake_priors"]
                or training["config"]["loss_D"] != "BCE(D(real),1) + BCE(D(G(z)),0)"):
            raise ValueError("This diagnostic expects the original equal-prior BCE discriminator")
        protocol = json.loads((args.evaluator / "summary.json").read_text())
        if digest(args.evaluator / "split-indices.npz") != protocol["data"]["split_sha256"]:
            raise ValueError("Frozen evaluator split mismatch")
        dataset = MNIST(str(args.data_path), train=True, download=False)
        for name, expected in protocol["data"]["raw_sha256"].items():
            if digest(Path(dataset.raw_folder) / name) != expected:
                raise ValueError(f"MNIST hash mismatch: {name}")
        with np.load(args.evaluator / "split-indices.npz") as split:
            real_indices = torch.from_numpy(split["real_control"].copy())
        real_images = dataset.data[real_indices].unsqueeze(1).float().div(255).sub(.5).div(.5)
        if len(real_images) != 10000:
            raise ValueError("Expected 10,000 real development-control images")
        discriminator = load_discriminator(d_path)
        d_before = {key: value.clone() for key, value in discriminator.state_dict().items()}
        sources = ["model.py", "sampling.py", "discriminator_diagnostics.py", "experiment_utils.py",
                   "local_metrics.py", "scripts/diagnose_discriminator.py", "scripts/evaluate_local.py",
                   "scripts/evaluate_frozen.py", "requirements.txt"]
        source_hashes = {}
        for name in sources:
            destination = run / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
            source_hashes[name] = digest(destination)
        summary = {
            **initial, "run_id": run.name, "method": "I2-D-score-diagnostic", "is_DRS": False,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip(),
            "source_sha256": source_hashes,
            "generator_checkpoint": str(g_path.resolve()), "generator_sha256": g_hash,
            "discriminator_checkpoint": str(d_path.resolve()), "discriminator_sha256": d_hash,
            "evaluator_protocol": protocol["protocol"],
            "extractor_sha256": protocol["extractor"]["checkpoint_sha256"],
            "real_split_sha256": protocol["data"]["split_sha256"],
            "hardware": {"device": "cpu", "platform": platform.platform(), "torch_threads": args.threads},
            "packages": {name: importlib.metadata.version(name) for name in
                         ["torch", "torchvision", "numpy", "scipy", "scikit-learn", "pillow"]},
            "config": {"seeds": args.seeds, "calibration_seed": args.calibration_seed,
                       "pool_size": args.pool_size, "selected_count": 10000, "batch_size": args.batch_size,
                       "sigma": 1., "selection_methods": ["first", "random", "top"],
                       "random_selection_seed": "generation_seed XOR 0x5EED",
                       "score": "original final linear pre-sigmoid logit; raw tanh images in [-1,1]",
                       "top_ties": "earlier candidate index first", "output_order": "ascending candidate index",
                       "calibration": "assessment only; balanced real_control vs independent generated seed",
                       "real_data_limit": "MNIST train only; held out from feature extractor, seen by GAN",
                       "export": "starter save_image, negative values clipped, no rescaling",
                       "timing": "per-condition generation_seconds = shared pool generation + raw scoring + selection/export; "
                                 "excludes global setup, export-shift diagnostics and evaluation; shared costs repeated per condition"},
            "official_metrics": {"FID": None, "precision": None, "recall": None, "status": "official evaluator unavailable"},
            "conditions": [], "pools": [],
        }
        write_json(run / "summary.json", summary)
        calibration_start = time.perf_counter()
        fake_images, calibration_noise_hash = candidate_pool(g_path, args.calibration_seed, 10000, args.batch_size)
        real_scores = discriminator_logits(discriminator, real_images, args.batch_size)
        fake_scores = discriminator_logits(discriminator, fake_images, args.batch_size)
        png_scores = discriminator_logits(discriminator, starter_export_normalized(fake_images), args.batch_size)
        summary["calibration"] = {
            "native_raw": calibration_diagnostics(real_scores, fake_scores),
            "exported_png_mapping": calibration_diagnostics(real_scores, png_scores),
            "raw_vs_exported_spearman": float(spearmanr(fake_scores.numpy(), png_scores.numpy()).statistic),
            "base_noise_sha256": calibration_noise_hash,
            "seconds": time.perf_counter() - calibration_start,
        }
        np.savez_compressed(run / "calibration-scores.npz", real_indices=real_indices.numpy(),
                            real_logits=real_scores.numpy(), raw_fake_logits=fake_scores.numpy(),
                            exported_fake_logits=png_scores.numpy())
        del fake_images, real_images
        write_json(run / "summary.json", summary)
        print(f"Native D assessment: { {key: summary['calibration']['native_raw'][key] for key in ['roc_auc','brier','ece']} }", flush=True)
        for seed in args.seeds:
            pool_start = time.perf_counter()
            images, noise_hash = candidate_pool(g_path, seed, args.pool_size, args.batch_size)
            generation_seconds = time.perf_counter() - pool_start
            score_start = time.perf_counter()
            scores = discriminator_logits(discriminator, images, args.batch_size)
            scoring_seconds = time.perf_counter() - score_start
            export_scores = discriminator_logits(discriminator, starter_export_normalized(images), args.batch_size)
            pool = {"seed": seed, "candidate_pool": args.pool_size, "base_noise_sha256": noise_hash,
                    "generator_weights_unchanged": True,
                    "forward_candidates": ((args.pool_size + args.batch_size - 1) // args.batch_size) * args.batch_size,
                    "pool_generation_seconds": generation_seconds, "pool_scoring_seconds": scoring_seconds,
                    "native_score": score_summary(scores), "exported_score": score_summary(export_scores),
                    "raw_vs_exported_spearman": float(spearmanr(scores.numpy(), export_scores.numpy()).statistic)}
            index_arrays = {"raw_logits": scores.numpy(), "exported_logits": export_scores.numpy()}
            for method in summary["config"]["selection_methods"]:
                selection_start = time.perf_counter()
                indices = select_indices(scores, 10000, method, seed ^ 0x5EED)
                samples = run / f"seed-{seed}" / method / "samples"
                count = export_selection(images, indices, samples, args.batch_size)
                if count != 10000 or len(indices.unique()) != count:
                    raise RuntimeError("Selection did not produce exactly 10,000 unique candidate IDs")
                index_arrays[f"indices_{method}"] = indices.numpy()
                condition = {
                    "run_id": f"{run.name}-seed{seed}-{method}", "method": "I2-D-score-diagnostic",
                    "seed": seed, "selection": method, "is_DRS": False,
                    "config": {**summary["config"], "seed": seed, "selection": method},
                    "samples": str(samples), "checkpoint": str(g_path.resolve()), "checkpoint_sha256": g_hash,
                    "discriminator_sha256": d_hash, "candidate_pool": args.pool_size, "accepted_images": count,
                    "selection_fraction": count / args.pool_size, "acceptance_rate": None,
                    "selected_raw_scores": score_summary(scores[indices]),
                    "generation_seconds": generation_seconds + scoring_seconds + time.perf_counter() - selection_start,
                    "official_metrics": summary["official_metrics"],
                }
                summary["conditions"].append(condition)
                write_json(run / "summary.json", summary)
                print(f"seed={seed} {method}: exported {count}; mean logit={scores[indices].mean():.4f}", flush=True)
            np.savez_compressed(run / f"seed-{seed}/selection-indices-and-scores.npz", **index_arrays)
            summary["pools"].append(pool)
            del images
        write_json(run / "evaluation-inputs.json", {"conditions": summary["conditions"]})
        comparison_grid(summary["conditions"], run / "comparison.png")
        subprocess.run([sys.executable, "-B", str(ROOT / "scripts/evaluate_frozen.py"),
                        "--manifest", str(run / "evaluation-inputs.json"), "--run-dir", str(run / "local-metrics"),
                        "--evaluator", str(args.evaluator.resolve()), "--data-path", str(args.data_path.resolve()),
                        "--threads", str(args.threads)], check=True)
        summary["evaluation"] = json.loads((run / "local-metrics/summary.json").read_text())
        if (not unchanged(discriminator, d_before) or digest(g_path) != g_hash or digest(d_path) != d_hash
                or any(digest(ROOT / name) != expected for name, expected in source_hashes.items())):
            raise RuntimeError("Weights or source changed during diagnostics")
        summary["validation"] = {"G_and_D_files_unchanged": True, "D_parameters_unchanged": True,
                                 "exact_10000_pngs_per_condition": True, "source_unchanged": True}
        summary["status"] = "completed"
        summary["total_seconds"] = time.perf_counter() - start
        write_json(run / "summary.json", summary)
        print(f"Completed diagnostic: {run}", flush=True)


if __name__ == "__main__":
    main()
