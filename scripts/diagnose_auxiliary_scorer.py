"""Test a frozen calibrated auxiliary scorer on fresh top-k/random candidate pools."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auxiliary_scorer import load_calibrated_scorer, score_candidates
from discriminator_diagnostics import score_summary, select_indices
from experiment_utils import configure_determinism, record_run, write_json
from sampling import generate_candidates
from scripts.diagnose_discriminator import comparison_grid, export_selection
from scripts.evaluate_local import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scorer-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/baseline-seed0-e10/G.pth")
    parser.add_argument("--evaluator", type=Path, default=ROOT / "experiments/local-metrics-mnist64-k3")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12])
    parser.add_argument("--pool-size", type=int, default=20000)
    parser.add_argument("--sigma", type=float, default=1.)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if (args.pool_size <= 10000 or min(args.batch_size, args.threads) < 1
            or len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**63 for seed in args.seeds)):
        parser.error("Require pool-size >10000, positive batch-size/threads, and unique valid seeds")
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    with record_run(run / "summary.json") as initial:
        start = time.perf_counter()
        torch.set_num_threads(args.threads)
        configure_determinism()
        model, metadata = load_calibrated_scorer(args.scorer_dir, args.checkpoint, args.sigma)
        if set(args.seeds).intersection(metadata["fake_seeds"].values()):
            raise ValueError("Diagnostic pools must use seeds separate from scorer fitting/validation/calibration")
        before = {key: value.clone() for key, value in model.state_dict().items()}
        sources = ["auxiliary_scorer.py", "discriminator_diagnostics.py", "sampling.py", "model.py",
                   "experiment_utils.py", "local_metrics.py", "scripts/diagnose_auxiliary_scorer.py",
                   "scripts/diagnose_discriminator.py", "scripts/evaluate_local.py", "scripts/evaluate_frozen.py"]
        hashes = {}
        for name in sources:
            destination = run / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
            hashes[name] = digest(destination)
        summary = {**initial, "run_id": run.name, "method": "I2-auxiliary-score-diagnostic", "is_DRS": False,
                   "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "source_sha256": hashes, "scorer_directory": str(args.scorer_dir.resolve()),
                   "scorer_metadata": metadata, "generator_checkpoint": str(args.checkpoint.resolve()),
                   "hardware": {"device": "cpu", "threads": args.threads},
                   "config": {"seeds": args.seeds, "pool_size": args.pool_size, "selected_count": 10000,
                              "sigma": args.sigma, "batch_size": args.batch_size,
                              "selection_methods": ["first", "random", "top"],
                              "score": "calibrated auxiliary logits after exact starter PNG mapping",
                              "selection_seed": "generation_seed XOR 0x5EED",
                              "ties": "earlier candidate index first", "output_order": "candidate order",
                              "timing": "generation_seconds repeats shared pool generation/scoring plus own selection/export; excludes global setup/evaluation"},
                   "official_metrics": {"FID": None, "precision": None, "recall": None},
                   "conditions": [], "pools": []}
        write_json(run / "summary.json", summary)
        for seed in args.seeds:
            pool_start = time.perf_counter()
            images, noise = generate_candidates(args.pool_size, seed, args.checkpoint, sigma=args.sigma,
                                                batch_size=args.batch_size, device="cpu", return_noise=True)
            generation_seconds = time.perf_counter() - pool_start
            noise_hash = hashlib.sha256(noise.numpy().tobytes()).hexdigest()
            del noise
            scoring_start = time.perf_counter()
            scores = score_candidates(model, images, args.batch_size, metadata["calibration"])
            scoring_seconds = time.perf_counter() - scoring_start
            index_arrays = {"calibrated_logits": scores.numpy()}
            summary["pools"].append({"seed": seed, "base_noise_sha256": noise_hash,
                                      "generation_seconds": generation_seconds, "scoring_seconds": scoring_seconds,
                                      "forward_candidates": ((args.pool_size + args.batch_size - 1) // args.batch_size) * args.batch_size,
                                      "scores": score_summary(scores)})
            for method in summary["config"]["selection_methods"]:
                selection_start = time.perf_counter()
                indices = select_indices(scores, 10000, method, seed ^ 0x5EED)
                samples = run / f"seed-{seed}" / method / "samples"
                count = export_selection(images, indices, samples, args.batch_size)
                if count != 10000 or len(indices.unique()) != count:
                    raise RuntimeError("Selection must yield exactly 10,000 unique candidate IDs")
                index_arrays[f"indices_{method}"] = indices.numpy()
                condition = {
                    "run_id": f"{run.name}-seed{seed}-{method}", "method": summary["method"],
                    "seed": seed, "selection": method, "sigma": args.sigma, "is_DRS": False,
                    "config": {**summary["config"], "seed": seed, "selection": method},
                    "samples": str(samples), "checkpoint": str(args.checkpoint.resolve()),
                    "checkpoint_sha256": metadata["generator_sha256"], "scorer_sha256": metadata["scorer_sha256"],
                    "candidate_pool": args.pool_size, "accepted_images": count,
                    "selection_fraction": count / args.pool_size, "acceptance_rate": None,
                    "selected_scores": score_summary(scores[indices]),
                    "generation_seconds": generation_seconds + scoring_seconds + time.perf_counter() - selection_start,
                    "official_metrics": summary["official_metrics"],
                }
                summary["conditions"].append(condition)
                write_json(run / "summary.json", summary)
                print(f"seed={seed} {method}: exported {count}; mean calibrated logit={scores[indices].mean():.4f}", flush=True)
            np.savez_compressed(run / f"seed-{seed}/selection-indices-and-scores.npz", **index_arrays)
        write_json(run / "evaluation-inputs.json", {"conditions": summary["conditions"]})
        comparison_grid(summary["conditions"], run / "comparison.png")
        subprocess.run([sys.executable, "-B", str(ROOT / "scripts/evaluate_frozen.py"),
                        "--manifest", str(run / "evaluation-inputs.json"), "--run-dir", str(run / "local-metrics"),
                        "--evaluator", str(args.evaluator.resolve()), "--data-path", str(args.data_path.resolve()),
                        "--threads", str(args.threads)], check=True)
        summary["evaluation"] = json.loads((run / "local-metrics/summary.json").read_text())
        if (any(not torch.equal(value, before[key]) for key, value in model.state_dict().items())
                or digest(args.checkpoint) != metadata["generator_sha256"]
                or digest(args.scorer_dir / "scorer.pth") != metadata["scorer_sha256"]
                or any(digest(ROOT / name) != expected for name, expected in hashes.items())):
            raise RuntimeError("Scorer/G weights or source changed during diagnostics")
        summary["validation"] = {"generator_and_scorer_files_unchanged": True,
                                 "scorer_parameters_unchanged": True, "source_unchanged": True,
                                 "exact_10000_pngs_per_condition": True, "fresh_diagnostic_seeds": True}
        summary["status"] = "completed"
        summary["total_seconds"] = time.perf_counter() - start
        write_json(run / "summary.json", summary)
        print(f"Completed auxiliary selection diagnostic: {run}", flush=True)


if __name__ == "__main__":
    main()
