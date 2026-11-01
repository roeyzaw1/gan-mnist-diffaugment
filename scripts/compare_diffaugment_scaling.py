"""Paired latent-scale comparison on Roey's shared, frozen DiffAugment G.

No retraining or DRS. Needs the trained checkpoint from the DiffAugment study;
the default expected hash identifies its recorded seed-0/10-epoch checkpoint.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_utils import record_run, write_json
from sampling import validate_sampling

EXPECTED_CHECKPOINT = "cfefa20e6f28c863f23e19a0521a64fbaf92364341de6b724feeb5ca6bc0522d"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/diffaugment-seed0-e10/G.pth")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-sha256", default=EXPECTED_CHECKPOINT,
                        help="Recorded trained G hash; explicitly override for another documented checkpoint.")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data")
    parser.add_argument("--evaluator", type=Path, default=ROOT / "experiments/local-metrics-mnist64-k3")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--sigmas", nargs="+", type=float, default=[1., 1.5, 2.],
                        help="Unique positive finite scales, including the sigma=1 control.")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if checkpoint.is_dir():
        checkpoint = checkpoint / "G.pth"
    if args.threads < 1 or len(set(args.seeds)) != len(args.seeds) or any(not 0 <= x < 2**63 for x in args.seeds):
        parser.error("Require positive threads and unique valid generation seeds")
    try:
        for sigma in args.sigmas:
            validate_sampling(10000, 256, sigma)
        if 1. not in args.sigmas or len(set(args.sigmas)) != len(args.sigmas) or len(args.sigmas) < 2:
            raise ValueError("Require at least two distinct sigmas, including sigma=1")
    except ValueError as error:
        parser.error(str(error))
    try:
        if sha(checkpoint) != args.expected_sha256:
            parser.error("Checkpoint hash differs from the declared DiffAugment G; refusing a mislabeled experiment")
    except OSError as error:
        parser.error(f"DiffAugment checkpoint unavailable: {error}")
    metadata_path = checkpoint.parent / "metadata.json"
    try:
        training_metadata = json.loads(metadata_path.read_text())
        if (training_metadata.get("status") != "completed"
                or training_metadata["checkpoint_sha256"]["G.pth"] != args.expected_sha256
                or not training_metadata["config"].get("diffaugment_policy")):
            parser.error("Expected completed DiffAugment training metadata matching the checkpoint")
        for source in ["model.py", "utils.py", "diffaugment.py"]:
            if training_metadata["source_sha256"][source] != sha(ROOT / source):
                parser.error(f"Current {source} differs from the recorded training implementation")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Invalid training provenance: {error}")
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    with record_run(run / "summary.json") as initial:
        started = time.perf_counter()
        source_hashes = {}
        for name in ["scripts/compare_diffaugment_scaling.py", "scripts/latent_scaling.py", "scripts/evaluate_frozen.py",
                     "scripts/evaluate_local.py", "sampling.py", "model.py", "utils.py", "diffaugment.py", "local_metrics.py", "experiment_utils.py", "requirements.txt"]:
            target = run / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
            source_hashes[name] = sha(target)
        summary = {
            **initial, "run_id": run.name, "method": "C1-DiffAugment+latent-scaling",
            "checkpoint": str(checkpoint), "checkpoint_sha256": args.expected_sha256,
            "source_sha256": source_hashes,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip(),
            "config": {"sigmas": args.sigmas, "generation_seeds": args.seeds, "generation_batch_size": 256,
                       "num_samples": 10000, "torch_threads": args.threads,
                       "training": "none; fixed trained DiffAugment G", "DRS": False,
                       "export": "starter RGB PNG, negative pixels clipped, no rescaling",
                       "checkpoint_provenance": "Roey's Juliet/A100 translation+cutout seed0 e10; shared in main at 166bda6"
                                                if args.expected_sha256 == EXPECTED_CHECKPOINT else "explicit alternative checkpoint; record training provenance separately",
                       "replication_limit": "generation seeds on one trained checkpoint; not independent training runs"},
            "official_metrics": {"FID": None, "precision": None, "recall": None},
            "training_metadata": training_metadata,
        }
        summary["registered_training_seed"] = training_metadata["config"]["seed"]
        summary["registered_gpu_sigma1_result"] = ({"FD_MNIST": 95.9753, "precision": .4329, "recall": .0414,
                                                    "note": "Recorded GPU result, not a substitute for the current paired sigma-1 evaluation"}
                                                   if args.expected_sha256 == EXPECTED_CHECKPOINT else None)
        write_json(run / "summary.json", summary)
        subprocess.run([sys.executable, "-B", str(ROOT / "scripts/latent_scaling.py"),
                        "--checkpoint", str(checkpoint), "--run-dir", str(run / "sampling"),
                        "--sigmas", *map(str, args.sigmas), "--seeds", *map(str, args.seeds),
                        "--num-samples", "10000", "--batch-size", "256", "--threads", str(args.threads),
                        "--ledger", str(run / "generation-ledger.jsonl")], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-B", str(ROOT / "scripts/evaluate_frozen.py"),
                        "--sweep", str(run / "sampling"), "--run-dir", str(run / "local-metrics"),
                        "--data-path", str(args.data_path.resolve()), "--evaluator", str(args.evaluator.resolve()),
                        "--threads", str(args.threads)], cwd=ROOT, check=True)
        sweep = json.loads((run / "sampling/summary.json").read_text())
        evaluation = json.loads((run / "local-metrics/summary.json").read_text())
        if sweep["status"] != "completed" or evaluation["status"] != "completed":
            raise RuntimeError("Incomplete sampling/evaluation")
        for seed in args.seeds:
            rows = [r for r in sweep["results"] if r["seed"] == seed]
            if len(rows) != len(args.sigmas) or len({r["base_noise_sha256"] for r in rows}) != 1:
                raise RuntimeError("Sigma conditions are not paired")
        summary["evaluation_protocol"] = evaluation["protocol"]
        summary["evaluation_extractor_sha256"] = evaluation["extractor"]["checkpoint_sha256"]
        summary["hardware"] = sweep["hardware"]
        summary["packages"] = sweep["packages"]
        summary["results"] = evaluation["results"]
        summary["means"] = []
        for sigma in args.sigmas:
            rows = [r for r in evaluation["results"] if r["sigma"] == sigma]
            if len(rows) != len(args.seeds):
                raise RuntimeError("Missing evaluated condition")
            mean = {"sigma": sigma, **{k: sum(r[k] for r in rows) / len(rows)
                                        for k in ["FD_MNIST", "precision", "recall"]}}
            summary["means"].append(mean)
            print(f"sigma={sigma:g}: FD-MNIST={mean['FD_MNIST']:.4f}, "
                  f"precision={mean['precision']:.3%}, recall={mean['recall']:.3%}", flush=True)
        with (run / "comparison.csv").open("w", newline="") as file:
            fields = ["seed", "sigma", "FD_MNIST", "precision", "recall", "image_bytes_sha256"]
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(evaluation["results"])
        summary["paired_deltas_vs_sigma1"] = [
            {"seed": row["seed"], "sigma": row["sigma"], **{
                key: row[key] - next(r for r in evaluation["results"] if r["seed"] == row["seed"] and r["sigma"] == 1.)[key]
                for key in ["FD_MNIST", "precision", "recall"]}}
            for row in evaluation["results"] if row["sigma"] != 1.]
        if sha(checkpoint) != args.expected_sha256 or any(sha(ROOT / name) != h for name, h in source_hashes.items()):
            raise RuntimeError("Checkpoint or source changed during experiment")
        summary["validation"] = {"checkpoint_unchanged": True, "source_unchanged": True,
                                 "paired_noise": True, "exact_10000_per_condition": True}
        summary["status"] = "completed"
        summary["total_seconds"] = time.perf_counter() - started
        write_json(run / "summary.json", summary)


if __name__ == "__main__":
    main()
