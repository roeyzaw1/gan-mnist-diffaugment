"""Bounded, paper-inspired practical DRS trial with paired unfiltered controls."""

import argparse
import hashlib
import importlib.metadata
import json
import math
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
from drs import PracticalDRS, SamplingBudgetExceeded, rejection_batches
from experiment_utils import configure_determinism, record_run, write_json
from sampling import export_pngs, generate_candidates, iter_candidates, load_seeded_generator
from scripts.diagnose_discriminator import comparison_grid
from scripts.evaluate_local import digest


def sample_condition(args, seed, method, directory):
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    deadline = started + args.max_seconds
    stats, audit, state = {}, {}, None
    pilot_seconds = 0.
    scorer_metadata = None
    def check_deadline():
        if time.perf_counter() >= deadline:
            raise SamplingBudgetExceeded("end-to-end time budget exhausted")
    try:
        if method == "drs":
            scorer, scorer_metadata = load_calibrated_scorer(args.scorer_dir, args.checkpoint, args.sigma)
            scorer_before = {k: v.clone() for k, v in scorer.state_dict().items()}
            pilot_start = time.perf_counter()
            pilot_images, pilot_noise = generate_candidates(args.pilot_size, args.pilot_seed, args.checkpoint,
                sigma=args.sigma, batch_size=args.batch_size, device="cpu", return_noise=True)
            pilot_scores = score_candidates(scorer, pilot_images, args.batch_size, scorer_metadata["calibration"])
            state = PracticalDRS(pilot_scores, epsilon=1e-6)
            pilot_noise_hash = hashlib.sha256(pilot_noise.numpy().tobytes()).hexdigest()
            np.savez_compressed(directory / "pilot.npz", logits=pilot_scores.numpy())
            del pilot_images, pilot_noise
            pilot_seconds = time.perf_counter() - pilot_start
            check_deadline()
        model, noise = load_seeded_generator(args.checkpoint, seed, torch.device("cpu"))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        check_deadline()
        if method == "drs":
            source = iter_candidates(model, args.max_candidates, args.batch_size, args.sigma, noise)
            selected = rejection_batches(source,
                lambda images: score_candidates(scorer, images, args.batch_size, scorer_metadata["calibration"]),
                state, count=10000, max_candidates=args.max_candidates,
                max_seconds=deadline - time.perf_counter(), acceptance_seed=seed ^ 0xD25, stats=stats, audit=audit)
        else:
            def unfiltered():
                h = hashlib.sha256()
                for batch in iter_candidates(model, 10000, args.batch_size, args.sigma, noise):
                    stats["production_forward_candidates"] = stats.get("production_forward_candidates", 0) + args.batch_size
                    stats["candidates_tested"] = stats.get("candidates_tested", 0) + len(batch.images)
                    check_deadline()
                    h.update(batch.eps.numpy().tobytes())
                    if batch.start == 0:
                        stats["first_batch_noise_sha256"] = h.hexdigest()
                    yield batch
                    check_deadline()
                stats.update(candidates_tested=10000, accepted_by_rule=10000, retained_images=10000,
                             dropped_final_acceptances=0, acceptance_rate=1., retained_output_rate=1.,
                             base_noise_sha256=h.hexdigest())
            selected = unfiltered()
        count = export_pngs(selected, directory / "samples")
        check_deadline()
        seconds = time.perf_counter() - started
        if count != 10000 or any(not torch.equal(value, before[k]) for k, value in model.state_dict().items()):
            raise RuntimeError("Wrong output count or mutated G parameters")
        if method == "drs" and any(not torch.equal(value, scorer_before[k]) for k, value in scorer.state_dict().items()):
            raise RuntimeError("Scoring changed auxiliary parameters")
        status, error = "completed", None
    except SamplingBudgetExceeded as failure:
        status, error = "budget_exhausted", str(failure)
        count = len(list((directory / "samples").glob("*.png")))
        seconds = time.perf_counter() - started
    tested = stats.get("candidates_tested", 0)
    pilot_forward = math.ceil(args.pilot_size / args.batch_size) * args.batch_size if method == "drs" else 0
    production_forward = tested if method == "drs" else stats.get("production_forward_candidates", 0)
    record = {
        "run_id": f"{args.run_dir.name}-seed{seed}-{method}", "method": "I2-practical-DRS" if method == "drs" else "I2-unfiltered-control",
        "selection": method, "seed": seed, "sigma": args.sigma, "is_DRS": method == "drs", "status": status, "error": error,
        "samples": str((directory / "samples").resolve()), "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": args.generator_sha256, "scorer_sha256": args.scorer_sha256 if method == "drs" else None,
        "config": {"sigma": args.sigma, "generation_seed": seed, "batch_size": args.batch_size,
                   "pilot_seed": args.pilot_seed if method == "drs" else None, "pilot_size": args.pilot_size if method == "drs" else 0,
                   "acceptance_seed": seed ^ 0xD25 if method == "drs" else None,
                   "epsilon": 1e-6, "gamma_policy": "median pilot F, recomputed at each empirical max update" if method == "drs" else None,
                   "max_candidates_production": args.max_candidates, "max_seconds_end_to_end": args.max_seconds},
        "accepted_images": count, "generation_seconds": seconds, "pilot_seconds": pilot_seconds,
        "candidates_generated": pilot_forward + production_forward, "pilot_forward_candidates": pilot_forward,
        "production_forward_candidates": production_forward,
        "acceptance_rate": stats.get("acceptance_rate"), "retained_output_rate": stats.get("retained_output_rate"),
        "stats": stats, "official_metrics": {"FID": None, "precision": None, "recall": None},
    }
    if state is not None:
        record["pilot"] = {"noise_sha256": pilot_noise_hash, "initial_log_bound": state.initial_log_bound,
                           "initial_gamma": state.initial_gamma, "final_log_bound": state.log_bound,
                           "final_gamma": state.gamma, "bound_updates": state.updates}
        arrays = {key: torch.cat(audit[key]).numpy() if audit.get(key) else np.empty(0, dtype=np.int64 if key == "candidate_ids" else np.float64)
                  for key in ["candidate_ids", "accepted_scores", "accepted_probabilities", "accepted_uniforms"]}
        np.savez_compressed(directory / "accepted-audit.npz", **arrays)
        write_json(directory / "bound-updates.json", audit.get("bound_updates", []))
    write_json(directory / "metadata.json", record)
    print(f"seed={seed} {method}: {status}, {count} images, {record['candidates_generated']} forward candidates, "
          f"acceptance={record['acceptance_rate']}, {seconds:.3f}s", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scorer-dir", type=Path, default=ROOT / "runs/drs-auxiliary-sigma2-v1")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/baseline-seed0-e10/G.pth")
    parser.add_argument("--evaluator", type=Path, default=ROOT / "experiments/local-metrics-mnist64-k3")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data")
    parser.add_argument("--sigma", type=float, default=2.)
    parser.add_argument("--seeds", type=int, nargs="+", default=[21, 22])
    parser.add_argument("--pilot-seed", type=int, default=20001)
    parser.add_argument("--pilot-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-candidates", type=int, default=102400)
    parser.add_argument("--max-seconds", type=float, default=60.)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if (min(args.pilot_size, args.batch_size, args.threads) < 1 or args.max_candidates < 10000
            or args.max_candidates % args.batch_size != 0 or not math.isfinite(args.max_seconds) or args.max_seconds <= 0
            or not math.isfinite(args.sigma) or args.sigma <= 0
            or len(set(args.seeds)) != len(args.seeds) or args.pilot_seed in args.seeds
            or any(not 0 <= seed < 2**63 for seed in [*args.seeds, args.pilot_seed])):
        parser.error("Invalid counts, seeds, sigma or budget; max-candidates must be a multiple of batch-size")
    torch.set_num_threads(args.threads)
    configure_determinism()
    _, meta = load_calibrated_scorer(args.scorer_dir, args.checkpoint, args.sigma)
    if set([*args.seeds, args.pilot_seed]).intersection(meta["fake_seeds"].values()):
        parser.error("Use pilot/production seeds distinct from auxiliary fitting/calibration")
    args.generator_sha256, args.scorer_sha256 = meta["generator_sha256"], meta["scorer_sha256"]
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    with record_run(args.run_dir / "summary.json") as initial:
        started = time.perf_counter()
        sources = ["drs.py", "auxiliary_scorer.py", "discriminator_diagnostics.py", "sampling.py", "model.py",
                   "experiment_utils.py", "scripts/run_drs.py", "scripts/evaluate_local.py", "scripts/evaluate_frozen.py", "local_metrics.py"]
        sources.extend(["scripts/diagnose_discriminator.py", "requirements.txt"])
        hashes = {}
        for name in sources:
            destination = args.run_dir / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
            hashes[name] = digest(destination)
        summary = {**initial, "run_id": args.run_dir.name, "method": "I2-practical-DRS", "is_DRS": True,
                   "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "source_sha256": hashes, "scorer_metadata": meta, "scorer_directory": str(args.scorer_dir.resolve()),
                   "generator_checkpoint": str(args.checkpoint.resolve()),
                   "config": {key: value for key, value in vars(args).items() if isinstance(value, (int, float, list, str))},
                   "protocol": {"reference": "https://arxiv.org/abs/1810.06758", "equation": 8,
                                "approximate": True, "maximum_update": "batch maximum before Bernoulli draws",
                                "gamma": "median F on independent pilot, refreshed on max updates; no output-quality tuning",
                                "time_budget": "per condition, includes G/scorer loading, pilot, scoring and all PNG writes; cooperative batch checks",
                                "repeat_rule": "second seed only after first DRS condition completes its output and budgets"},
                   "hardware": {"device": "cpu", "threads": args.threads},
                   "packages": {name: importlib.metadata.version(name) for name in
                                ["torch", "torchvision", "numpy", "scipy", "scikit-learn", "pillow"]}, "results": []}
        write_json(args.run_dir / "summary.json", summary)
        for seed in args.seeds:
            control = sample_condition(args, seed, "unfiltered", args.run_dir / f"seed-{seed}/unfiltered")
            summary["results"].append(control)
            write_json(args.run_dir / "summary.json", summary)
            if control["status"] != "completed":
                break
            drs = sample_condition(args, seed, "drs", args.run_dir / f"seed-{seed}/drs")
            summary["results"].append(drs)
            if drs["status"] == "completed" and drs["stats"]["first_batch_noise_sha256"] != control["stats"]["first_batch_noise_sha256"]:
                raise RuntimeError("DRS and control do not share the same proposal-noise prefix")
            write_json(args.run_dir / "summary.json", summary)
            if drs["status"] != "completed":
                break
        complete = [r for r in summary["results"] if r["status"] == "completed"]
        if complete:
            write_json(args.run_dir / "evaluation-inputs.json", {"conditions": complete})
            comparison_grid(complete, args.run_dir / "comparison.png")
            subprocess.run([sys.executable, "-B", str(ROOT / "scripts/evaluate_frozen.py"),
                "--manifest", str(args.run_dir / "evaluation-inputs.json"), "--run-dir", str(args.run_dir / "local-metrics"),
                "--evaluator", str(args.evaluator.resolve()), "--data-path", str(args.data_path.resolve()),
                "--threads", str(args.threads)], check=True)
            summary["evaluation"] = json.loads((args.run_dir / "local-metrics/summary.json").read_text())
        if (digest(args.checkpoint) != args.generator_sha256 or digest(args.scorer_dir / "scorer.pth") != args.scorer_sha256
                or any(digest(ROOT / name) != expected for name, expected in hashes.items())):
            raise RuntimeError("Frozen weights or source changed")
        summary["trial_outcome"] = "completed" if len(complete) == 2 * len(args.seeds) else "budget_exhausted"
        summary["status"] = "completed"
        summary["total_seconds"] = time.perf_counter() - started
        summary["validation"] = {"frozen_weights_unchanged": True, "source_unchanged": True,
                                 "no_fallback": True, "partial_batches_not_evaluated": True}
        write_json(args.run_dir / "summary.json", summary)
        print(f"Trial outcome: {summary['trial_outcome']}", flush=True)


if __name__ == "__main__":
    main()
