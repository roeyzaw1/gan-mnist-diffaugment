"""Write a manifest for scripts/evaluate_frozen.py from completed runs.

Usage:
  python scripts/diffaug_manifest.py STUDY_DIR                  -> STUDY_DIR/manifest.json, all runs in it
  python scripts/diffaug_manifest.py --out M.json RUN_DIR ...   -> M.json, exactly these runs
run_id = "<condition>-seed<training seed>" in --out mode, else the run directory name.
"""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("paths", nargs="+", type=Path)
parser.add_argument("--out", type=Path, default=None)
args = parser.parse_args()
if args.out is None:
    study = args.paths[0].resolve()
    runs, out, with_seed = sorted(p for p in study.iterdir() if (p / "metadata.json").exists()), study / "manifest.json", False
else:
    runs, out, with_seed = [p.resolve() for p in args.paths], args.out.resolve(), True

conditions = []
for run in runs:
    meta = json.loads((run / "metadata.json").read_text())
    if meta.get("status") != "completed":
        raise SystemExit(f"{run} did not complete")
    checkpoint = run / "checkpoints" / "G.pth"
    cfg = meta["config"]
    conditions.append({
        "run_id": f"{run.name}-seed{cfg['seed']}" if with_seed else run.name, "method": meta["method"],
        "samples": str(run / "samples"), "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "config": {"training_seed": cfg["seed"], "generation_seed": cfg["generation_seed"],
                   "epochs": cfg["epochs"], "lr_G": cfg["lr_G"], "lr_D": cfg["lr_D"],
                   "generation_batch_size": cfg["generation_batch_size"], "sigma": 1.0,
                   "diffaugment_policy": cfg.get("diffaugment_policy", []),
                   "instance_noise_initial_std": cfg.get("instance_noise_initial_std", 0.0)}})
out.write_text(json.dumps({"conditions": conditions}, indent=2) + "\n")
print(f"{len(conditions)} conditions -> {out}")
