"""Write <study>/manifest.json for scripts/evaluate_frozen.py from completed runs in <study>/."""
import hashlib
import json
import sys
from pathlib import Path

study = Path(sys.argv[1]).resolve()
conditions = []
for run in sorted(p for p in study.iterdir() if (p / "metadata.json").exists()):
    meta = json.loads((run / "metadata.json").read_text())
    if meta.get("status") != "completed":
        raise SystemExit(f"{run.name} did not complete")
    checkpoint = run / "checkpoints" / "G.pth"
    cfg = meta["config"]
    conditions.append({
        "run_id": run.name, "method": meta["method"],
        "samples": f"{run.name}/samples", "checkpoint": f"{run.name}/checkpoints/G.pth",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "config": {"training_seed": cfg["seed"], "generation_seed": cfg["generation_seed"],
                   "epochs": cfg["epochs"], "lr_G": cfg["lr_G"], "lr_D": cfg["lr_D"],
                   "generation_batch_size": cfg["generation_batch_size"], "sigma": 1.0,
                   "diffaugment_policy": cfg.get("diffaugment_policy", []),
                   "instance_noise_initial_std": cfg.get("instance_noise_initial_std", 0.0)}})
(study / "manifest.json").write_text(json.dumps({"conditions": conditions}, indent=2) + "\n")
print(f"{len(conditions)} conditions -> {study / 'manifest.json'}")
