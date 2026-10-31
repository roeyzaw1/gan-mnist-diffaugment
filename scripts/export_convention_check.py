"""Does the PNG export convention change local metrics? Same checkpoint, same noise, two exports.

clip    : starter convention, save_image(x) on tanh output -> negatives become 0 (black).
rescale : save_image((x + 1) / 2), the exact inverse of training's Normalize(0.5, 0.5).
Writes <out>/<sigma>-<convention>/samples/0..9999.png and <out>/manifest.json for
scripts/evaluate_frozen.py. No training; generator weights are only read.
Usage: python scripts/export_convention_check.py OUT_DIR [--checkpoint ...] [--seed 1]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sampling import generate_candidates

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("out", type=Path)
parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/baseline-seed0-e10/G.pth")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 2.0])
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
conventions = {"clip": lambda x: x, "rescale": lambda x: (x + 1) / 2}
conditions = []
for sigma in args.sigmas:
    images = generate_candidates(10000, args.seed, args.checkpoint, sigma=sigma, batch_size=256)
    for name, transform in conventions.items():
        run_id = f"sigma{sigma:g}-{name}"
        samples = args.out / run_id / "samples"
        samples.mkdir(parents=True)
        for i, image in enumerate(images):
            save_image(transform(image), samples / f"{i}.png")
        conditions.append({"run_id": run_id, "method": f"export-{name}",
                           "samples": f"{run_id}/samples", "checkpoint": str(args.checkpoint.resolve()),
                           "checkpoint_sha256": digest,
                           "config": {"generation_seed": args.seed, "sigma": sigma, "export": name,
                                      "generation_batch_size": 256}})
        print(run_id, "exported", flush=True)
(args.out / "manifest.json").write_text(json.dumps({"conditions": conditions}, indent=2) + "\n")
