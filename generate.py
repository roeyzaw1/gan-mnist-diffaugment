"""Generate exactly 10,000 PNGs with optional inference-time latent scaling."""

import argparse
from pathlib import Path

import torch

from sampling import (default_device, export_pngs, iter_candidates,
                      load_generator, load_seeded_generator, validate_sampling)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--sigma", type=float, default=1.0, help="Scale of Gaussian latent noise; baseline: 1.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed, including model initialization.")
    parser.add_argument("--checkpoint", default="checkpoints/G.pth")
    parser.add_argument("--output-dir", "--output_dir", default=None)
    parser.add_argument("--overwrite", action="store_true", help="Allow reusing an explicit output directory.")
    args = parser.parse_args()
    try:
        validate_sampling(10000, args.batch_size, args.sigma)
        if args.seed is not None and not 0 <= args.seed < 2**63:
            raise ValueError("seed must be in [0, 2**63)")
        if args.output_dir is not None:
            output = Path(args.output_dir)
            if output.exists() and (not output.is_dir() or (any(output.iterdir()) and not args.overwrite)):
                raise ValueError("Explicit output directory already contains files; use a new path or --overwrite")
    except ValueError as error:
        parser.error(str(error))

    device = default_device()
    print(f"Using device: {device}")
    print("Model Loading...")
    if args.seed is None:
        model, noise_rng = load_generator(args.checkpoint, device), None
    else:
        model, noise_rng = load_seeded_generator(args.checkpoint, args.seed, device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.eval()
    print("Model loaded.")
    print("Start Generating")
    batches = iter_candidates(model, 10000, args.batch_size, args.sigma, noise_rng)
    # The required default invocation retains the starter's overwrite semantics.
    export_pngs(batches, args.output_dir or "samples",
                overwrite=args.output_dir is None or args.overwrite)


if __name__ == "__main__":
    main()
