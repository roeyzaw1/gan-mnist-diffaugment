"""Inference-only sampling for the unchanged MNIST generator.

Noise is drawn on CPU in full batches, just like the starter exporter.
Explicit seeds include model initialization, preserving the recorded baseline's
RNG convention. Seed, batch size and sample index identify paired candidates.
"""

from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torchvision.utils import save_image

from model import Generator


@dataclass
class CandidateBatch:
    start: int
    images: torch.Tensor  # (N, 1, 28, 28), raw tanh outputs on the model device
    eps: torch.Tensor  # (N, 100), unscaled CPU noise for pairing/provenance


def validate_sampling(n, batch_size, sigma):
    if type(n) is not int or n < 0:
        raise ValueError("n must be a nonnegative integer")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(sigma, bool) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and strictly positive")


def sigma_tag(sigma):
    """Shortest round-trippable label; unlike :g, distinct scales cannot collide."""
    return str(float(sigma)).removesuffix(".0")


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")


def load_generator(checkpoint, device):
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "G.pth"
    model = Generator(784).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict({key.replace("module.", ""): value for key, value in state.items()})
    return model.eval()


def load_seeded_generator(checkpoint, seed, device):
    """Return G and its private CPU noise RNG; leave global RNG state intact."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = load_generator(checkpoint, device)
        noise_rng = torch.Generator(device="cpu")
        noise_rng.set_state(torch.get_rng_state())
    return model, noise_rng


def iter_candidates(model, n, batch_size=256, sigma=1.0, noise_generator=None):
    """Yield exact-count batches without retaining a gradient graph.

    Model construction happens outside this iterator. With noise_generator=None,
    CPU global RNG semantics match the starter. The final full noise/forward
    batch is drawn even if fewer images are retained, also matching the starter.
    """
    validate_sampling(n, batch_size, sigma)
    model.eval()
    device = next(model.parameters()).device
    for start in range(0, n, batch_size):
        eps = torch.randn(batch_size, 100, generator=noise_generator, device="cpu")
        z = eps.to(device)
        if sigma != 1.0:
            z = sigma * z
        with torch.no_grad():
            images = model(z).reshape(batch_size, 1, 28, 28)
            if not torch.isfinite(images).all():
                raise RuntimeError("Generator returned non-finite samples")
        count = min(batch_size, n - start)
        yield CandidateBatch(start, images[:count], eps[:count])


def generate_candidates(n, seed, checkpoint, sigma=1.0, batch_size=256,
                        device=None, return_noise=False):
    """Return raw CPU images; optionally also return their unscaled CPU noise.

    Outputs are (N,1,28,28) in [-1,1]. This function never exports/rescales images.
    Reuse seed AND batch_size for paired comparisons across sigma values.
    """
    validate_sampling(n, batch_size, sigma)
    device = default_device() if device is None else torch.device(device)
    model, noise_rng = load_seeded_generator(checkpoint, seed, device)
    images, noise = [], []
    for batch in iter_candidates(model, n, batch_size, sigma, noise_rng):
        images.append(batch.images.cpu())
        if return_noise:
            noise.append(batch.eps)
    result = torch.cat(images) if images else torch.empty(0, 1, 28, 28)
    if return_noise:
        return result, torch.cat(noise) if noise else torch.empty(0, 100)
    return result


def export_pngs(batches, output_dir, *, overwrite=False):
    """Write starter-compatible PNGs; require permission to reuse a directory.

    Experiment runners must provide a fresh directory. No rescaling is applied:
    save_image clips negative tanh outputs, preserving the starter convention.
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory already contains files: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for batch in batches:
        for offset, image in enumerate(batch.images):
            save_image(image, output_dir / f"{batch.start + offset}.png")
            count += 1
    return count
