"""Diagnostics for the original equal-prior BCE discriminator; no DRS sampler."""

from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from model import Discriminator


def load_discriminator(checkpoint, device="cpu"):
    """Load the unchanged sigmoid discriminator without consuming global CPU RNG."""
    with torch.random.fork_rng(devices=[]):
        model = Discriminator(784)
    state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def discriminator_logits(model, images, batch_size=256):
    """Read pre-sigmoid scores via a temporary hook on the original final layer.

    Inputs use the GAN training convention, (N,1,28,28) in [-1,1]. No logit of
    rounded probabilities, clipping or architecture substitution is involved.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
        raise ValueError("Expected images with shape (N,1,28,28)")
    if not torch.isfinite(images).all() or (images < -1).any() or (images > 1).any():
        raise ValueError("Images must be finite and normalized to [-1,1]")
    captured, scores = [], []
    def capture(module, inputs, output):
        captured.append(output.detach().reshape(-1).cpu())
    handle = model.fc4.register_forward_hook(capture)
    try:
        model.eval()
        device = next(model.parameters()).device
        for block in images.split(batch_size):
            if len(block):
                model(block.flatten(1).to(device))
        scores = torch.cat(captured) if captured else torch.empty(0)
    finally:
        handle.remove()
    if not torch.isfinite(scores).all():
        raise RuntimeError("Discriminator returned non-finite logits")
    return scores


def select_indices(scores, count, method, seed=0):
    """Unique indices in original candidate order; deterministic tie handling."""
    if scores.ndim != 1 or not torch.isfinite(scores).all():
        raise ValueError("Expected a finite score vector")
    if type(count) is not int or not 1 <= count <= len(scores):
        raise ValueError("count must be between 1 and the candidate pool size")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if method == "top":
        indices = torch.argsort(scores, descending=True, stable=True)[:count]
    elif method == "random":
        indices = torch.randperm(len(scores), generator=torch.Generator().manual_seed(seed))[:count]
    elif method == "first":
        indices = torch.arange(count)
    else:
        raise ValueError("method must be top, random or first")
    return indices.sort().values


def score_summary(logits):
    logits = torch.as_tensor(logits, dtype=torch.float64)
    probabilities = logits.sigmoid()
    actual_probabilities = logits.float().sigmoid()
    q = torch.tensor([0., .01, .1, .5, .9, .99, 1.], dtype=torch.float64)
    return {
        "count": len(logits), "logit_mean": logits.mean().item(),
        "logit_quantiles": torch.quantile(logits, q).tolist(),
        "probability_mean": probabilities.mean().item(),
        "probability_quantiles": torch.quantile(probabilities, q).tolist(),
        "probability_at_most_1e-4": (probabilities <= 1e-4).double().mean().item(),
        "probability_at_least_1_minus_1e-4": (probabilities >= 1 - 1e-4).double().mean().item(),
        "sigmoid_float32_exact_zero": (actual_probabilities == 0).double().mean().item(),
        "sigmoid_float32_exact_one": (actual_probabilities == 1).double().mean().item(),
    }


def calibration_diagnostics(real_logits, fake_logits, bins=15):
    """Balanced real-vs-generated assessment, without fitting calibration.

    Brier/log loss measure probability quality, not calibration alone. Reliability
    bins and ECE are descriptive finite-bin calibration estimates.
    """
    real, fake = [torch.as_tensor(x, dtype=torch.float64) for x in (real_logits, fake_logits)]
    if (real.ndim != 1 or fake.ndim != 1 or len(real) == 0 or len(real) != len(fake)
            or not torch.isfinite(real).all() or not torch.isfinite(fake).all()
            or type(bins) is not int or bins < 1):
        raise ValueError("Need finite, equally sized nonempty real/fake vectors and positive bins")
    logits = torch.cat([real, fake])
    labels = torch.cat([torch.ones_like(real), torch.zeros_like(fake)])
    probabilities = logits.sigmoid()
    bin_ids = (probabilities * bins).long().clamp_max(bins - 1)
    reliability, ece = [], 0.
    for i in range(bins):
        mask = bin_ids == i
        count = mask.sum().item()
        if count:
            confidence, frequency = probabilities[mask].mean().item(), labels[mask].mean().item()
            ece += count / len(logits) * abs(confidence - frequency)
            reliability.append({"bin": i, "lower": i / bins, "upper": (i + 1) / bins,
                                "count": count, "mean_probability_real": confidence,
                                "observed_real_fraction": frequency})
    return {
        "equal_real_fake_priors": True, "n_real": len(real), "n_fake": len(fake),
        "roc_auc": float(roc_auc_score(labels.numpy(), logits.numpy())),
        "brier": (probabilities - labels).square().mean().item(),
        "log_loss": (torch.nn.functional.softplus(logits) - labels * logits).mean().item(),
        "accuracy_at_half": ((logits >= 0) == labels.bool()).double().mean().item(),
        "ece": ece, "ece_bins": bins, "reliability": reliability,
        "constant_half_baseline": {"brier": .25, "log_loss": float(np.log(2)), "accuracy": .5},
        "real": score_summary(real), "fake": score_summary(fake),
    }


def starter_export_normalized(images):
    """Exact save_image uint8 mapping, then evaluator/D normalization.

    Only used to diagnose export shift; actual metric batches are read from PNG.
    """
    pixels = images.mul(255).add(.5).clamp(0, 255).to(torch.uint8)
    return pixels.float().div(255).sub(.5).div(.5)
