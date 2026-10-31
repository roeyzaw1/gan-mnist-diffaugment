"""A separate logistic scorer for one frozen G/proposal/export configuration."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch
from torch import nn

from discriminator_diagnostics import starter_export_normalized

PREPROCESSING = "starter-save-image-uint8-then-(pixel/255-0.5)/0.5-v1"


class AuxiliaryScorer(nn.Module):
    """Fresh 784→256→128→1 MLP; output is a logit, not a GAN generator."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.LeakyReLU(.2),
                                     nn.Linear(256, 128), nn.LeakyReLU(.2), nn.Linear(128, 1))

    def forward(self, images):
        return self.network(images).flatten()


def real_splits(excluded, *, total=60000, seed=3028, sizes=None):
    """Private seeded partition; no overlap with evaluator reference/control IDs."""
    sizes = sizes or {"train": 20000, "validation": 5000, "calibration": 5000, "diagnostic": 10000}
    excluded = torch.as_tensor(excluded)
    if (excluded.ndim != 1 or excluded.dtype != torch.int64
            or len(excluded.unique()) != len(excluded) or (excluded < 0).any() or (excluded >= total).any()
            or any(type(n) is not int or n < 1 for n in sizes.values())
            or sum(sizes.values()) > total - len(excluded)):
        raise ValueError("Invalid excluded indices or split sizes")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be in [0,2**63)")
    mask = torch.ones(total, dtype=torch.bool)
    mask[excluded] = False
    available = torch.arange(total)[mask]
    order = available[torch.randperm(len(available), generator=torch.Generator().manual_seed(seed))]
    splits, start = {}, 0
    for name, size in sizes.items():
        splits[name] = order[start:start + size].clone()
        start += size
    return splits


def fit_sigmoid_calibration(real_logits, fake_logits):
    """Fit positive affine logits on disjoint balanced calibration examples.

    A positive slope preserves score ranking. Parameters are approximate
    probability calibration, not a proof of exact density-ratio recovery.
    """
    real, fake = [np.asarray(x, dtype=np.float64) for x in (real_logits, fake_logits)]
    if (real.ndim != 1 or fake.ndim != 1 or len(real) == 0 or len(real) != len(fake)
            or not np.isfinite(real).all() or not np.isfinite(fake).all()):
        raise ValueError("Expected balanced nonempty finite calibration vectors")
    scores = np.concatenate([real, fake])
    labels = np.concatenate([np.ones_like(real), np.zeros_like(fake)])
    def objective(parameters):
        slope, intercept = parameters
        logits = slope * scores + intercept
        loss = np.where(labels == 1, np.logaddexp(0, -logits), np.logaddexp(0, logits)).mean()
        residual = expit(logits) - labels
        gradient = np.array([(residual * scores).mean(), residual.mean()])
        return float(loss), gradient
    result = minimize(objective, [1., 0.], method="L-BFGS-B", jac=True,
                      bounds=[(1e-8, None), (None, None)],
                      options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-9})
    if not result.success or not np.isfinite(result.x).all() or not math.isfinite(result.fun):
        raise RuntimeError(f"Calibration did not converge: {result.message}")
    return {"method": "positive-affine-sigmoid", "slope": float(result.x[0]),
            "intercept": float(result.x[1]), "equal_real_fake_priors": True,
            "n_real": len(real), "n_fake": len(fake),
            "fit_log_loss_before": objective([1., 0.])[0], "fit_log_loss_after": float(result.fun),
            "iterations": int(result.nit)}


def calibrated_logits(logits, calibration):
    slope, intercept = calibration["slope"], calibration["intercept"]
    if not math.isfinite(slope) or slope <= 0 or not math.isfinite(intercept):
        raise ValueError("Calibration requires a finite positive slope and finite intercept")
    result = torch.as_tensor(logits, dtype=torch.float64) * slope + intercept
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite calibrated logits")
    return result


@torch.no_grad()
def score_normalized(model, images, batch_size=256, calibration=None):
    """Score images already in the final exported/normalized image space."""
    if (images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28)
            or not torch.isfinite(images).all() or (images < -1).any() or (images > 1).any()
            or type(batch_size) is not int or batch_size < 1):
        raise ValueError("Expected finite normalized (N,1,28,28) images and positive batch size")
    model.eval()
    device = next(model.parameters()).device
    blocks = [model(block.to(device)).detach().cpu() for block in images.split(batch_size) if len(block)]
    logits = torch.cat(blocks) if blocks else torch.empty(0)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Scorer returned non-finite logits")
    return calibrated_logits(logits, calibration) if calibration is not None else logits


def score_candidates(model, raw_images, batch_size=256, calibration=None):
    """Future rejection samplers must score the exported version of raw G outputs."""
    if not torch.isfinite(raw_images).all() or (raw_images < -1).any() or (raw_images > 1).any():
        raise ValueError("Raw generator outputs must be finite in [-1,1]")
    return score_normalized(model, starter_export_normalized(raw_images), batch_size, calibration)


def load_calibrated_scorer(directory, generator_checkpoint, sigma):
    """Refuse accidental reuse with different G weights, sigma or preprocessing."""
    directory = Path(directory)
    metadata = json.loads((directory / "scorer-metadata.json").read_text())
    if metadata["status"] != "completed" or metadata["preprocessing"] != PREPROCESSING:
        raise ValueError("Incomplete or incompatible scorer metadata")
    for name in ["auxiliary_scorer.py", "discriminator_diagnostics.py"]:
        if hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() != metadata["source_sha256"][name]:
            raise ValueError(f"Scorer implementation differs from its saved protocol: {name}")
    if (isinstance(sigma, bool) or not math.isfinite(sigma) or sigma <= 0
            or float(sigma) != metadata["sigma"]):
        raise ValueError("Scorer sigma differs from proposal sigma; train a separate scorer")
    if hashlib.sha256(Path(generator_checkpoint).read_bytes()).hexdigest() != metadata["generator_sha256"]:
        raise ValueError("Scorer belongs to a different generator checkpoint")
    checkpoint = directory / "scorer.pth"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metadata["scorer_sha256"]:
        raise ValueError("Scorer weight hash mismatch")
    if hashlib.sha256((directory / "split-indices.npz").read_bytes()).hexdigest() != metadata["split_sha256"]:
        raise ValueError("Scorer split hash mismatch")
    # Validate calibration even if callers request the uncalibrated logits later.
    calibrated_logits(torch.zeros(1), metadata["calibration"])
    with torch.random.fork_rng(devices=[]):
        model = AuxiliaryScorer()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.eval().requires_grad_(False), metadata
