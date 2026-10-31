"""Practical approximate DRS (Azadi et al., Eq. 8), with bounded streaming.

An estimated scorer, empirical running maximum and shifted acceptance do not
provide an exact density-ratio or target-distribution guarantee.
"""

import hashlib
import math
import time

import torch

from sampling import CandidateBatch


def base_acceptance_logits(scores, log_bound, epsilon=1e-6):
    """s-log(M)-log(1-exp(s-log(M)-epsilon)), stably in float64."""
    scores = torch.as_tensor(scores, dtype=torch.float64, device="cpu")
    if (scores.ndim != 1 or not torch.isfinite(scores).all()
            or not math.isfinite(log_bound) or not math.isfinite(epsilon) or epsilon <= 0
            or (scores > log_bound).any()):
        raise ValueError("Need finite vector scores bounded by log_bound and positive epsilon")
    delta = scores - log_bound
    return delta - torch.log(-torch.expm1(delta - epsilon))


class PracticalDRS:
    """Running empirical maximum and pilot-only percentile shift.

    Gamma is the fixed pilot's median F, recomputed when log(M) increases. This
    declared practical variant keeps the operating point near half acceptance
    without choosing gamma from output quality or each production batch.
    """

    def __init__(self, pilot_scores, epsilon=1e-6):
        self.pilot = torch.as_tensor(pilot_scores, dtype=torch.float64, device="cpu").clone()
        if self.pilot.ndim != 1 or not len(self.pilot) or not torch.isfinite(self.pilot).all():
            raise ValueError("Need a nonempty finite pilot score vector")
        self.epsilon = epsilon
        self.initial_log_bound = self.log_bound = self.pilot.max().item()
        self.gamma = self._gamma()
        self.initial_gamma = self.gamma
        self.updates = 0

    def _gamma(self):
        return torch.quantile(base_acceptance_logits(self.pilot, self.log_bound, self.epsilon), .5).item()

    def probabilities(self, scores):
        scores = torch.as_tensor(scores, dtype=torch.float64, device="cpu")
        if scores.ndim != 1 or not len(scores) or not torch.isfinite(scores).all():
            raise ValueError("Need a nonempty finite production score vector")
        maximum = scores.max().item()
        if maximum > self.log_bound:
            self.log_bound = maximum
            self.gamma = self._gamma()
            self.updates += 1
        probabilities = (base_acceptance_logits(scores, self.log_bound, self.epsilon) - self.gamma).sigmoid()
        if not torch.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
            raise RuntimeError("Invalid DRS probabilities")
        return probabilities


class SamplingBudgetExceeded(RuntimeError):
    pass


def rejection_batches(batches, score, state, *, count, max_candidates, max_seconds,
                      acceptance_seed, stats, audit, clock=time.perf_counter):
    """Yield exactly count retained outputs or fail; never fill from rejected ones.

    Candidate sources must preserve monotonically increasing IDs and respect
    their own forward budget. Time is checked around each batch, including the
    consumer's export time. A single kernel/write batch is not preempted.
    """
    if (type(count) is not int or count < 1 or type(max_candidates) is not int or max_candidates < count
            or not math.isfinite(max_seconds) or max_seconds <= 0
            or type(acceptance_seed) is not int or not 0 <= acceptance_seed < 2**63):
        raise ValueError("Invalid output count, candidate/time budget or acceptance seed")
    started = clock()
    rng = torch.Generator().manual_seed(acceptance_seed)
    noise_hash = hashlib.sha256()
    stats.update(candidates_tested=0, accepted_by_rule=0, retained_images=0,
                 dropped_final_acceptances=0, probability_sum=0., soft_probability_count=0,
                 maximum_probability=0., minimum_probability=1., acceptance_rate=0., retained_output_rate=0.)
    audit.update(candidate_ids=[], accepted_scores=[], accepted_probabilities=[], accepted_uniforms=[], bound_updates=[])
    def checkpoint():
        stats.update(production_seconds=clock() - started, base_noise_sha256=noise_hash.hexdigest(),
                     log_bound=state.log_bound, gamma=state.gamma, bound_update_count=state.updates)
        if stats["candidates_tested"]:
            stats["acceptance_rate"] = stats["accepted_by_rule"] / stats["candidates_tested"]
            stats["retained_output_rate"] = stats["retained_images"] / stats["candidates_tested"]
        if stats["production_seconds"] >= max_seconds:
            raise SamplingBudgetExceeded("time budget exhausted")
    iterator = iter(batches)
    previous_end = 0
    while stats["retained_images"] < count:
        checkpoint()
        if stats["candidates_tested"] >= max_candidates:
            raise SamplingBudgetExceeded("candidate budget exhausted")
        try:
            batch = next(iterator)
        except StopIteration:
            raise SamplingBudgetExceeded("candidate stream exhausted before output target") from None
        if batch.start != previous_end or len(batch.images) == 0:
            raise ValueError("Candidate batches must have consecutive, nonempty IDs starting at zero")
        if stats["candidates_tested"] + len(batch.images) > max_candidates:
            raise ValueError("Candidate source exceeded the declared processing budget")
        previous_end = batch.start + len(batch.images)
        scores = torch.as_tensor(score(batch.images), dtype=torch.float64, device="cpu")
        if scores.shape != (len(batch.images),):
            raise ValueError("Scorer must return one logit per candidate")
        previous_bound = state.log_bound
        probabilities = state.probabilities(scores)
        if state.log_bound != previous_bound:
            audit["bound_updates"].append({"candidate_start": batch.start, "log_bound": state.log_bound, "gamma": state.gamma})
        uniforms = torch.rand(len(probabilities), generator=rng, dtype=torch.float64)
        accepted = torch.where(uniforms < probabilities)[0]
        retained = accepted[:count - stats["retained_images"]]
        noise_hash.update(batch.eps.detach().cpu().numpy().tobytes())
        if stats["candidates_tested"] == 0:
            stats["first_batch_noise_sha256"] = noise_hash.hexdigest()
        stats["candidates_tested"] += len(batch.images)
        stats["accepted_by_rule"] += len(accepted)
        stats["probability_sum"] += probabilities.sum().item()
        stats["soft_probability_count"] += ((probabilities > .01) & (probabilities < .99)).sum().item()
        stats["minimum_probability"] = min(stats["minimum_probability"], probabilities.min().item())
        stats["maximum_probability"] = max(stats["maximum_probability"], probabilities.max().item())
        stats["dropped_final_acceptances"] += len(accepted) - len(retained)
        start = stats["retained_images"]
        stats["retained_images"] += len(retained)
        if len(retained):
            audit["candidate_ids"].append(retained + batch.start)
            audit["accepted_scores"].append(scores[retained])
            audit["accepted_probabilities"].append(probabilities[retained])
            audit["accepted_uniforms"].append(uniforms[retained])
        checkpoint()
        if len(retained):
            yield CandidateBatch(start, batch.images[retained], batch.eps[retained])
        checkpoint()
    stats["mean_acceptance_probability"] = stats["probability_sum"] / stats["candidates_tested"]
