import math
import unittest

import numpy as np
import torch

from drs import (PracticalDRS, SamplingBudgetExceeded, base_acceptance_logits,
                 rejection_batches)
from sampling import CandidateBatch


def batches(total, size=8):
    for start in range(0, total, size):
        n = min(size, total - start)
        yield CandidateBatch(start, torch.zeros(n, 1, 28, 28),
                             torch.arange(start, start + n).float()[:, None].expand(-1, 100))


class DRSTests(unittest.TestCase):
    def test_stable_equation_matches_direct_formula_and_extremes(self):
        scores = torch.tensor([-1000000., -50., -5., -1., 0.], dtype=torch.float64)
        result = base_acceptance_logits(scores, 0.)
        self.assertTrue(torch.isfinite(result).all())
        expected = scores.numpy() - np.log(-np.expm1(scores.numpy() - 1e-6))
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-12)
        self.assertAlmostEqual(result[-1].item(), -math.log(-math.expm1(-1e-6)))

    def test_probabilities_are_bounded_monotone_and_update_empirical_max(self):
        state = PracticalDRS(torch.linspace(-20, -5, 1001))
        scores = torch.tensor([-1e6, -20., -12.5, -5., 1000.])
        probabilities = state.probabilities(scores)
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        self.assertTrue((probabilities[1:] >= probabilities[:-1]).all())
        self.assertEqual(state.log_bound, 1000.)
        self.assertEqual(state.updates, 1)
        self.assertAlmostEqual(probabilities[2].item(), .5, places=6)
        state.probabilities(torch.tensor([0.]))
        self.assertEqual(state.updates, 1)

    def test_constant_score_gives_probabilistic_half_acceptance(self):
        state = PracticalDRS(torch.zeros(20))
        self.assertTrue(torch.equal(state.probabilities(torch.zeros(8)), torch.full((8,), .5, dtype=torch.float64)))

    def test_stream_is_seeded_unique_exact_count_and_records_final_surplus(self):
        def run(seed):
            stats, audit = {}, {}
            output = list(rejection_batches(batches(64), lambda x: torch.zeros(len(x)), PracticalDRS(torch.zeros(8)),
                count=13, max_candidates=64, max_seconds=10., acceptance_seed=seed, stats=stats, audit=audit))
            return output, stats, audit
        before = torch.get_rng_state().clone()
        output, stats, audit = run(3)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        ids = torch.cat(audit['candidate_ids'])
        self.assertEqual(sum(len(x.images) for x in output), 13)
        self.assertEqual(ids.numel(), ids.unique().numel())
        self.assertTrue((ids[1:] > ids[:-1]).all())
        expected = torch.where(torch.rand(64, generator=torch.Generator().manual_seed(3), dtype=torch.float64) < .5)[0][:13]
        self.assertTrue(torch.equal(ids, expected))
        self.assertTrue(torch.equal(ids, torch.cat(run(3)[2]['candidate_ids'])))
        self.assertFalse(torch.equal(ids, torch.cat(run(4)[2]['candidate_ids'])))
        self.assertEqual(stats['accepted_by_rule'], 13 + stats['dropped_final_acceptances'])
        self.assertAlmostEqual(stats['acceptance_rate'], stats['accepted_by_rule']/stats['candidates_tested'])
        self.assertTrue((torch.cat(audit['accepted_uniforms']) < torch.cat(audit['accepted_probabilities'])).all())

    def test_candidate_budget_fails_without_fallback(self):
        stats, audit = {}, {}
        with self.assertRaisesRegex(SamplingBudgetExceeded, 'candidate budget'):
            list(rejection_batches(batches(16), lambda x: torch.full((len(x),), -1e6), PracticalDRS(torch.zeros(8)),
                count=4, max_candidates=16, max_seconds=10., acceptance_seed=3, stats=stats, audit=audit))
        self.assertEqual(stats['candidates_tested'], 16)
        self.assertEqual(stats['retained_images'], 0)
        self.assertEqual(audit['candidate_ids'], [])

    def test_time_budget_counts_consumer_export_time(self):
        now = [0.]
        stats, audit = {}, {}
        stream = rejection_batches(batches(64), lambda x: torch.zeros(len(x)), PracticalDRS(torch.zeros(8)),
            count=13, max_candidates=64, max_seconds=1., acceptance_seed=3, stats=stats, audit=audit,
            clock=lambda: now[0])
        next(stream)
        now[0] = 2.
        with self.assertRaisesRegex(SamplingBudgetExceeded, 'time budget'):
            next(stream)

    def test_finite_short_source_fails_and_bad_ids_are_rejected(self):
        with self.assertRaisesRegex(SamplingBudgetExceeded, 'stream exhausted'):
            list(rejection_batches(batches(8), lambda x: torch.full((len(x),), -1e6), PracticalDRS(torch.zeros(8)),
                count=4, max_candidates=16, max_seconds=10., acceptance_seed=3, stats={}, audit={}))
        wrong = [CandidateBatch(10, torch.zeros(8, 1, 28, 28), torch.zeros(8, 100))]
        with self.assertRaisesRegex(ValueError, 'consecutive'):
            list(rejection_batches(wrong, lambda x: torch.zeros(len(x)), PracticalDRS(torch.zeros(8)),
                count=4, max_candidates=16, max_seconds=10., acceptance_seed=3, stats={}, audit={}))

    def test_invalid_probabilities_and_parameters_are_rejected(self):
        for pilot in [torch.empty(0), torch.tensor([float('nan')])]:
            with self.assertRaises(ValueError):
                PracticalDRS(pilot)
        with self.assertRaises(ValueError):
            base_acceptance_logits(torch.ones(3), 0.)
        with self.assertRaises(ValueError):
            base_acceptance_logits(torch.zeros(3), 0., epsilon=0)
        with self.assertRaises(ValueError):
            list(rejection_batches(batches(8), lambda x: torch.zeros(len(x)), PracticalDRS(torch.zeros(8)),
                count=4, max_candidates=3, max_seconds=10., acceptance_seed=3, stats={}, audit={}))
