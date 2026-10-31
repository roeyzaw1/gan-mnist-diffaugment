import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from auxiliary_scorer import (AuxiliaryScorer, PREPROCESSING, calibrated_logits,
                              fit_sigmoid_calibration, load_calibrated_scorer,
                              real_splits, score_candidates, score_normalized)
from discriminator_diagnostics import starter_export_normalized

ROOT = Path(__file__).resolve().parents[1]


class AuxiliaryScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_splits_are_disjoint_excluded_and_reproducible(self):
        excluded = torch.tensor([0, 2, 4, 6], dtype=torch.int64)
        sizes = {"train": 8, "validation": 3, "calibration": 2, "diagnostic": 3}
        before = torch.get_rng_state().clone()
        splits = real_splits(excluded, total=20, sizes=sizes, seed=7)
        again = real_splits(excluded, total=20, sizes=sizes, seed=7)
        all_ids = torch.cat(list(splits.values()))
        self.assertEqual(len(all_ids.unique()), 16)
        self.assertTrue(set(all_ids.tolist()).isdisjoint(excluded.tolist()))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(splits[k], again[k]) for k in splits))
        with self.assertRaises(ValueError):
            real_splits(torch.tensor([1, 1]), total=20, sizes=sizes)
        with self.assertRaises(ValueError):
            real_splits(excluded, total=20, sizes={"train": 17})

    def test_sigmoid_fit_improves_distorted_probabilities_and_preserves_rank(self):
        rng = np.random.default_rng(20)
        real = rng.normal(1., 1., 2000) * 3 + 4
        fake = rng.normal(-1., 1., 2000) * 3 + 4
        calibration = fit_sigmoid_calibration(real, fake)
        self.assertGreater(calibration["slope"], 0)
        self.assertLess(calibration["fit_log_loss_after"], calibration["fit_log_loss_before"])
        scores = torch.tensor([-20., -1., 0., 1., 20.])
        fitted = calibrated_logits(scores, calibration)
        self.assertTrue((fitted[1:] > fitted[:-1]).all())
        with self.assertRaises(ValueError):
            fit_sigmoid_calibration(np.ones(2), np.ones(3))
        with self.assertRaises(ValueError):
            calibrated_logits(scores, {"slope": -1, "intercept": 0})

    def test_scoring_matches_export_space_and_keeps_parameters_unchanged(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2)
            model = AuxiliaryScorer()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        raw = torch.linspace(-1, 1, 5 * 784).reshape(5, 1, 28, 28)
        calibration = {"slope": .7, "intercept": .2}
        expected = score_normalized(model, starter_export_normalized(raw), 2, calibration)
        actual = score_candidates(model, raw, 2, calibration)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(actual.requires_grad)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
        self.assertEqual(score_candidates(model, raw[:0]).numel(), 0)
        with self.assertRaises(ValueError):
            score_candidates(model, raw * 2)

    def test_loader_binds_weights_to_proposal_and_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            generator = path / "G.pth"
            generator.write_bytes(b"proposal generator")
            model = AuxiliaryScorer()
            torch.save(model.state_dict(), path / "scorer.pth")
            np.savez(path / "split-indices.npz", train=np.arange(3))
            sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            metadata = {"status": "completed", "preprocessing": PREPROCESSING, "sigma": 1.,
                        "generator_sha256": sha(generator), "scorer_sha256": sha(path / "scorer.pth"),
                        "split_sha256": sha(path / "split-indices.npz"),
                        "calibration": {"slope": 1., "intercept": 0.},
                        "source_sha256": {name: sha(ROOT / name) for name in
                                          ["auxiliary_scorer.py", "discriminator_diagnostics.py"]}}
            (path / "scorer-metadata.json").write_text(json.dumps(metadata))
            before = torch.get_rng_state().clone()
            loaded, actual_metadata = load_calibrated_scorer(path, generator, sigma=1.)
            self.assertEqual(metadata, actual_metadata)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(loaded.state_dict()[key], value))
            with self.assertRaisesRegex(ValueError, "sigma"):
                load_calibrated_scorer(path, generator, sigma=2.)
            generator.write_bytes(b"different proposal")
            with self.assertRaisesRegex(ValueError, "different generator"):
                load_calibrated_scorer(path, generator, sigma=1.)

    def test_auxiliary_parameters_train_with_stable_bce(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4)
            model = AuxiliaryScorer()
        images = torch.cat([torch.full((8, 1, 28, 28), .8), torch.full((8, 1, 28, 28), -.8)])
        labels = torch.cat([torch.ones(8), torch.zeros(8)])
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        first_loss = None
        for _ in range(8):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(images), labels)
            first_loss = loss.item() if first_loss is None else first_loss
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), first_loss)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
