import io
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import torch
from torchvision.utils import save_image

from discriminator_diagnostics import (calibration_diagnostics, discriminator_logits,
                                       load_discriminator, score_summary, select_indices,
                                       starter_export_normalized)

ROOT = Path(__file__).resolve().parents[1]


class DiscriminatorDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_loaded_discriminator_preserves_rng_and_is_frozen(self):
        before = torch.get_rng_state().clone()
        model = load_discriminator(ROOT / "checkpoints/baseline-seed0-e10/D.pth")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_native_logits_match_forward_without_weight_or_hook_leaks(self):
        model = load_discriminator(ROOT / "checkpoints/baseline-seed0-e10/D.pth")
        before = {key: value.clone() for key, value in model.state_dict().items()}
        images = torch.rand(9, 1, 28, 28, generator=torch.Generator().manual_seed(3)) * 2 - 1
        logits = discriminator_logits(model, images, batch_size=4)
        with torch.no_grad():
            expected = torch.cat([model(x.flatten(1)) for x in images.split(4)]).flatten()
        self.assertTrue(torch.equal(logits.sigmoid(), expected))
        self.assertFalse(logits.requires_grad)
        self.assertEqual(len(model.fc4._forward_hooks), 0)
        self.assertTrue(all(torch.equal(before[key], value) for key, value in model.state_dict().items()))
        self.assertEqual(discriminator_logits(model, images[:0]).numel(), 0)

    def test_saturated_sigmoid_does_not_destroy_logit_scores(self):
        model = load_discriminator(ROOT / "checkpoints/baseline-seed0-e10/D.pth")
        with torch.no_grad():
            model.fc4.weight.zero_()
            model.fc4.bias.fill_(1000)
        logits = discriminator_logits(model, torch.zeros(3, 1, 28, 28))
        self.assertTrue(torch.equal(logits, torch.full((3,), 1000.)))
        self.assertEqual(score_summary(logits)["sigmoid_float32_exact_one"], 1.)

    def test_selection_has_stable_ties_unique_ids_and_private_rng(self):
        scores = torch.tensor([1., 5., 5., -10., 0.])
        self.assertEqual(select_indices(scores, 2, "top").tolist(), [1, 2])
        self.assertEqual(select_indices(torch.ones(5), 3, "top").tolist(), [0, 1, 2])
        self.assertEqual(select_indices(scores, 3, "first").tolist(), [0, 1, 2])
        before = torch.get_rng_state().clone()
        first = select_indices(torch.arange(20.), 7, "random", seed=8)
        again = select_indices(torch.arange(20.), 7, "random", seed=8)
        other = select_indices(torch.arange(20.), 7, "random", seed=9)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.equal(first, again))
        self.assertFalse(torch.equal(first, other))
        self.assertEqual(len(first.unique()), 7)
        self.assertEqual(first.tolist(), sorted(first.tolist()))

    def test_balanced_constant_probability_and_perfect_ranking_controls(self):
        neutral = calibration_diagnostics(torch.zeros(8), torch.zeros(8))
        self.assertEqual(neutral["roc_auc"], .5)
        self.assertEqual(neutral["brier"], .25)
        self.assertAlmostEqual(neutral["log_loss"], np.log(2))
        self.assertEqual(neutral["ece"], 0.)
        self.assertEqual(sum(row["count"] for row in neutral["reliability"]), 16)
        perfect = calibration_diagnostics(torch.full((8,), 10.), torch.full((8,), -10.))
        self.assertEqual(perfect["roc_auc"], 1.)
        self.assertEqual(perfect["accuracy_at_half"], 1.)
        self.assertLess(perfect["brier"], 1e-8)
        self.assertLess(perfect["ece"], 1e-4)
        inverted = calibration_diagnostics(torch.full((8,), -1000.), torch.full((8,), 1000.))
        self.assertEqual(inverted["log_loss"], 1000.)
        self.assertEqual(inverted["roc_auc"], 0.)

    def test_export_shift_mapping_matches_actual_starter_png_pixels(self):
        image = torch.linspace(-1, 1, 784).reshape(1, 28, 28)
        buffer = io.BytesIO()
        save_image(image, buffer, format="PNG")
        with Image.open(io.BytesIO(buffer.getvalue())) as png:
            expected = torch.from_numpy(np.array(png)[:, :, 0].copy()).float().div(255).sub(.5).div(.5)
        actual = starter_export_normalized(image[None])[0, 0]
        self.assertTrue(torch.equal(actual, expected))

    def test_invalid_scores_and_inputs_are_rejected(self):
        for scores, count, method in [(torch.ones(3), 0, "top"), (torch.ones(3), 4, "top"),
                                      (torch.tensor([float("nan")]), 1, "top"), (torch.ones(3), 2, "unknown")]:
            with self.assertRaises(ValueError):
                select_indices(scores, count, method)
        with self.assertRaises(ValueError):
            calibration_diagnostics(torch.ones(3), torch.ones(4))
        model = load_discriminator(ROOT / "checkpoints/baseline-seed0-e10/D.pth")
        for images in [torch.zeros(3, 784), torch.full((2, 1, 28, 28), float("nan")),
                       torch.full((2, 1, 28, 28), 2.)]:
            with self.assertRaises(ValueError):
                discriminator_logits(model, images)


if __name__ == "__main__":
    unittest.main()
