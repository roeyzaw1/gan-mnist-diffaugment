import hashlib
import io
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch
from torchvision.utils import save_image

from model import Generator
from sampling import (export_pngs, generate_candidates, iter_candidates,
                      load_seeded_generator, sigma_tag)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints/baseline-seed0-e10/G.pth"


class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)

    def test_sigma_one_matches_starter_rng_and_forward(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1)
            original = Generator(784).eval()
            original.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
            with torch.no_grad():
                expected = original(torch.randn(256, 100)).reshape(256, 1, 28, 28)[:17]
        actual = generate_candidates(17, 1, CHECKPOINT, batch_size=256, device="cpu")
        self.assertTrue(torch.equal(expected, actual))

    def test_sigma_one_matches_archived_baseline_png(self):
        actual = generate_candidates(1, 1, CHECKPOINT, batch_size=256, device="cpu")
        buffer = io.BytesIO()
        save_image(actual[0], buffer, format="PNG")
        with Image.open(ROOT / "experiments/baseline-seed0-e10/sample-grid.png") as grid:
            # save_image's default grid padding is two pixels on each side.
            expected = grid.crop((2, 2, 30, 30))
            with Image.open(io.BytesIO(buffer.getvalue())) as image:
                self.assertEqual(image.tobytes(), expected.tobytes())

    def test_scaling_reuses_noise_and_applies_to_latents(self):
        baseline, eps = generate_candidates(19, 8, CHECKPOINT, batch_size=8, return_noise=True, device="cpu")
        scaled, other_eps = generate_candidates(19, 8, CHECKPOINT, sigma=0.7, batch_size=8,
                                                 return_noise=True, device="cpu")
        self.assertTrue(torch.equal(eps, other_eps))
        model, _ = load_seeded_generator(CHECKPOINT, 8, torch.device("cpu"))
        with torch.no_grad():
            expected = model(eps[:8] * 0.7).reshape(8, 1, 28, 28)
        self.assertTrue(torch.equal(scaled[:8], expected))
        self.assertFalse(torch.equal(baseline, scaled))

    def test_exact_count_range_and_no_grad_or_weight_mutation(self):
        model, rng = load_seeded_generator(CHECKPOINT, 3, torch.device("cpu"))
        before = {key: value.clone() for key, value in model.state_dict().items()}
        grad_enabled = torch.is_grad_enabled()
        batches = list(iter_candidates(model, 19, batch_size=8, sigma=1.3, noise_generator=rng))
        self.assertEqual([len(batch.images) for batch in batches], [8, 8, 3])
        self.assertEqual([batch.start for batch in batches], [0, 8, 16])
        self.assertEqual(torch.is_grad_enabled(), grad_enabled)
        images = torch.cat([batch.images for batch in batches])
        self.assertEqual(tuple(images.shape), (19, 1, 28, 28))
        self.assertFalse(images.requires_grad)
        self.assertTrue(torch.isfinite(images).all())
        self.assertTrue((images >= -1).all() and (images <= 1).all())
        self.assertTrue(all(torch.equal(value, before[key]) for key, value in model.state_dict().items()))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1486352)

    def test_private_seed_does_not_change_global_rng(self):
        state = torch.get_rng_state().clone()
        first = generate_candidates(4, 1, CHECKPOINT, batch_size=4, device="cpu")
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        again = generate_candidates(4, 1, CHECKPOINT, batch_size=4, device="cpu")
        other = generate_candidates(4, 2, CHECKPOINT, batch_size=4, device="cpu")
        self.assertTrue(torch.equal(first, again))
        self.assertFalse(torch.equal(first, other))

    def test_export_starter_format_and_pixel_mapping(self):
        model, rng = load_seeded_generator(CHECKPOINT, 1, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as directory:
            count = export_pngs(iter_candidates(model, 5, 3, 0.85, rng), directory)
            self.assertEqual(count, 5)
            self.assertEqual({p.name for p in Path(directory).iterdir()}, {f"{i}.png" for i in range(5)})
            with Image.open(Path(directory) / "0.png") as image:
                self.assertEqual((image.format, image.mode, image.size), ("PNG", "RGB", (28, 28)))

    def test_invalid_sampling_parameters(self):
        for sigma in [0, -1, float("nan"), float("inf"), True]:
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                generate_candidates(1, 1, CHECKPOINT, sigma=sigma)
        for n, batch_size in [(-1, 8), (1, 0), (1, -1), (True, 8), (1, True)]:
            with self.subTest(n=n, batch_size=batch_size), self.assertRaises(ValueError):
                generate_candidates(n, 1, CHECKPOINT, batch_size=batch_size)
        for seed in [-1, 2**63, True]:
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                generate_candidates(1, seed, CHECKPOINT)

    def test_export_refuses_existing_outputs_before_generating(self):
        def unexpected_generation():
            self.fail("Existing output must be rejected before generating")
            yield
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "0.png"
            sentinel.write_bytes(b"previous experiment")
            with self.assertRaises(FileExistsError):
                export_pngs(unexpected_generation(), directory)
            self.assertEqual(sentinel.read_bytes(), b"previous experiment")
            model, rng = load_seeded_generator(CHECKPOINT, 1, torch.device("cpu"))
            self.assertEqual(export_pngs(iter_candidates(model, 1, 1, 1, rng),
                                         directory, overwrite=True), 1)
            with Image.open(sentinel) as image:
                self.assertEqual(image.size, (28, 28))

    def test_sigma_labels_preserve_distinct_scales(self):
        self.assertEqual(sigma_tag(1.0), "1")
        self.assertEqual(sigma_tag(1.75), "1.75")
        self.assertNotEqual(sigma_tag(1.0), sigma_tag(1.0000001))

    def test_empty_pool_and_checkpoint_hash(self):
        before = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
        empty = generate_candidates(0, 0, CHECKPOINT, device="cpu")
        self.assertEqual(tuple(empty.shape), (0, 1, 28, 28))
        self.assertEqual(before, hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
