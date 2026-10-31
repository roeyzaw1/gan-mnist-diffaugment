"""Regression test of the required CLI against the archived starter exporter."""

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class GenerationCLITests(unittest.TestCase):
    def test_explicit_output_is_protected_before_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "0.png"
            sentinel.write_bytes(b"previous experiment")
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "generate.py"),
                 "--output-dir", directory, "--checkpoint", "missing.pth"],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2)
            self.assertIn("--overwrite", result.stderr)
            self.assertEqual(sentinel.read_bytes(), b"previous experiment")

    def test_required_invocation_matches_all_starter_pngs(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = []
            scripts = [ROOT / "experiments/baseline-seed0-e10/source/generate.py", ROOT / "generate.py"]
            for index, script in enumerate(scripts):
                working = Path(directory) / str(index)
                (working / "checkpoints").mkdir(parents=True)
                (working / "checkpoints/G.pth").symlink_to(ROOT / "checkpoints/G.pth")
                code = (
                    "import runpy, sys, torch; "
                    f"sys.path.insert(0, {str(ROOT)!r}); "
                    "torch.set_num_threads(2); torch.manual_seed(1); "
                    "sys.argv=['generate.py', '--batch_size', '256']; "
                    f"runpy.run_path({str(script)!r}, run_name='__main__')"
                )
                subprocess.run([sys.executable, "-B", "-c", code], cwd=working,
                               check=True, capture_output=True, text=True, timeout=180)
                outputs.append(working / "samples")
            expected_names = {f"{i}.png" for i in range(10000)}
            for output in outputs:
                self.assertEqual({p.name for p in output.iterdir()}, expected_names)
            for name in expected_names:
                first, second = [output / name for output in outputs]
                self.assertEqual(hashlib.sha256(first.read_bytes()).digest(),
                                 hashlib.sha256(second.read_bytes()).digest(), name)
                with Image.open(second) as image:
                    image.load()
                    self.assertEqual((image.format, image.mode, image.size), ("PNG", "RGB", (28, 28)))


if __name__ == "__main__":
    unittest.main()
