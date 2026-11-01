"""Input/provenance guards must reject mislabeled studies before creating outputs."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DiffAugmentScalingCLITests(unittest.TestCase):
    def test_invalid_scales_and_wrong_checkpoint_leave_no_run(self):
        cases = [(['--sigmas', '1', '1'], 'distinct'),
                 (['--sigmas', '1', 'nan'], 'finite'),
                 (['--checkpoint', str(ROOT / 'checkpoints/baseline-seed0-e10/G.pth')], 'hash differs')]
        with tempfile.TemporaryDirectory() as folder:
            for index, (options, message) in enumerate(cases):
                with self.subTest(options=options):
                    target = Path(folder) / f'run-{index}'
                    result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/compare_diffaugment_scaling.py'),
                                             '--run-dir', str(target), *options], cwd=ROOT, capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    self.assertFalse(target.exists())
