"""Regression checks for experiment integrity and fail-fast validation."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiment_utils import (evaluation_conditions, record_run,
                              validate_evaluation_sweep, write_json)


class ExperimentUtilitiesTests(unittest.TestCase):
    def test_invalid_json_does_not_replace_valid_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            write_json(path, {"score": 1})
            with self.assertRaises(ValueError):
                write_json(path, {"score": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"score": 1})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_failed_and_interrupted_runs_keep_partial_results(self):
        for error, status in [(RuntimeError("setup failed"), "failed"),
                              (KeyboardInterrupt(), "interrupted")]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "summary.json"
                with self.assertRaises(type(error)):
                    with record_run(path) as initial:
                        write_json(path, {**initial, "results": [{"seed": 1}]})
                        raise error
                state = json.loads(path.read_text())
                self.assertEqual(state["status"], status)
                self.assertEqual(state["results"], [{"seed": 1}])
                self.assertIn("finished_at_utc", state)
                self.assertIn(type(error).__name__, state["error"])

    def test_completed_run_preserves_shared_checkpoint_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            with record_run(path) as initial:
                state = {**initial, "status": "completed", "finished_at_utc": "saved timestamp"}
                write_json(path, state)
            self.assertEqual(json.loads(path.read_text()), state)

    def test_sweep_validation_rejects_partial_or_mismatched_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "G.pth"
            checkpoint.write_bytes(b"frozen generator")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            samples = root / "seed-1/sigma-1/samples"
            samples.mkdir(parents=True)
            for i in range(4):
                (samples / f"{i}.png").touch()
            valid = {"status": "completed", "checkpoint": "G.pth", "checkpoint_sha256": digest,
                     "config": {"num_samples": 4, "batch_size": 2, "seeds": [1], "sigmas": [1.]},
                     "results": [{"seed": 1, "sigma": 1., "accepted_images": 4,
                                  "checkpoint_sha256": digest}]}
            path = root / "summary.json"
            write_json(path, valid)
            self.assertEqual(validate_evaluation_sweep(root, root, n=4), valid)
            mutations = [lambda x: x.update(status="running"),
                         lambda x: x["config"].update(num_samples=3),
                         lambda x: x["config"].update(seeds=[1, 1]),
                         lambda x: x["config"].update(sigmas=[1., 2.]),
                         lambda x: x["results"].append(x["results"][0]),
                         lambda x: x["results"][0].update(accepted_images=3),
                         lambda x: x["results"][0].update(checkpoint_sha256="wrong")]
            for mutate in mutations:
                invalid = copy.deepcopy(valid)
                mutate(invalid)
                write_json(path, invalid)
                with self.assertRaises(ValueError):
                    validate_evaluation_sweep(root, root, n=4)
            write_json(path, valid)
            (samples / "3.png").unlink()
            with self.assertRaisesRegex(ValueError, "Missing"):
                validate_evaluation_sweep(root, root, n=4)
            checkpoint.write_bytes(b"changed generator")
            with self.assertRaisesRegex(ValueError, "checkpoint hash"):
                validate_evaluation_sweep(root, root, n=4)

    def test_invalid_inputs_fail_before_creating_runs_or_loading_data(self):
        from scripts import baseline, evaluate_local
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "unused-run"
            for flags in [["--lr", "nan"], ["--generation-seed", "-1"]]:
                with patch("sys.argv", ["baseline.py", "--run-dir", str(run), *flags]):
                    with self.assertRaises(SystemExit) as result:
                        baseline.main()
                    self.assertEqual(result.exception.code, 2)
                    self.assertFalse(run.exists())
            sweep = Path(directory) / "sweep"
            sweep.mkdir()
            write_json(sweep / "summary.json", {"status": "running"})
            with patch("sys.argv", ["evaluate_local.py", "--run-dir", str(run),
                                    "--sweep", str(sweep)]):
                with self.assertRaises(SystemExit) as result:
                    evaluate_local.main()
                self.assertEqual(result.exception.code, 2)
                self.assertFalse(run.exists())

    def test_generic_conditions_preserve_method_provenance_without_sigma(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = root / "manifests"
            manifests.mkdir()
            samples = root / "diffaugment-seed0/generated-images"
            samples.mkdir(parents=True)
            for i in range(4):
                (samples / f"{i}.png").touch()
            checkpoint = samples.parent / "G.pth"
            checkpoint.write_bytes(b"different trained generator")
            row = {"run_id": "diffaugment-seed0", "method": "DiffAugment",
                   "config": {"training_seed": 0, "generation_seed": 1, "policy": "translation"},
                   "samples": "../diffaugment-seed0/generated-images", "checkpoint": "../diffaugment-seed0/G.pth",
                   "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
            manifest = manifests / "inputs.json"
            write_json(manifest, {"conditions": [row]})
            actual = evaluation_conditions(root=root, manifest=manifest, n=4)
            self.assertEqual(actual, [{**row, "samples": str(samples), "checkpoint": str(checkpoint)}])
            self.assertNotIn("sigma", actual[0])
            mutations = [lambda rows: rows.append(rows[0]),
                         lambda rows: rows[0].update(config=[]),
                         lambda rows: rows[0].update(checkpoint_sha256="wrong"),
                         lambda rows: rows[0].update(method=""),
                         lambda rows: rows[0].update(run_id="")]
            for mutate in mutations:
                rows = copy.deepcopy([row])
                mutate(rows)
                write_json(manifest, {"conditions": rows})
                with self.assertRaises(ValueError):
                    evaluation_conditions(root=root, manifest=manifest, n=4)
            write_json(manifest, {"conditions": [row]})
            (samples / "3.png").unlink()
            with self.assertRaisesRegex(ValueError, "exactly 4"):
                evaluation_conditions(root=root, manifest=manifest, n=4)

    def test_evaluation_requires_one_unambiguous_input_source(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            evaluation_conditions(root=Path.cwd())
        with self.assertRaisesRegex(ValueError, "exactly one"):
            evaluation_conditions(root=Path.cwd(), manifest="inputs.json", sweep_dir="sweep")
