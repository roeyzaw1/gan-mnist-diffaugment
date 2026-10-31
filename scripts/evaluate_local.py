"""Train/freeze a MNIST feature extractor and evaluate declared sample sets."""

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_utils import configure_determinism, record_run, write_json
from experiment_utils import evaluation_conditions
from local_metrics import MNISTFeatures, frechet_distance, knn_radii, precision_recall


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def extract(model, images, batch_size=256):
    features, predictions, confidence = [], [], []
    for block in images.split(batch_size):
        f = model.features(block)
        probability = model.head(f).softmax(1)
        features.append(f)
        predictions.append(probability.argmax(1))
        confidence.append(probability.max(1).values)
    return torch.cat(features), torch.cat(predictions), torch.cat(confidence)


def read_pngs(directory, n=10000):
    if {p.name for p in directory.iterdir()} != {f"{i}.png" for i in range(n)}:
        raise ValueError(f"Expected exactly {n} numbered images in {directory}")
    images, h = [], hashlib.sha256()
    for i in range(n):
        path = directory / f"{i}.png"
        h.update(path.read_bytes())
        with Image.open(path) as image:
            image.load()
            if image.size != (28, 28) or image.mode != "RGB" or image.format != "PNG":
                raise ValueError(f"Invalid PNG: {path}")
            rgb = np.array(image)
            if not (np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(rgb[:, :, 0], rgb[:, :, 2])):
                raise ValueError(f"Expected replicated grayscale channels: {path}")
            images.append(torch.from_numpy(rgb[:, :, 0].copy()))
    return torch.stack(images).unsqueeze(1).float().div_(255).sub_(0.5).div_(0.5), h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--sweep", type=Path, help="Legacy paired latent-scaling sweep.")
    inputs.add_argument("--manifest", type=Path, help="Method-independent sample/checkpoint manifest.")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()
    if min(args.epochs, args.threads, args.chunk_size) < 1:
        parser.error("epochs, threads and chunk size must be positive")
    try:
        conditions = evaluation_conditions(root=ROOT, manifest=args.manifest,
                                           sweep_dir=args.sweep)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    if not 0 <= args.seed < 2**63:
        parser.error("seed must be in [0, 2**63)")
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    with record_run(run / "summary.json") as run_state:
        start = time.perf_counter()
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        configure_determinism()
        dataset = MNIST(str(args.data_path), train=True, download=False)
        permutation = torch.randperm(60000, generator=torch.Generator().manual_seed(args.seed))
        split = {"classifier_train": permutation[:35000], "classifier_validation": permutation[35000:40000],
                 "real_reference": permutation[40000:50000], "real_control": permutation[50000:]}
        np.savez(run / "split-indices.npz", **{key: value.numpy() for key, value in split.items()})
        def subset(key):
            indices = split[key]
            return dataset.data[indices].unsqueeze(1).float().div(255).sub(0.5).div(0.5), dataset.targets[indices]
        train_x, train_y = subset("classifier_train")
        val_x, val_y = subset("classifier_validation")
        reference_x, reference_y = subset("real_reference")
        control_x, control_y = subset("real_control")
        model = MNISTFeatures()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loader = DataLoader(TensorDataset(train_x, train_y), batch_size=256, shuffle=True, num_workers=0)
        best_accuracy = -1
        training = []
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            loss_total = 0
            for x, y in loader:
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite classifier loss")
                loss.backward()
                optimizer.step()
                loss_total += loss.item() * len(x)
            model.eval()
            _, pred, _ = extract(model, val_x)
            accuracy = (pred == val_y).float().mean().item()
            record = {"epoch": epoch, "loss": loss_total / len(train_x), "validation_accuracy": accuracy,
                      "seconds": time.perf_counter() - epoch_start}
            training.append(record)
            if accuracy > best_accuracy:
                best_accuracy, best_epoch = accuracy, epoch
                torch.save(model.state_dict(), run / "extractor.pth")
            print(f"Extractor epoch {epoch}: validation accuracy {accuracy:.4%}", flush=True)
        model.load_state_dict(torch.load(run / "extractor.pth", map_location="cpu", weights_only=True))
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        source_hashes = {}
        for name in ["local_metrics.py", "sampling.py", "experiment_utils.py", "scripts/evaluate_local.py"]:
            destination = run / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
            source_hashes[name] = digest(destination)
        reference_features, reference_pred, _ = extract(model, reference_x)
        control_features, control_pred, _ = extract(model, control_x)
        torch.save(reference_features, run / "reference-features.pt")
        real_radii = knn_radii(reference_features, k=3, chunk_size=args.chunk_size)
        summary = {
            "protocol": "local-MNIST64-k3-v1", "official": False,
            "started_at_utc": run_state["started_at_utc"],
            "input_manifest": str(args.manifest.resolve()) if args.manifest else None,
            "input_sweep": str(args.sweep.resolve()) if args.sweep else None,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": source_hashes, "seed": args.seed,
            "extractor": {"architecture": str(model), "feature_dimension": 64,
                          "checkpoint_sha256": digest(run / "extractor.pth"), "training": training,
                          "selected_epoch": best_epoch, "validation_accuracy": best_accuracy,
                          "reference_accuracy": (reference_pred == reference_y).float().mean().item(),
                          "control_accuracy": (control_pred == control_y).float().mean().item()},
            "data": {"source": "MNIST train split only; MNIST test unused", "split_sizes": {k:len(v) for k,v in split.items()},
                     "split_sha256": digest(run / "split-indices.npz"),
                     "reference_not_seen_by_classifier": True, "reference_seen_by_GAN": True,
                     "preprocessing": "PNG grayscale uint8 / 255, then (x-0.5)/0.5; same normalization for real data",
                     "raw_sha256": {name:digest(Path(dataset.raw_folder)/name) for name in
                                    ["train-images-idx3-ubyte", "train-labels-idx1-ubyte"]}},
            "metric": {"distance": "FD-MNIST: Frechet Gaussian distance of raw 64-D classifier ReLU features; not Inception FID",
                       "precision_recall": "k-NN manifold membership, Euclidean metric, k=3, exact blockwise float64",
                       "n_real": 10000, "n_generated": 10000, "k":3, "chunk_size": args.chunk_size},
            "environment": {"torch": torch.__version__, "numpy": np.__version__, "device":"cpu", "threads":args.threads},
            "results": [], "controls": {},
        }
        def calculate(features):
            return {"FD_MNIST": frechet_distance(reference_features.numpy(), features.numpy()),
                    **precision_recall(reference_features, features, k=3, chunk_size=args.chunk_size, real_radii=real_radii)}
        for name, features in [("real_vs_real", control_features),
                               ("black_images", extract(model, torch.full((10000,1,28,28), -1.0))[0])]:
            summary["controls"][name] = calculate(features)
            print(f"Control {name}: {summary['controls'][name]}", flush=True)
        write_json(run / "summary.json", summary)
        for condition in conditions:
            condition_start = time.perf_counter()
            source = Path(condition["samples"])
            images, image_hash = read_pngs(source)
            features, predictions, confidence = extract(model, images)
            record = {**condition,
                      **calculate(features), "image_bytes_sha256":image_hash,
                      "predicted_class_counts":torch.bincount(predictions,minlength=10).tolist(),
                      "mean_classifier_confidence":confidence.mean().item(),
                      "evaluation_seconds":time.perf_counter()-condition_start}
            summary["results"].append(record)
            write_json(run / "summary.json", summary)
            print(f"{condition['run_id']}: FD-MNIST={record['FD_MNIST']:.4f}, "
                  f"precision={record['precision']:.4%}, recall={record['recall']:.4%}", flush=True)
        summary["status"] = "completed"
        summary["total_seconds"] = time.perf_counter() - start
        write_json(run / "summary.json", summary)


if __name__ == "__main__":
    main()
