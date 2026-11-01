"""Results table and figures for the DiffAugment study from evaluate_frozen summaries.

Usage: python scripts/diffaug_figures.py SUMMARY.json [SUMMARY.json ...]
Writes experiments/diffaugment-results.csv, experiments/figures/diffaugment-metrics.png,
experiments/figures/diffaugment-classes.png. Needs matplotlib (figures only; not a
dependency of training or generate.py).
"""
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["baseline", "noise0.1", "da-translation", "da-translation-cutout",
         "da-translation-cutout-color", "ada-translation-cutout"]
LABEL = {"baseline": "baseline", "noise0.1": "instance\nnoise", "da-translation": "DA\ntransl.",
         "da-translation-cutout": "DA transl.\n+cutout", "da-translation-cutout-color": "DA t.+cut.\n+color",
         "ada-translation-cutout": "ADA t.\n+cutout"}

(ROOT / "experiments/figures").mkdir(parents=True, exist_ok=True)
rows = []
for path in sys.argv[1:]:
    summary = json.loads(Path(path).read_text())
    for r in summary["results"]:
        seed = r["config"].get("training_seed")
        rows.append({"run_id": re.sub(r"-seed\d+$", "", r["run_id"]), "training_seed": seed, "FD_MNIST": r["FD_MNIST"],
                     "precision": r["precision"], "recall": r["recall"],
                     "classes": r["predicted_class_counts"], "checkpoint_sha256": r["checkpoint_sha256"]})
rows.sort(key=lambda r: (ORDER.index(r["run_id"]), r["training_seed"]))

with open(ROOT / "experiments/diffaugment-results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run_id", "training_seed", "FD_MNIST", "precision", "recall"]
               + [f"pred_class_{k}" for k in range(10)] + ["checkpoint_sha256"])
    for r in rows:
        w.writerow([r["run_id"], r["training_seed"], round(r["FD_MNIST"], 4), r["precision"], r["recall"]]
                   + r["classes"] + [r["checkpoint_sha256"]])

conds = [c for c in ORDER if any(r["run_id"] == c for r in rows)]
colors = ["#8E9CA6", "#B9A06B", "#426981", "#2E8B57", "#5FA37A", "#B5651D"]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, key, title in zip(axes, ["FD_MNIST", "precision", "recall"],
                          ["FD-MNIST (lower is better)", "Precision (higher is better)", "Recall (higher is better)"]):
    for x, c in enumerate(conds):
        vals = [r[key] * (100 if key != "FD_MNIST" else 1) for r in rows if r["run_id"] == c]
        mean = sum(vals) / len(vals)
        ax.bar(x, mean, color=colors[ORDER.index(c)], width=0.7)
        if len(vals) > 1:
            ax.scatter([x] * len(vals), vals, color="black", s=12, zorder=3)
        top = max([mean] + vals)
        ax.text(x, top * 1.02, f"{mean:.0f}" if key == "FD_MNIST" else f"{mean:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([LABEL[c] for c in conds], fontsize=7.5)
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("DiffAugment study — 10 epochs, frozen local-MNIST64-k3 evaluator"
             + (" (bars: mean over training seeds; dots: seeds)" if any(
                 sum(r["run_id"] == c for r in rows) > 1 for c in conds) else " (one training seed)"),
             fontsize=10)
fig.tight_layout()
fig.savefig(ROOT / "experiments/figures/diffaugment-metrics.png", dpi=160)

fig, axes = plt.subplots(1, len(conds), figsize=(2.3 * len(conds), 2.6), sharey=True)
for ax, c in zip(axes, conds):
    counts = [r["classes"] for r in rows if r["run_id"] == c and r["training_seed"] == 0][0]
    share = [100 * k / sum(counts) for k in counts]
    ax.bar(range(10), share, color=colors[ORDER.index(c)])
    ax.axhline(10, color="black", lw=0.8, ls="--")
    ax.set_xticks(range(10))
    ax.tick_params(labelsize=7)
    ax.set_title(LABEL[c].replace("\n", " "), fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("% of 10,000 samples", fontsize=8)
fig.suptitle("Digit predicted by the frozen classifier (seed 0); dashed line = 10% (uniform)", fontsize=9)
fig.tight_layout()
fig.savefig(ROOT / "experiments/figures/diffaugment-classes.png", dpi=160)
print("wrote CSV and 2 figures for", len(rows), "rows")


# Simple 3-method figure for slides: baseline vs DiffAugment (translation+cutout) vs ADA.
MAIN = [("baseline", "Baseline"), ("da-translation-cutout", "DiffAugment\n(shift + square)"),
        ("ada-translation-cutout", "ADA\n(adaptive)")]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, key, title in zip(axes, ["FD_MNIST", "precision", "recall"],
                          ["FD (lower = better)", "Precision = quality (%)", "Recall = diversity (%)"]):
    for x, (c, _) in enumerate(MAIN):
        vals = [r[key] * (100 if key != "FD_MNIST" else 1) for r in rows if r["run_id"] == c]
        mean = sum(vals) / len(vals)
        ax.bar(x, mean, color=colors[ORDER.index(c)], width=0.6)
        ax.scatter([x] * len(vals), vals, color="black", s=14, zorder=3)
        ax.text(x, max([mean] + vals) * 1.03, f"{mean:.0f}" if key == "FD_MNIST" else f"{mean:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(MAIN)))
    ax.set_xticklabels([label for _, label in MAIN], fontsize=8.5)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    ax.spines[["top", "right"]].set_visible(False)
collapsed = {c: sum(1 for r in rows if r["run_id"] == c and r["precision"] == 0) for c, _ in MAIN}
tries = {c: sum(1 for r in rows if r["run_id"] == c) for c, _ in MAIN}
fig.suptitle("Bars: average of " + "/".join(str(tries[c]) for c, _ in MAIN) + " training runs; dots: each run.   "
             "Collapsed runs: " + ", ".join(f"{label.split(chr(10))[0]} {collapsed[c]}/{tries[c]}" for c, label in MAIN),
             fontsize=9.5)
fig.tight_layout()
fig.savefig(ROOT / "experiments/figures/diffaugment-main.png", dpi=160)
print("main figure written")
