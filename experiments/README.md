# Experiment evidence policy

Each execution writes complete outputs under **`runs/<run-id>/`**, ignored by
Git: samples, intermediate checkpoints, per-condition metadata, grids, logs,
source snapshots, environment exports and feature caches. Do not force-add
complete run directories or copy them into `experiments/`.

Version only reviewed, compact evidence:

- One evolving Markdown report per workstream.
- One consolidated result table and one `<workstream>-ledger.jsonl` per workstream.
  The shared `ledger.jsonl` records common baseline runs.
- One provenance manifest per study, covering configurations, source/checkpoint
  hashes, stage history and selection decisions.
- A small curated figure set under `figures/`.
- Selected shared/final G and D checkpoints under `checkpoints/`, saved once.

The frozen [local evaluator](local-metrics.md) is a reusable exception: its
weights, split indices and protocol metadata are shared once under
`local-metrics-mnist64-k3/`. Source snapshots and feature caches stay local.
Existing shared-baseline fixtures are retained for the team and regression
tests; they are not a template for storing every subsequent execution.

The `.gitignore` blocks new per-execution directories under `experiments/`,
except `figures/` and the three approved evaluator files. New artifact exceptions
should be deliberate and reusable. No runner publishes or commits its outputs.

A ledger may point to local metadata under `runs/`, but must also link to compact
portable provenance and results. Repeated controls should remain identifiable
and must not be counted as independent replications. Store official scores
separately from local metrics; unavailable official scores remain null.
