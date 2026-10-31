# DataLabAssignement2

## Using GPUs with MesoNet

### 1. Access MesoNet
To use the provided GPUs, you must first:
- Create a MesoNet account and request access to the project.
- Once accepted, add your SSH key to MesoNet for secure access. Follow the instructions [here](https://www.mesonet.fr/documentation/user-documentation/code_form/juliet/connexion).

### 2. Set Up Your Environment
On Juliet (MesoNet's cluster), you need to:

1. Create a virtual environment for Python:
   ```bash
   python -m venv venv
   ```

2. Activate the environment:
   ```bash
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Run Training or Generation
- Use the provided shell scripts (`scripts/train.sh` or `scripts/generate.sh`) to launch jobs.
- These scripts will automatically activate the virtual environment (`venv`).
- Outside Juliet, set the `DATA` environment variable to an existing MNIST dataset directory, or leave it unset to download MNIST into `data/`. The starter `train.py` has no `--data_path` argument.

### 4. Hardware Compatibility
The code is compatible with:
- **NVIDIA GPUs (CUDA)**
- **Apple Silicon (MPS, for M1/M2/M3/M4 chips)**
- **CPU-only mode** (not recommended for performance)

We recommend using CUDA (for NVIDIA GPUs) or MPS (for Apple Silicon).

### 5. Slurm Resources
- If you need help with Slurm (MesoNet's job scheduler), there will be a Slurm lecture on 29/10/2025 or refer to the official documentation: [Slurm Documentation](https://slurm.schedmd.com/documentation.html).

### 6. Example Commands
To submit a job to MesoNet:
```bash
# Make the script executable
chmod +x scripts/train.sh

# Submit the job to Slurm
scripts/train.sh
```
## train.py
This script performs adversarial training for a GAN. Before running it, ensure the `DATA` variable in `scripts/train.sh` points to your dataset directory. If left unchanged, the script will create a default `data` folder in the repository root. If you use Juliet, no changes are required—the default path is already configured. You can adjust the learning rate, number of epochs, and batch size directly in `scripts/train.sh`.

## generate.py
Use the file *generate.py* to generate 10000 samples of MNIST in the folder samples. You can launch this script on juliet with `scripts/generate.sh`

Example:
  > python3 generate.py --batch_size 64

or 

```bash
# Make the script executable
chmod +x scripts/generate.sh

# Submit the job to Slurm
scripts/generate.sh
```

## requirements.txt
Among the good pratice of datascience, we encourage you to use conda or virtualenv to create python environment. 
To test your code on our platform, you are required to update the *requirements.txt*, with the different librairies you might use. 
When your code will be test, we will execute: 
  > pip install -r requirements.txt


## Checkpoints
Push the minimal amount of models in the folder *checkpoints*.

## Reproducible local baseline

Create an environment and install the starter dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/baseline.py --run-dir runs/baseline-seed0-e1 \
  --epochs 1 --seed 0 --generation-seed 1 --generation-batch-size 256
```

Run from the repository root. MNIST is downloaded to `data/` if needed; use
`--data-path /path/to/dataset` to select another location. A run directory must
be new: existing runs are never overwritten. No external experiment service or
additional dependency is required.

This is a minimal baseline of **one full training epoch**, not evidence of
convergence or a quality benchmark. The starter CLI defaults to 100 epochs and
the provided Slurm script uses 10. The recorded initial shared reference uses
**10 epochs**, matching that Slurm budget. Reproduce it with a new run ID:

```bash
python scripts/baseline.py --run-dir runs/baseline-reproduction-e10 \
  --epochs 10 --seed 0 --generation-seed 1 --generation-batch-size 256
```

See [baseline results and limitations](experiments/baseline.md), the
[experiment ledger](experiments/ledger.jsonl), and the
[recorded package versions](experiments/baseline-environment.txt). The snapshots
under each run preserve the exact source state at launch, including uncommitted
code. Keep the training budget explicit when comparing subsequent interventions.

The runner uses the original `Generator`, `Discriminator`, `D_train`, and
`G_train`, from fresh random initialization. It keeps batch size 64, Gaussian
100-dimensional noise, equal-prior BCE training, and Adam with learning rate
0.0002 and default betas (0.9, 0.999) for both models. It does not load the
supplied `checkpoints/G.pth`. Local runtime settings are explicit: deterministic
algorithms, two CPU threads, no DataLoader subprocesses, no pinned memory, and
automatic CUDA/MPS/CPU selection. Unlike `train.py`, the runner saves the final
checkpoint even for runs shorter than 10 epochs. Cross-device bitwise equality
is not guaranteed.

Each run contains:

- `checkpoints/G.pth`, `checkpoints/D.pth`, and checkpoint metadata.
- `metadata.json`: base commit, dirty working-tree status, source hashes, seeds,
  hardware, configuration, dataset hashes, update counts, timings, and validation.
- `source/`: the exact source files used plus the tracked working-tree diff.
- `environment.txt` and `torch-config.txt`: installed versions and PyTorch build.
- `training.jsonl`: sample-weighted training losses and epoch timings.
- `samples/0.png` through `samples/9999.png`, `sample-grid.png` (first 64 exported
  images, without selection), and `generation.log`.

Generation executes the unmodified `generate.py` from the isolated run directory
with a separate RNG seed and batch size 256. Its measured time includes process
startup, checkpoint loading, inference, and writing all 10,000 PNGs; subsequent
validation and grid creation are excluded. The final batch generates 240 unused
candidates at this batch size, as in the starter. The exporter clips negative
generator outputs when saving rather than rescaling `[-1,1]` to `[0,1]`. This
convention is preserved for the baseline and must be checked against official
evaluation instructions before introducing a different export protocol.

The runner validates every output's name, count, dimensions, and PNG/RGB format,
checks that G was trained, and verifies that generation preserved the saved
checkpoints and the original repository checkpoint. Successful runs append to
`experiments/ledger.jsonl`. FID, precision, and recall remain `null` with status
`not yet evaluated` until an official evaluator is available. Training losses
are not quality metrics.

`runs/`, downloaded datasets, samples, environments, and local `codex.md` are
ignored by Git. The shared 10-epoch reference G and D are versioned under
`checkpoints/baseline-seed0-e10/`; its metadata, logs, grid and original source
snapshot are under `experiments/baseline-seed0-e10/`. Compact evidence from the
pilot is also versioned, while intermediate checkpoints and all 10,000-image
batches remain local. `checkpoints/G.pth` remains the original starter checkpoint;
the root `generate.py` invocation still loads that file. The baseline runner
loads its own trained checkpoint from the isolated run directory.
The existing checkpoint is preserved during baseline runs. The cluster scripts
still expect an environment named `venv/`; the commands above use `.venv/` for
local execution.

