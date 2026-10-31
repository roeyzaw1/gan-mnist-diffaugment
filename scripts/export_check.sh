#!/bin/bash
# Export-convention check on the shared baseline checkpoint (clip vs rescale, sigma 1 and 2).
# Usage (repository root, venv/ created):  bash scripts/export_check.sh
DATA=/projects/m25146/data
OUT=runs/export-convention-check
mkdir -p outputs runs
sbatch <<EOT
#!/bin/bash
#SBATCH -p mesonet
#SBATCH --account=m25146
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=export-check
#SBATCH --output=outputs/%x_%j.out
source venv/bin/activate
python scripts/export_convention_check.py ${OUT}
python scripts/evaluate_frozen.py --manifest ${OUT}/manifest.json \
    --run-dir ${OUT}-local-metrics --data-path ${DATA}
EOT
