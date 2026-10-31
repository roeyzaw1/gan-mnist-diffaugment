#!/bin/bash
# DiffAugment / instance-noise study on Juliet: one GPU job per condition (Slurm array),
# then one evaluation job with the frozen local-MNIST64-k3-v1 protocol once all succeed.
# Usage (from the repository root, with venv/ created):  bash scripts/diffaug_study.sh [seed]
SEED=${1:-0}
EPOCHS=10
DATA=/projects/m25146/data
STUDY=runs/diffaug-seed${SEED}-e${EPOCHS}
mkdir -p outputs runs

ARRAY=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH -p mesonet
#SBATCH --account=m25146
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --array=0-5
#SBATCH --job-name=diffaug
#SBATCH --output=outputs/%x_%A_%a.out
source venv/bin/activate
NAMES=(baseline noise0.1 da-translation da-translation-cutout da-translation-cutout-color ada-translation-cutout)
FLAGS=("" "--instance-noise 0.1" "--augment translation" "--augment translation,cutout" "--augment translation,cutout,color" "--augment translation,cutout --ada-target 0.6")
i=\$SLURM_ARRAY_TASK_ID
python scripts/baseline.py --run-dir ${STUDY}/\${NAMES[\$i]} --data-path ${DATA} \
    --epochs ${EPOCHS} --seed ${SEED} --generation-seed 1 --generation-batch-size 256 \${FLAGS[\$i]}
EOT
)
echo "training array: ${ARRAY}"

sbatch --dependency=afterok:${ARRAY} <<EOT
#!/bin/bash
#SBATCH -p mesonet
#SBATCH --account=m25146
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=diffaug-eval
#SBATCH --output=outputs/%x_%j.out
source venv/bin/activate
python scripts/diffaug_manifest.py ${STUDY}
python scripts/evaluate_frozen.py --manifest ${STUDY}/manifest.json \
    --run-dir ${STUDY}-local-metrics --data-path ${DATA}
EOT
