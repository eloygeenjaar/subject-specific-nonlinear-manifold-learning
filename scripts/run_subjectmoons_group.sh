#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --mem=128g
#SBATCH -p qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH -t 500
#SBATCH -J decomp
#SBATCH -e /data/users1/egeenjaar/subject-learning/jobs/error-comms-%A.out
#SBATCH -o /data/users1/egeenjaar/subject-learning/jobs/out-comms-%A.out
#SBATCH -A trends53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=egeenjaar@gsu.edu
#SBATCH --oversubscribe
#SBATCH --exclude=arctrdagn007

sleep 10s

export PATH=/data/users1/egeenjaar/miniconda3/bin:$PATH
source /data/users1/egeenjaar/miniconda3/etc/profile.d/conda.sh
conda activate /data/users1/egeenjaar/miniconda3/envs/pytorch
python main.py --data_config configs/data/subject_moons.json --model_config configs/models/group_classifier.json --run_config configs/runs/classification.json -hn $SLURM_ARRAY_TASK_ID

sleep 30s
