#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --mem=32g
#SBATCH -p qTRDGPU
#SBATCH --gres=gpu:RTX:1
#SBATCH -t 4000
#SBATCH -J sg
#SBATCH -e /data/users1/egeenjaar/subject-learning/jobs/error-sg-%A.out
#SBATCH -o /data/users1/egeenjaar/subject-learning/jobs/out-sg-%A.out
#SBATCH -A trends53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=egeenjaar@gsu.edu
#SBATCH --oversubscribe
#SBATCH --exclude=arctrdagn007

sleep 10s

export PATH=/data/users1/egeenjaar/miniconda3/bin:$PATH
source /data/users1/egeenjaar/miniconda3/etc/profile.d/conda.sh
conda activate /data/users1/egeenjaar/miniconda3/envs/pytorch
python subject_generalization.py --data_config configs/data/fbirn_subject.json --model_config configs/models/decomposed_vae.json --run_config configs/runs/tvae.json -sn 0
python subject_generalization.py --data_config configs/data/fbirn_subject.json --model_config configs/models/decomposed_vae.json --run_config configs/runs/tvae.json -sn 1
python subject_generalization.py --data_config configs/data/fbirn_subject.json --model_config configs/models/decomposed_vae.json --run_config configs/runs/tvae.json -sn 2
python subject_generalization.py --data_config configs/data/fbirn_subject.json --model_config configs/models/decomposed_vae.json --run_config configs/runs/tvae.json -sn 3

sleep 30s
