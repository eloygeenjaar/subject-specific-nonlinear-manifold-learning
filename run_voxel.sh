#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --mem=128g
#SBATCH -p qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH -t 4000
#SBATCH -J voxel
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
python voxel_classification.py --data_config configs/data/sherlock_aud.json
python voxel_classification.py --data_config configs/data/sherlock_ev.json
python voxel_classification.py --data_config configs/data/sherlock_pmc.json
python voxel_classification.py --data_config configs/data/forrest_dmn.json
python voxel_classification.py --data_config configs/data/forrest_aud.json
python voxel_classification.py --data_config configs/data/forrest_cc.json
python voxel_classification.py --data_config configs/data/sherlock_full.json
python voxel_classification.py --data_config configs/data/forrest_full.json


sleep 30s
