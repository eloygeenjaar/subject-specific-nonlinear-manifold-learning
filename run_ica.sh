#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=32g
#SBATCH -p qTRDHM
#SBATCH -t 240
#SBATCH -J preproc
#SBATCH -e jobs/error-preproc-%A.err
#SBATCH -o jobs/out-preproc-%A.out
#SBATCH -A trends53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=egeenjaar@gsu.edu
#SBATCH --oversubscribe

sleep 10s

export PATH=/data/users1/egeenjaar/miniconda3/bin:$PATH
source /data/users1/egeenjaar/miniconda3/etc/profile.d/conda.sh
conda activate /data/users1/egeenjaar/miniconda3/envs/pytorch
python run_ica_subjects.py

sleep 30s