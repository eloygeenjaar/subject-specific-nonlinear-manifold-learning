#/bin/bash
sbatch --array=0-95 scripts/run_subjectmoons_group.sh
sbatch --array=0-95 scripts/run_subjectmoons_decomposed.sh
sbatch --array=0-95 scripts/run_subjectmoons_subject.sh
