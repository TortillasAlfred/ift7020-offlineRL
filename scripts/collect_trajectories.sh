#!/bin/bash

#SBATCH --account=rrg-corbeilj-ac                                                   # Account with resources
#SBATCH --cpus-per-task=1                                                           # Number of CPUs
#SBATCH --mem=5G                                                                    # memory (per node)
#SBATCH --time=0-00:30                                                              # time (DD-HH:MM)
#SBATCH --mail-user=mathieu.godbout.3@ulaval.ca                                     # Where to email
#SBATCH --mail-type=FAIL                                                            # Email when a job fails
#SBATCH --output=/project/def-adurand/magod/opti_combi/slurm_outputs/%A_%a.out      # Default write output on scratch, to jobID_arrayID.out file
#SBATCH --array=0-599                                                               # Launch 600 jobs

source /home/magod/venvs/opti_combi/bin/activate
export PYTHONPATH=/home/magod/git/ift7020-offlineRL/:$PYTHONPATH

python -u prepare_rl_training.py --collect_trajectories 1 --collection_root /scratch/magod/opti_combi/data --job_index $SLURM_ARRAY_TASK_ID