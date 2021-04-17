#!/bin/bash

#SBATCH --account=rrg-corbeilj-ac                                                   # Account with resources
#SBATCH --cpus-per-task=4                                                           # Number of CPUs
#SBATCH --mem=20G                                                                   # memory (per node)
#SBATCH --time=0-12:00                                                              # time (DD-HH:MM)
#SBATCH --mail-user=mathieu.godbout.3@ulaval.ca                                     # Where to email
#SBATCH --mail-type=FAIL                                                            # Email when a job fails
#SBATCH --output=/project/def-adurand/magod/opti_combi/slurm_outputs/%A_%a.out      # Default write output on scratch, to jobID_arrayID.out file
#SBATCH --array=0-3                                                                 # Launch 4 jobs

source /home/magod/venvs/last_try/bin/activate
export PYTHONPATH=/home/magod/git/ift7020-offlineRL/:$PYTHONPATH

cp /scratch/magod/opti_combi/datasets.tar.gz $SLURM_TMPDIR
tar -xzf $SLURM_TMPDIR/datasets.tar.gz -C $SLURM_TMPDIR/

module load cuda

python -u prepare_rl_training.py --train_bc 1 --working_path $SLURM_TMPDIR --saving_path /scratch/magod/opti_combi/results --job_index $SLURM_ARRAY_TASK_ID