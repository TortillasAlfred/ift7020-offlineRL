#!/bin/bash

#SBATCH --account=rrg-corbeilj-ac                                                   # Account with resources
#SBATCH --cpus-per-task=4                                                           # Number of CPUs
#SBATCH --mem=20G                                                                   # memory (per node)
#SBATCH --time=1-00:00                                                              # time (DD-HH:MM)
#SBATCH --mail-user=mathieu.godbout.3@ulaval.ca                                     # Where to email
#SBATCH --mail-type=FAIL                                                            # Email when a job fails
#SBATCH --output=/project/def-adurand/magod/opti_combi/slurm_outputs/%A_%a.out      # Default write output on scratch, to jobID_arrayID.out file
#SBATCH --array=13                                                                  # Launch 13 jobs

source /home/magod/venvs/last_try/bin/activate
export PYTHONPATH=/home/magod/git/ift7020-offlineRL/:$PYTHONPATH

module load cuda
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/11.0.2/lib64:$LD_LIBRARY_PATH

python core/rl_testing_more.py --src_path /scratch/magod/opti_combi/test_instances --saving_path /scratch/magod/opti_combi/results --job_index $SLURM_ARRAY_TASK_ID