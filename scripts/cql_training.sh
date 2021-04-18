#!/bin/bash

#SBATCH --account=def-adurand                                                       # Account with resources
#SBATCH --gres=gpu:t4:1                                                             # Number of GPUs
#SBATCH --cpus-per-task=4                                                           # Number of CPUs
#SBATCH --mem=20G                                                                   # memory (per node)
#SBATCH --time=2-00:00                                                              # time (DD-HH:MM)
#SBATCH --mail-user=mathieu.godbout.3@ulaval.ca                                     # Where to email
#SBATCH --mail-type=FAIL                                                            # Email when a job fails
#SBATCH --output=/project/def-adurand/magod/opti_combi/slurm_outputs/%A_%a.out      # Default write output on scratch, to jobID_arrayID.out file
#SBATCH --array=0-47                                                                # Launch 48 jobs

source /home/magod/venvs/last_try/bin/activate
export PYTHONPATH=/home/magod/git/ift7020-offlineRL/:$PYTHONPATH

cp /scratch/magod/opti_combi/datasets.tar.gz $SLURM_TMPDIR
tar -xzf $SLURM_TMPDIR/datasets.tar.gz -C $SLURM_TMPDIR/

module load cuda
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/11.0.2/lib64:$LD_LIBRARY_PATH

python core/rl.py --working_path $SLURM_TMPDIR --saving_path /scratch/magod/opti_combi/results --job_index $SLURM_ARRAY_TASK_ID