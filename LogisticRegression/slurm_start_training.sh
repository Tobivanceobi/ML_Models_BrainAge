#!/bin/bash
#SBATCH --job-name=LogR
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=1000
#SBATCH --time=10:00:00
#SBATCH --array=0-5
#SBATCH --mail-type=FAIL
#SBATCH --output out/output_LogR_%a.txt
#SBATCH --error err/error_LogR_%a.txt

# Remove previous results
# rm err/*; rm out/*; rm -r runs/*;

source $HOME/miniconda/etc/profile.d/conda.sh
conda init bash
conda activate pytorch
python3 $HOME/tobias_ettling/ML_Models_BrainAge/LogisticRegression/training.py $SLURM_ARRAY_TASK_ID

conda deactivate