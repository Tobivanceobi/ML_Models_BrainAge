#!/bin/bash
#SBATCH --job-name=MLP
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=3000
#SBATCH --array=0-2
#SBATCH --time=20:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output out/output_MLP_%a.txt
#SBATCH --error err/error_MLP_%a.txt

# Remove previous results
# rm err/*; rm out/*; rm -r runs/*;

source $HOME/miniconda/etc/profile.d/conda.sh
conda init bash
conda activate pytorch
python3 $HOME/tobias_ettling/ML_Models_BrainAge/MLP/training.py $SLURM_ARRAY_TASK_ID

conda deactivate