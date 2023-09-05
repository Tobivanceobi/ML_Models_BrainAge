#!/bin/bash
#SBATCH --job-name=KNN
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=1000
#SBATCH --time=10:00:00
#SBATCH --array=0-0
#SBATCH --output out/output_KNN.txt
#SBATCH --error err/error_KNN.txt

# Remove previous results
# rm err/*; rm out/*; rm -r runs/*;

source $HOME/miniconda/etc/profile.d/conda.sh
conda init bash
conda activate pytorch
python3 $HOME/tobias_ettling/ML_Models_BrainAge/KNN/training.py $SLURM_ARRAY_TASK_ID

conda deactivate