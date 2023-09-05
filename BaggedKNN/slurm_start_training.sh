#!/bin/bash
#SBATCH --job-name=BKNN
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=3000
#SBATCH --tmp=20000
#SBATCH --array=0-0
#SBATCH --time=10:00:00
#SBATCH --output out/output_BKNN.txt
#SBATCH --error err/error_BKNN.txt

# Remove previous results
# rm err/*; rm out/*; rm -r runs/*;

source $HOME/miniconda/etc/profile.d/conda.sh
conda init bash
conda activate pytorch
python3 $HOME/tobias_ettling/ML_Models_BrainAge/BaggedKNN/training.py $SLURM_ARRAY_TASK_ID

conda deactivate