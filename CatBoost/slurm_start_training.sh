#!/bin/bash
#SBATCH --job-name=CatB
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=1000
#SBATCH --time=05:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output out/output_CatB.txt
#SBATCH --error err/error_CatB.txt

# Remove previous results
# rm err/*; rm out/*; rm -r runs/*;

source $HOME/miniconda/etc/profile.d/conda.sh
conda init bash
conda activate pytorch
python3 $HOME/tobias_ettling/ML_Models_BrainAge/CatBoost/training.py

conda deactivate