#!/bin/bash
#SBATCH --job-name=XGBoost
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5000
#SBATCH --time=10:00:00
#SBATCH --output out/output_XGBoost.txt
#SBATCH --error err/error_XGBoost.txt

# Remove previous results
# rm err/*; rm out/*; rm -r runs/*;
rm /scratch/modelrep/sadiya/students/tobias/data/jobs/*
source $HOME/miniconda/etc/profile.d/conda.sh
conda init bash
conda activate pytorch
python3 $HOME/tobias_ettling/ML_Models_BrainAge/XGBoost/training.py

conda deactivate