#!/bin/bash
#SBATCH --job-name=KNN_Shap
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=3000
#SBATCH --time=10:00:00
#SBATCH --tmp=20000
#SBATCH --output out/output_KNN_Shap.txt
#SBATCH --error err/error_KNN_Shap.txt

# Remove previous results
# rm err/*; rm out/*; rm -r runs/*;
source $HOME/miniconda/etc/profile.d/conda.sh
conda init bash
conda activate pytorch
python3 $HOME/tobias_ettling/ML_Models_BrainAge/KNN/shapValues.py

conda deactivate