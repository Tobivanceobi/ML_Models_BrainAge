#!/bin/bash
#SBATCH --job-name=KRige_shap
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=1000
#SBATCH --time=10:00:00
#SBATCH --output out/output_KRige_shap.txt
#SBATCH --error err/error_KRige_shap.txt

# Remove previous results
# rm err/*; rm out/*; rm -r runs/*;
source $HOME/miniconda/etc/profile.d/conda.sh
conda init bash
conda activate pytorch
python3 $HOME/tobias_ettling/ML_Models_BrainAge/KernalRige/shapValues.py

conda deactivate