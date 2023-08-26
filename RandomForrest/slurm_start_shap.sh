#!/bin/bash
#SBATCH --job-name=RF_shap
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=1000
#SBATCH --time=15:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output out/output_RF_shap.txt
#SBATCH --error err/error_RF_shap.txt

# Remove previous results
# rm err/*; rm out/*; rm -r runs/*;
source $HOME/miniconda/etc/profile.d/conda.sh
conda init bash
conda activate pytorch
python3 $HOME/tobias_ettling/ML_Models_BrainAge/RandomForrest/shapValues.py

conda deactivate