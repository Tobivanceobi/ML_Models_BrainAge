#!/bin/bash
#SBATCH --job-name=Lasso_Shap
#SBATCH --partition=general2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3000
#SBATCH --time=10:00:00
#SBATCH --tmp=20000
#SBATCH --output out/output_Lasso_Shap.txt
#SBATCH --error err/error_Lasso_Shap.txt

# Remove previous results
# rm err/*; rm out/*; rm -r runs/*;
source $HOME/miniconda/etc/profile.d/conda.sh
conda init bash
conda activate pytorch
python3 $HOME/tobias_ettling/ML_Models_BrainAge/LassoRegression/shapValues.py

conda deactivate