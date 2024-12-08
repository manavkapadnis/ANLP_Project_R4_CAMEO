#!/bin/bash
#SBATCH --job-name=caption_skills_vqa
#SBATCH --output=caption_skills_vqa.out
#SBATCH --error=caption_skills_vqa.err
#SBATCH --partition=shire-general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12  # Adjusted to match the number of workers in your Python script
#SBATCH --mem=16G  # Memory allocation as per your interactive command
#SBATCH --time=12:00:00  # Adjusted to match your interactive command

# Load necessary modules or activate environments
source /home/mkapadni/.bashrc
conda activate gpu_env

# Navigate to the project directory
cd /home/mkapadni/work/anlp_project/src

python organizing_preds_caption_skill_vqa.py