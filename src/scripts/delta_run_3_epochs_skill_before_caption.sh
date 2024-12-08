#!/bin/bash
#SBATCH --job-name=delta_run_3_epochs_skill_before_caption
#SBATCH --output=delta_run_3_epochs_skill_before_caption.out
#SBATCH --error=delta_run_3_epochs_skill_before_caption.err
#SBATCH --partition=shire-general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12  # Adjusted to match the number of workers in your Python script
#SBATCH --gres=gpu:A100_80GB:1  # Requesting one A100 80GB GPU
#SBATCH --mem=16G  # Memory allocation as per your interactive command
#SBATCH --time=6:00:00  # Adjusted to match your interactive command

# Load necessary modules or activate environments
source /home/mkapadni/.bashrc  # Ensure this points to the correct file if different
conda activate gpu_env

# Navigate to the project directory
cd /home/mkapadni/work/anlp_project/src

# Create the save directory if it doesn't exist and run the Python script
dataset="vizwiz"
annotation="/home/mkapadni/work/anlp_project/data"
base_dir="/home/mkapadni/work/anlp_project/data/images/"

version="v1_delta_new_prompt_skill_before_caption"
savepath="./save/$dataset/$version"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

python -u creating_mtl_baseline_with_skill_before_caption.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --image_dir ${base_dir} \
    --batch_size 8 \
    --val_batch_size 16 \
    --freeze_vm False \
    --vis_use_lora False \
    --llm_use_lora True\
    --savedmodel_path ${savepath} \
    --max_length 100 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 16 \
    --devices 1 \
    --max_epochs 3 \
    --limit_val_batches 0.5 \
    --val_check_interval 0.5 \
    --num_sanity_val_steps 2 \
    2>&1 |tee -a ${savepath}/log.txt