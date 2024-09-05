#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=icg

source activate diff_mri
cd /home/lg/score-MRI/diffusion-generalization-mri

for model in tv
do
    python evaluate_control.py \
        --model $model \
        --ct t \
        --problem few_view \
        --num_view 60 

done