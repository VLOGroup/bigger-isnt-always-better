#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1

source activate diff_mri
cd /home/lg/score-MRI/diffusion-generalization-mri

for model in celeba_4_attention celeba_4 celeba_3 celeba_2 celeba_1
do
    python evaluate_jalal.py \
        --model $model \
        --N 500 \
        --mask_type radial
    
    python evaluate_jalal.py \
        --model $model \
        --N 500 \
        --mask_type gaussian1d \
        --center_fraction 0.04 \
        --acc_factor 4
    
    python evaluate_jalal.py \
        --model $model \
        --N 500 \
        --mask_type gaussian1d \
        --center_fraction 0.04 \
        --acc_factor 8

    python evaluate_jalal.py \
        --model $model \
        --N 500 \
        --mask_type gaussian2d \
        --acc_factor 4
        
    python evaluate_jalal.py \
        --model $model \
        --N 500 \
        --mask_type poisson \
        --acc_factor 15
done
