#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=icg

source activate diff_mri
cd /home/lg/score-MRI/diffusion-generalization-mri

for model in fastmri_knee_4_attention fastmri_knee_4 fastmri_knee_3 fastmri_knee_2 fastmri_knee_1 celeba_4_attention celeba_4 celeba_3 celeba_2 celeba_1
do
    python evaluate_jalal.py \
        --model $model \
        --N 500 \
        --brain t \
        --mask_type radial
    
    python evaluate_jalal.py \
        --model $model \
        --N 500 \
        --mask_type gaussian1d \
        --center_fraction 0.04 \
        --brain t \
        --acc_factor 4
    
    python evaluate_jalal.py \
        --model $model \
        --N 500 \
        --mask_type gaussian1d \
        --center_fraction 0.04 \
        --brain t \
        --acc_factor 8

    python evaluate_jalal.py \
        --model $model \
        --N 500 \
        --mask_type gaussian2d \
        --brain t \
        --acc_factor 4
        
    python evaluate_jalal.py \
        --model $model \
        --N 500 \
        --mask_type poisson \
        --brain t \
        --acc_factor 15
done
