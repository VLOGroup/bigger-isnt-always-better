#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=icg

source activate diff_mri
cd /home/lg/score-MRI/diffusion-generalization-mri


export CUDA_VISIBLE_DEVICES=1

for lam in 0.0 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 0.02 0.03 0.04 0.05 1.0
do
    python evaluate_lambda.py \
        --model celeba_1 \
        --N 250 \
        --mask_type gaussian1d \
        --acc_factor 4 \
        --center_fraction 0.04 \
        --lam $lam

    python evaluate_lambda.py \
        --model celeba_2 \
        --N 250 \
        --mask_type gaussian1d \
        --acc_factor 4 \
        --center_fraction 0.04 \
        --lam $lam

    python evaluate_lambda.py \
        --model celeba_3 \
        --N 250 \
        --mask_type gaussian1d \
        --acc_factor 4 \
        --center_fraction 0.04 \
        --lam $lam

    python evaluate_lambda.py \
        --model celeba_4 \
        --N 250 \
        --mask_type gaussian1d \
        --acc_factor 4 \
        --center_fraction 0.04 \
        --lam $lam

    python evaluate_lambda.py \
        --model celeba_4_attention \
        --N 250 \
        --mask_type gaussian1d \
        --acc_factor 4 \
        --center_fraction 0.04 \
        --lam $lam
done
