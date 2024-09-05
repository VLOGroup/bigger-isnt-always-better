#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=icg

source activate diff_mri
cd /home/lg/score-MRI/diffusion-generalization-mri

for model in ct_4_attention ct_4
do
    python evaluate_ct.py \
        --model $model \
        --N 500 \
        --problem few_view \
        --num_view 30

    # python evaluate_ct.py \
    #     --model $model \
    #     --N 1000 \
    #     --problem few_view \
    #     --num_view 30 \
    #     --head t \
    #     --jalal t
done

