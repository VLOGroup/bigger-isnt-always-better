#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --partition=l40

source activate diff_mri

for set in head
do
    python evaluation_main_control.py \
        --model_config=configs/models/tv.yaml \
        --data_config=configs/datasets/ct_${set}.yaml \
        --evaluation_config=configs/evaluation/sparse_view_60.yaml \
        --sample_config=configs/samplers/dps_sampler.yaml \
        --savedir=/srv/local/lg/ijcv_update/tv_ct_${set}_sv60
done
