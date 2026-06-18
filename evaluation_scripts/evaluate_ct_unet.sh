#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=ivc
#SBATCH --ntasks=1
#SBATCH --nodelist=nvcluster-node1

source activate diff_mri
export CUDA_VISIBLE_DEVICES=3

for set in thorax head
do
    python evaluation_main_control.py \
        --model_config=configs/models/unet_ct.yaml \
        --data_config=configs/datasets/ct_${set}.yaml \
        --evaluation_config=configs/evaluation/fanbeam_60.yaml \
        --sample_config=configs/samplers/dps_sampler.yaml \
        --savedir=/mount/data/glaszner/ijcv_update/unet_ct_${set}_fb60
done
