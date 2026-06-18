#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=ivc
#SBATCH --ntasks=1
#SBATCH --nodelist=nvcluster-node1

source activate diff_mri
export CUDA_VISIBLE_DEVICES=3

for set in corpd corpdfs brain
do
    python evaluation_main_control.py \
        --model_config=configs/models/tv.yaml \
        --data_config=configs/datasets/fast_mri_${set}.yaml \
        --evaluation_config=configs/evaluation/gaussian_2d_4.yaml \
        --sample_config=configs/samplers/dps_sampler.yaml \
        --savedir=/mount/data/glaszner/ijcv_update/tv_fastmri_${set}_g2d4
done
