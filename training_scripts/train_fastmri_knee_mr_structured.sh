#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=ivc
#SBATCH --nodelist=nvcluster-node1

export CUDA_VISIBLE_DEVICES=3

source activate diff_mri
cd /home/glaszner/ijcv/ijcv

python -u training_main_structured.py \
 --model_config=configs/models/mrncsn_structured.yaml \
 --data_config=configs/datasets/fast_mri_corpd.yaml \
 --training_config=configs/training_configs.yaml \
 --sample_config=configs/samplers/pc_sampler.yaml \
 --workdir=/data/glaszner/workdir/fastmri_knee_mr_structured