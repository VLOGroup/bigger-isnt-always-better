#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=l40

source activate diff_mri
cd /home/glaszner/ijcv/ijcv

srun python -u training_main.py \
 --model_config=configs/models/ncsnpp_celeba_4_small.yaml \
 --data_config=configs/datasets/celeba.yaml \
 --training_config=configs/training_configs.yaml \
 --sample_config=configs/samplers/pc_sampler.yaml \
 --workdir=/srv/local/lg/ijcv_update/celeba_4_small 