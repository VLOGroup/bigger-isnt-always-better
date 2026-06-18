#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=l40
#SBATCH --nodelist=nvcluster-node6

source activate diff_mri
cd /home/glaszner/ijcv/ijcv

srun python -u training_main.py \
 --model_config=configs/models/ncsnpp_fastmri_1_large.yaml \
 --data_config=configs/datasets/fast_mri_corpd.yaml \
 --training_config=configs/training_configs.yaml \
 --sample_config=configs/samplers/pc_sampler.yaml \
 --workdir=/srv/local/lg/ijcv_update/fastmri_knee_1_large