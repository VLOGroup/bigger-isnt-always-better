#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=ivc
#SBATCH --mail-user=lukas.glaszner@student.tugraz.at

source activate diff_mri
cd /home/lg/ijcv/ijcv

python -u training_main.py \
 --model_config=configs/models/msncsn.yaml \
 --data_config=configs/fast_mri_corpd.yaml \
 --training_config=configs/training_configs.yaml \
 --sample_config=configs/pc_sampler.yaml \
 --workdir=/srv/local/lg/workdir/fastmri_knee_ms