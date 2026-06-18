#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=ivc
#SBATCH --mail-user=lukas.glaszner@student.tugraz.at

source activate diff_mri
cd /home/lg/ijcv/ijcv

srun python -u training_main.py \
 --model_config=configs/models/mrncsn.yaml \
 --data_config=configs/datasets/ct.yaml \
 --training_config=configs/training_configs.yaml \
 --sample_config=configs/samplers/pc_sampler.yaml \
 --workdir=/srv/local/lg/workdir/ct_mr