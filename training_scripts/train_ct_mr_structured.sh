#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=ivc

source activate diff_mri
cd /home/lg/ijcv/ijcv

srun python -u training_main_structured.py \
 --model_config=configs/models/mrncsn_structured.yaml \
 --data_config=configs/datasets/ct.yaml \
 --training_config=configs/training_configs.yaml \
 --sample_config=configs/samplers/pc_sampler.yaml \
 --workdir=/srv/local/lg/workdir/ct_mr_structured