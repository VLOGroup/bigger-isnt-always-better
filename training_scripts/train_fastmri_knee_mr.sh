#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=ivc
#SBATCH --mail-user=lukas.glaszner@student.tugraz.at
export CUDA_VISIBLE_DEVICES=0

source activate diff_mri
cd ..

python -u training_main.py \
 --model_config=configs/models/mrncsn_vp.yaml \
 --data_config=configs/datasets/fast_mri_corpd.yaml \
 --training_config=configs/training_configs.yaml \
 --sample_config=configs/samplers/pc_sampler.yaml \
 --workdir=/media/lukasglaszner/data/workdir/fastmri_knee_mr_vp