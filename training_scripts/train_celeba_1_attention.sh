#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=ivc
#SBATCH --nodelist=nvcluster-node1

source activate diff_mri
cd /home/glaszner/ijcv/ijcv

python main_fastmri_new.py \
 --config=configs/ve/celeba_1_attention.py \
 --eval_folder=/srv/local/lg/eval/celeba_1_attention \
 --mode='train' \
 --workdir=/srv/local/lg/workdir/celeba_1_attention