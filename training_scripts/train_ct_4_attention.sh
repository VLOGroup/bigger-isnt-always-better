#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=ivc

source activate diff_mri
cd /home/lg/score-MRI/diffusion-generalization-mri

python main_fastmri_new.py \
 --config=configs/ve/ct_4_attention.py \
 --eval_folder=/srv/local/lg/eval_2/ct_4_attention \
 --mode='train'  \
 --workdir=/srv/local/lg/workdir_2/ct_4_attention