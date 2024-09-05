#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=icg

source activate diff_mri
cd /home/lg/score-MRI/diffusion-generalization-mri

python main_fastmri_new.py \
 --config=configs/ve/ct_3.py \
 --eval_folder=/srv/local/lg/eval/ct_3 \
 --mode='train'  \
 --workdir=/srv/local/lg/workdir/ct_3