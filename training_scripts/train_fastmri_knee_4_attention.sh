#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python main_fastmri_corrected.py \
 --config=configs/ve/fastmri_knee_4_attention.py \
 --eval_folder=eval/fastmri_knee_4_attention \
 --mode='train'  \
 --workdir=workdir/fastmri_knee_4_attention