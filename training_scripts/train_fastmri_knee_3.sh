#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

python main_fastmri_corrected.py \
 --config=configs/ve/fastmri_knee_3.py \
 --eval_folder=eval/fastmri_knee_3 \
 --mode='train'  \
 --workdir=workdir/fastmri_knee_3