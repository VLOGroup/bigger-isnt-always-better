#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

python main_fastmri_corrected.py \
 --config=configs/ve/fastmri_knee_2.py \
 --eval_folder=eval/fastmri_knee_2 \
 --mode='train'  \
 --workdir=workdir/fastmri_knee_2