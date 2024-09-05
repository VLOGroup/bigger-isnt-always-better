#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

python main_fastmri_new.py \
 --config=configs/ve/fastmri_knee_4.py \
 --eval_folder=eval/fastmri_knee_4 \
 --mode='train'  \
 --workdir=workdir/fastmri_knee_4